import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from transformers import CLIPModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from typing import List, Optional
import math
import torch.nn.functional as F
from torch.optim import AdamW


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def resample_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
    ):
        # sort out sizes, assume square if old size not provided
        num_pos_tokens = posemb.shape[0]
        num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
            return posemb

        if old_size is None:
            hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
            old_size = hw, hw

        if num_prefix_tokens:
            posemb_prefix, posemb = posemb[:num_prefix_tokens, :], posemb[num_prefix_tokens:, :]
        else:
            posemb_prefix, posemb = None, posemb

        # do the interpolation
        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()  # interpolate needs float32
        posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
        posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        posemb = posemb.to(orig_dtype)

        # add back extra (class, etc) prefix tokens
        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb[0]], dim=0)

        return posemb

class DAugModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg        
        assert self.cfg.model.heatmap in [None, 'channel', 'add']
        
        self.logit_scale = nn.Parameter(torch.tensor(self.cfg.model.logit_scale), requires_grad=False)
        self.vision = self.build_vision_model(cfg)

        self.text = self.build_text_model(cfg)
        if self.cfg.model.heatmap == 'add':
            self.vision_hmp = self.build_vision_model(cfg)

    def build_vision_model(self, cfg):
        model = CLIPVisionModelWithProjection.from_pretrained(cfg.model.pretrain)

        if model.vision_model.embeddings.image_size != cfg.data.img_size:
            w = cfg.data.img_size // model.vision_model.embeddings.patch_size
            pos_embed = resample_pos_embed(model.vision_model.embeddings.position_embedding.weight, [w, w])
            model.vision_model.embeddings.position_embedding = nn.Embedding(num_embeddings=w*w+1, embedding_dim=model.vision_model.embeddings.embed_dim, _weight=pos_embed)
            model.vision_model.embeddings.image_size = cfg.data.img_size
            model.vision_model.embeddings.num_patches = (model.vision_model.embeddings.image_size // model.vision_model.embeddings.patch_size) ** 2
            model.vision_model.embeddings.num_positions = model.vision_model.embeddings.num_patches + 1
            model.vision_model.embeddings.register_buffer("position_ids", torch.arange(model.vision_model.embeddings.num_positions).expand((1, -1)), persistent=False)

        return model

    def build_text_model(self, cfg):
        model = CLIPTextModelWithProjection.from_pretrained(cfg.model.pretrain)

        if model.text_model.config.max_position_embeddings != cfg.data.max_text_len:
            pos_embed = resample_pos_embed(
                model.text_model.embeddings.position_embedding.weight, 
                [1, cfg.data.max_text_len], 
                [1, model.text_model.config.max_position_embeddings], 
                num_prefix_tokens=0
            )[0]
            model.text_model.embeddings.position_embedding = nn.Embedding(
                num_embeddings=cfg.data.max_text_len, 
                embedding_dim=model.text_model.config.hidden_size, 
                _weight=pos_embed)
            model.text_model.embeddings.register_buffer("position_ids", torch.arange(cfg.data.max_text_len).expand((1, -1)), persistent=False)
            model.text_model.config.max_position_embeddings = cfg.data.max_text_len
            
        return model

    def forward(self, x_img, x_hmp, y):
        if self.cfg.model.heatmap is None:
            # do not use heatmap (baseline)
            x = self.vision(x_img).image_embeds
        elif self.cfg.model.heatmap == 'channel':
            # replace the 3rd channel of medical images with the heatmap
            c = 0 # channel to replace
            x_img[:, c, :, :] = x_hmp[:, c, :, :]
            x = self.vision(x_img).image_embeds
        elif self.cfg.model.heatmap == 'add':
            # use two VIT backbones, and add their features
            x_img = self.vision(x_img).image_embeds
            x_hmp = self.vision_hmp(x_hmp).image_embeds
            x = (x_img + x_hmp) / 2
        else:
            raise NotImplementedError
                
        t_report = self.text(**y['reports']).text_embeds
        t_class = self.text(**y['prompts']).text_embeds
        
        x = x / x.norm(dim=1, keepdim=True)
        t_report = t_report / t_report.norm(dim=1, keepdim=True)
        t_class = t_class / t_class.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_text2img = torch.matmul(t_report, x.t()) * logit_scale
        logits_classes = torch.matmul(x, t_class.t()) * logit_scale
        
        if not self.training:            
            if self.cfg.model.label_bce:
                return x, t_report, logits_classes.sigmoid()
            else:
                logits_classes = logits_classes.reshape([logits_classes.shape[0], logits_classes.shape[1] // 2, 2])
                return x, t_report, logits_classes.softmax(dim=-1)[:, :, 1]
        else:
            loss = self.cfg.model.w_con_loss * clip_loss(logits_text2img)
            if self.cfg.model.label_bce:
                loss += (1-self.cfg.model.w_con_loss) * binary_cross_entropy_with_logits(logits_classes, y['labels'].float())
            else:
                logits_classes = logits_classes.reshape([-1, 2]) # N, 14, 2 -> N*14, 2
                target = y['labels'].flatten().long() # N, 14 -> N*14
                loss += (1-self.cfg.model.w_con_loss) * cross_entropy(logits_classes, target)

            return loss
        
def build_optimizer(model, cfg):
    param_optimizer = [n for n in model.named_parameters()]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.solver.decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.solver.base_lr, weight_decay=cfg.solver.decay)
    return optimizer

if __name__ == '__main__':
    # some testing code
    from omegaconf import DictConfig, OmegaConf
    from transformers import CLIPProcessor, CLIPImageProcessor, CLIPTokenizerFast
    from PIL import Image

    cfg = OmegaConf.load('conf.yaml')

    model = DAugModel(cfg).cuda()

    processor = CLIPProcessor.from_pretrained(cfg.model.pretrain, do_resize=True, size=[cfg.data.img_size, cfg.data.img_size], do_center_crop=False)

    img = Image.open('test.jpg')

    if cfg.task == 'retrieval':
        y = [
            'text 1. report of the first image.',
            'text 2. report of the second image'
        ]
        
        # simulate a batch of 2 images
        inputs = processor(text=y, images = [img, img], return_tensors="pt", padding=True, max_length=cfg.data.max_text_len, truncation=True)

    elif cfg.task == 'classification':
        y = {
            'prompts': [
                'A chest-xray image of enlarged cardiomediastinum.',
                'A chest-xray image of cardiomegaly.',
                'A chest-xray image of lung opacity',
                'A chest-xray image of lung lesion.',
                'A chest-xray image of edema',
                'A chest-xray image of consolidation',
                'A chest-xray image of pneumonia',
                'A chest-xray image of atelectasis',
                'A chest-xray image of pneumothorax',
                'A chest-xray image of pleural effusion',
                'A chest-xray image of other pleural diseases',
                'A chest-xray image of rib fracture',
                'A chest-xray image of support devices',
                'A healthy chest-xray image with no findings. A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.A healthy chest-xray image with no findings.',
            ],
            'labels':
            torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]).float(),
        }

        inputs = processor(text=y['prompts'], images = [img, img], return_tensors="pt", padding=True, max_length=cfg.data.max_text_len, truncation=True)
    
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    x = inputs.pop('pixel_values')

    if cfg.task == 'retrieval':
        y = inputs
    else:
        y['prompts'] = inputs
        y['labels'] = y['labels'].cuda()

    model.train()
    loss = model(x, None, y)

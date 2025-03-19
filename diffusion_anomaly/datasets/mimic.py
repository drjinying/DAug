import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    # this function is designed to support any customized type and to be compatible
    # with the default collate function
    ele = batch[0]
    if isinstance(ele, dict):
        return {key: collate_fn([d[key] for d in batch]) for key in ele}
    elif isinstance(ele, (tuple, list)):
        return [collate_fn(x) for x in zip(*batch)]
    else:
        if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
            if not all(b.shape == batch[0].shape for b in batch[1:]):
                assert all(len(b.shape) == len(batch[0].shape) for b in batch[1:])
                shape = torch.tensor([b.shape for b in batch])
                max_shape = tuple(shape.max(dim=0)[0].tolist())
                batch2 = []
                for b in batch:
                    if any(c < m for c, m in zip(b.shape, max_shape)):
                        b2 = torch.zeros(max_shape, dtype=b.dtype, device=b.device)
                        if b.dim() == 1:
                            b2[:b.shape[0]] = b
                        elif b.dim() == 2:
                            b2[:b.shape[0], :b.shape[1]] = b
                        elif b.dim() == 3:
                            b2[:b.shape[0], :b.shape[1], :b.shape[2]] = b
                        else:
                            raise NotImplementedError
                        b = b2
                    batch2.append(b)
                batch = batch2
        return default_collate(batch)


class MimicDataset(Dataset):
    """Dataset statistics (train)

    Visual CheXbert 
    ('No Finding' can co-exist with other findings):

                                        pathology  total  ratio  n_multi  n_single
    0                                [No Finding]  76764  48.53     7050     69714
    1  [Enlarged Cardiomediastinum, Cardiomegaly]  61883  39.12    57329      4554
    2                               [Lung Lesion]  14178   8.96    14159        19
    3           [Consolidation, Edema, Pneumonia]  56295  35.59    56112       183
    4                 [Atelectasis, Pneumothorax]  50381  31.85    49252      1129
    5           [Pleural Effusion, Pleural Other]  47025  29.73    46235       790
    6                           [Support Devices]  40841  25.82    33568      7273


    CheXpert 
    Can skip uncertain becuase the percentage is low
    {-2: "Blank", 1: "Positive", 0: "Negative", -1: "Uncertain"}:

    (uncertain as negative)             pathology  total  ratio  n_multi  n_single
    0                                [No Finding]  70076  48.50     7356     62720
    1  [Enlarged Cardiomediastinum, Cardiomegaly]  22529  15.59    15904      6625
    2                               [Lung Lesion]   4018   2.78     2160      1858
    3           [Consolidation, Edema, Pneumonia]  26665  18.45    16884      9781
    4                 [Atelectasis, Pneumothorax]  29000  20.07    20515      8485
    5           [Pleural Effusion, Pleural Other]  27548  19.06    21665      5883
    6                           [Support Devices]  29754  20.59    26355      3399

    (skip uncertain)                    pathology  total  ratio  n_multi  n_single
    0                                [No Finding]  70040  54.45     7356     62684
    1  [Enlarged Cardiomediastinum, Cardiomegaly]  18640  14.49    13473      5167
    2                               [Lung Lesion]   3140   2.44     1684      1456
    3           [Consolidation, Edema, Pneumonia]  22499  17.49    14363      8136
    4                 [Atelectasis, Pneumothorax]  23522  18.29    16854      6668
    5           [Pleural Effusion, Pleural Other]  21445  16.67    17317      4128
    6                           [Support Devices]  24528  19.07    22759      1769

    """
    def __init__(self, cfg, tokenizer, split) -> None:
        super().__init__()

        assert split in ['train', 'test', 'validate', 'train_val']
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize([cfg.model.backbone.img_size, cfg.model.backbone.img_size]),
            transforms.ToTensor(),
        ])

        data = torch.load(os.path.join(cfg.data.data_root, cfg.data.anno_path))
        self.meta = data['meta_data']
        if split == 'train_val':
            data = list(data['train'].values()) + list(data['validate'].values())
        else:
            data = list(data[split].values())

        self.image_info = []
        self.gt_report = []
        self.gt_report_split = []
        self.gt_prompt_cls = []
        self.condition_class = []
        self.info_dict = []

        if cfg.data.mode == 'by_pathology':
            input_class_names = self.meta['condition_names']
            super_classes = [self.cfg.data.neg_classes] + self.cfg.data.pos_classes
            super_class_idx = [[input_class_names.index(c) for c in names] for names in super_classes]

        is_master = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        import random
        random.seed(1351)
        random.shuffle(data)

        for s in tqdm(data, desc='loading %s' % split, disable=not is_master):
            if cfg.data.mode == 'by_pathology':
                conditions = []
                conds_ori = np.array(s['pred_%s' % cfg.data.class_provider])
                for i in range(len(super_classes)):
                    conditions.append(conds_ori[super_class_idx[i]].max())                
                if max(conditions) < 1:
                    continue                
                if cfg.data.class_provider == 'chexpert':
                    conditions = np.array(conditions, dtype=np.int32)
                    conditions[conditions == -2] = 0 # treat blank as negative
                    if cfg.data.chexpert_uncertain == 'skip':
                        if -1 in conditions:
                            continue
                    else:                        
                        conditions[conditions == -1] = 0                        
                    
                    conditions = conditions.tolist()
            else:
                conditions = s['pred_%s' % cfg.data.class_provider]

            # '', 'AP', 'AP AXIAL', 'AP LLD', 'AP RLD', 'LAO', 'LATERAL', 'LL', 'LPO', 'PA', 'PA LLD', 'RAO', 'SWIMMERS', 'XTABLE LATERAL'
            views = np.array(s['images']['view_position'])
            idx_pa = np.flatnonzero(views == 'PA').tolist()
            idx_ap = np.flatnonzero(np.logical_or(views == 'AP', views == 'AP AXIAL')).tolist()
            idx_lat = np.flatnonzero(views == 'LATERAL').tolist()
            idx_empty = np.flatnonzero(views == '').tolist()
            idx_unrecognized = list(filter(lambda i: views[i] not in ['PA', 'AP', 'LATERAL', ''], range(len(views))))

            img_idx = {
                'by_study': [s['images']['prefered_idx'][0]], # one image per study
                'by_image': list(range(len(s['images']['id']))), # all images
                'by_study_repeated': list(range(len(s['images']['id']))), # return names of all_images, but actually load the default image
                'by_pathology': idx_pa + idx_ap,
                'by_view': s['images']['prefered_idx'], # all AP/PA images
            }[cfg.data.mode]

            for idx in img_idx:
                self.image_info.append({
                    'path': os.path.join(cfg.data.data_root, cfg.data.img_dir, s['path'], '%s.jpg' % s['images']['id'][idx]),
                    'size_hw': s['images']['size_hw'][idx],
                    'view_position': s['images']['view_position'][idx],
                    'patient_orientation': s['images']['patient_orientation'][idx]
                })                    
                self.gt_report.append(s['gt_report'])
                self.gt_report_split.append(s['gt_report_split'])
                self.gt_prompt_cls.append(s['gt_report_split_cls'])
                self.condition_class.append(conditions)
                self.info_dict.append(s)

        self.condition_class = np.array(self.condition_class, dtype=np.int32)

    def pathology_mapping(self, input_labels):
        input_class_names = self.meta['condition_names']
        super_classes = [self.cfg.data.neg_classes] + self.cfg.data.pos_classes
        super_class_idx = [[input_class_names.index(c) for c in names] for names in super_classes]
        output_labels = []
        for i in range(len(super_classes)):
            output_labels.append((np.array(input_labels)[super_class_idx[i]].max() == 1) * 1)
        return output_labels

    def stat_pathologies(self):
        import pandas as pd
        stats = pd.DataFrame(columns=['pathology', 'total', 'ratio', 'n_multi', 'n_single'])

        for i, pathology in enumerate([self.cfg.data.neg_classes] + self.cfg.data.pos_classes):
            idx = self.condition_class[:, i] == 1
            stats.loc[len(stats.index)] = [
                pathology, 
                idx.sum(),
                idx.sum() / self.condition_class.shape[0] * 100,
                (self.condition_class[idx].sum(axis=1) > 1).sum(),
                (self.condition_class[idx].sum(axis=1) == 1).sum(),
            ]

        with pd.option_context('display.precision', 2,):
            print(stats)
    
    def get_pos_weights(self):
        w = (self.condition_class == 0).sum(axis=0) / self.condition_class.sum(axis=0)
        return torch.tensor(w)

    def __len__(self):
        return len(self.image_info)
    
    def __getitem__(self, index):
        if self.cfg.data.mode == 'by_study_repeated':
            # load the default image N times, N is number of all images in the same study
            # this is used for direct comparision with baselines, where each image is a standalone sample            
            img_idx_in_study = self.info_dict[index]['images']['prefered_idx'][0]
            img_fp = os.path.join(self.cfg.data.data_root, self.cfg.data.img_dir, self.info_dict[index]['path'], '%s.jpg' % self.info_dict[index]['images']['id'][img_idx_in_study])
            x = Image.open(img_fp).convert('RGB')
        else:
            x = Image.open(self.image_info[index]['path']).convert('RGB')
        
        x = self.transform(x)
        x = (x - x.min()) / (x.max() - x.min())
        
        y = {
            'gt_report': self.gt_report[index],
            'gt_report_split': self.gt_report_split[index],
            'gt_prompt_cls': self.gt_prompt_cls[index],
            'condition_class': torch.tensor(self.condition_class[index]),
            'view_position': self.image_info[index]['view_position'],
            'path': self.image_info[index]['path'],
            'index': index
        }

        caption_target = self.tokenizer(
            y['gt_report_split'],             
            add_special_tokens=False, 
            truncation=True, 
            padding='max_length',
            max_length=self.cfg.model.captioner.max_seq_len)
        
        caption_tokens = torch.zeros([self.cfg.model.prompter.max_seq_len, self.cfg.model.captioner.max_seq_len], dtype=torch.long)        
        caption_tokens[:len(y['gt_prompt_cls'])] = torch.tensor(caption_target['input_ids'])[:self.cfg.model.prompter.max_seq_len, :self.cfg.model.captioner.max_seq_len]
        caption_length = torch.zeros(self.cfg.model.prompter.max_seq_len, dtype=torch.int)
        caption_length[:len(y['gt_prompt_cls'])] = torch.tensor(caption_target['attention_mask'])[:self.cfg.model.prompter.max_seq_len].sum(dim=1).clamp(max=self.cfg.model.captioner.max_seq_len) # length of non-padding tokens per sentence

        prompt_tokens = torch.zeros(self.cfg.model.prompter.max_seq_len, dtype=torch.long)
        prompt_tokens[:len(y['gt_prompt_cls'])] = torch.tensor([600 + i for i in y['gt_prompt_cls']])[:self.cfg.model.prompter.max_seq_len]

        y['gt_report_length'] = caption_length
        y['gt_report_tokens'] = caption_tokens
        y['gt_prompt_tokens'] = prompt_tokens

        info = self.info_dict[index]

        return x, y, info
    

def build_mimic_dataloader(cfg, tokenizer, split, dist=True):
    dataset = MimicDataset(cfg, tokenizer, split)    
    if dist:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, sampler=sampler, num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=(split!='test'), num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    pos_weight = dataset.get_pos_weights()
    return loader, pos_weight
    
if __name__ == '__main__':
    from omegaconf import DictConfig, OmegaConf
    cfg = OmegaConf.load('/home/azureuser/promed/lpnet/configs/default.yaml')
    cfg.data.class_provider = ['visual_chexbert', 'chexpert'][0]
    cfg.data.chexpert_uncertain = 'skip' # 'skip' or 'negative'
    cfg.data.num_workers = 1
    
    assert cfg.data.class_provider in ['visual_chexbert', 'chexpert']
    assert cfg.data.chexpert_uncertain in ['skip', 'negative']

    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    loader, pos_weight = build_mimic_dataloader(cfg, tokenizer, 'test', False)
    for x, y, info in loader:
        pass

    dataset = MimicDataset(cfg, 'train')
    dataset.stat_pathologies()

    import random
    import json

    conds = np.array(['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                'Support Devices', 'No Finding'])
    sup_conds = np.array([cfg.data.neg_classes] + cfg.data.pos_classes, dtype=object)

    # while True:
    #     i = random.randint(0, len(dataset))
    #     x, y, info = dataset[i]
    #     x.save('test.jpg')
    #     print(x.size)
    #     print(y['gt_report'])
    #     # print('Classes: ')
    #     # print(conds[y['_condition_class'].astype(bool)])
    #     print('Superclasses: ')
    #     print(sup_conds[y['condition_class'].astype(bool)])
    #     break
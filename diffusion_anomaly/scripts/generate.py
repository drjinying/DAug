"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
import torch
import sys
sys.path.append("..")
sys.path.append(".")
import json
import random
from guided_diffusion import dist_util2 as dist_util
from datasets.mimic import build_mimic_dataloader
import torch.nn.functional as F
import numpy as np
import copy
import torch as th
from guided_diffusion import logger
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import matplotlib.cm as cm
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from itertools import chain
from transformers import BertTokenizerFast
from tqdm import tqdm

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def main():
    with hydra.initialize(config_path='../datasets', version_base=None):
        cfg = hydra.compose(config_name='conf')                

    args = create_argparser().parse_args()

    cfg.blob_root = args.blob_root
    OmegaConf.resolve(cfg)
    
    args.model_path = cfg.diff_model_path
    args.classifier_path = cfg.classifier_path

    print(cfg.blob_root)
    
    os.environ['NCCL_DEBUG'] = 'VERSION'
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        master_ip = os.environ['MASTER_IP'] if 'MASTER_IP' in os.environ else os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
        master_port = '6688'

        master_uri = "tcp://%s:%s" % (master_ip, master_port)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=master_uri,
            rank=world_rank,
            world_size=world_size
        )
    else:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()

    torch.cuda.set_device(local_rank)

    #dist_util.setup_dist()
    logger.configure()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    datal_train, pos_weight = build_mimic_dataloader(
        cfg,
        tokenizer,
        'train'
    )

    #data = chain(iter(datal_test), iter(datal_train), iter(datal_val))
    data = iter(datal_train)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
   
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")

    classifier_args = copy.deepcopy(args)
    classifier_args.num_classes = 1 + len(cfg.data.pos_classes)
    classifier = create_classifier(**args_to_dict(classifier_args, classifier_defaults().keys()))
    classifier.load_state_dict(torch.load(args.classifier_path)['state_dict'])        

    print('loaded classifier')
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)

    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None, fp=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            if y.item() != 0:
                logits = logits[:, 1:]
                y -= 1
            log_probs = F.log_softmax(logits)
            selected = log_probs[range(len(logits)), y.long().view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale

    def cond_fn_neg(x, t, y=None, fp=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)    
            log_probs = F.log_softmax(logits)
            selected = log_probs[range(len(logits)), y.long().view(-1)]
            a= - th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale
        
    def model_fn_0(x, t, y=None):
        assert y is not None
        return model(x, t, torch.tensor([0]).cuda())
    
    def model_fn_1(x, t, y=None):
        assert y is not None
        return model(x, t, torch.tensor([1]).cuda())

    logger.log("sampling...")

    cnt_exist = 0

    for batch in tqdm(data, total=len(datal_train)):
        x, y, info = batch
        x = x[:, 0, :, :]
        x = x[:, None, :, :]

        # cond = {'y': y['condition_class'][:, 0].cuda()}
        cond = {'y': torch.tensor([0]).cuda()}
        # print(cond)
        #cond = {'y': torch.tensor([0]).cuda()}
        x = x.cuda()

        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )

        _model_fn = model_fn_1 # go to normal; cleaner output
        noise_level = 500

        dst_dir = os.path.dirname(y['path'][0].replace('mimic-cxr-jpg-2.0.0', cfg.data.output_dir))

        # if os.path.exists(dst_dir):
        #     cnt_exist += 1

        # continue

        os.makedirs(dst_dir, exist_ok=True)

        cls_name = [
            'nof',
            'car',
            'lsn',
            'con',
            'atl',
            'plr',
            'dev'                        
        ]

        for cls in cls_name:            
            id = os.path.basename(y['path'][0]).split('.')[0]
            mode = '%s%s' % ('+' if cls == 'nof' else '-', cls)
            outfp = os.path.join(dst_dir, '%s_%s.npy' % (id, mode))
            img_out_fp = os.path.join(dst_dir, '%s_%s.jpg' % (id, mode))

            if os.path.exists(outfp) and os.path.exists(img_out_fp):
                break

            _cond_fn = cond_fn if cls == 'nof' else cond_fn_neg

            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    sample, x_noisy, org = sample_fn(
                        _model_fn,
                        (args.batch_size, 4, args.image_size, args.image_size), (x, cond), org=(x, cond),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=cond,
                        cond_fn=_cond_fn,
                        device=dist_util.dev(),
                        noise_level=noise_level
                    )

            diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
            diff=np.array(diff.cpu())                        
            np.save(outfp, diff)

            heatmap = cm.hot(diff)            
            Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8)).save(img_out_fp)        

    print(len(data))
    print(cnt_exist)
    print(cnt_exist / len(data))

def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,        
        batch_size=1,        
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='mimic-cxr',
        num_channels=128,
        class_cond=True,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=True,
        use_scale_shift_norm=False,
        attention_resolutions="16",
        diffusion_steps=1000,
        noise_schedule='linear',
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        classifier_attention_resolutions="32,16,8",
        classifier_depth=4,
        classifier_width=32,
        classifier_pool='attention',
        classifier_resblock_updown=True,
        classifier_use_scale_shift_norm=True,        
        num_samples=1,
        timestep_respacing='ddim1000',
        use_ddim=True,
        blob_root='/yjblob'
    )

    defaults2 = copy.deepcopy(defaults)

    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(defaults2)
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    torch.manual_seed(666)    
    random.seed(666)
    np.random.seed(666)    

    main()

# OMP_NUM_THREADS=5 torchrun --standalone --nnodes=1 --nproc_per_node=8 generate.py


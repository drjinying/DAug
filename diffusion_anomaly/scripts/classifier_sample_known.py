"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
from visdom import Visdom
import torch
import sys
sys.path.append("..")
sys.path.append(".")
import json
import random
from guided_diffusion import dist_util2 as dist_util
from lpnet.datasets.mimic import build_mimic_dataloader
from guided_diffusion.bratsloader import BRATSDataset
import torch.nn.functional as F
import numpy as np
import copy
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import logger
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import matplotlib.cm as cm
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
    args = create_argparser().parse_args()

    print(args.i_cls)
    torch.cuda.set_device(args.i_cls)    

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.dataset=='brats':
      ds = BRATSDataset(args.data_dir, test_flag=True)
      datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    
    elif args.dataset=='chexpert':
     data = load_data(
         data_dir=args.data_dir,
         batch_size=args.batch_size,
         image_size=args.image_size,
         class_cond=True,
     )
     datal = iter(data)

    assert args.dataset == 'mimic-cxr'
    from omegaconf import DictConfig, OmegaConf
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../datasets/conf.yaml'))
    cfg.batch_size = args.batch_size
    datal, pos_weight = build_mimic_dataloader(
        cfg,
        'train'
    )
    data = iter(datal)
    print('dataset is mimic-cxr')
   
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")

    classifier_args = copy.deepcopy(args)
    classifier_args.num_classes = 1 + len(cfg.pos_classes)
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
        
    # def model_fn(x, t, y=None):
    #     assert y is not None
    #     return model(x, t, y if args.class_cond else None)

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

    for batch in datal:
        x, y, info = batch
        # cond = {'y': y['condition_class'][:, 0].cuda()}
        cond = {'y': torch.tensor([args.i_cls]).cuda()}
        # print(cond)
        #cond = {'y': torch.tensor([0]).cuda()}
        x = x.cuda()

        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )

        for diff_cls in [0, 1]:
            _model_fn = [model_fn_0, model_fn_1][diff_cls]
            for direction in ['+']:
                _cond_fn = {
                    '+': cond_fn,
                    #'-': cond_fn_neg
                }[direction]
                for noise_level in [500]:
                    start = th.cuda.Event(enable_timing=True)
                    end = th.cuda.Event(enable_timing=True)
                    start.record()
                    sample, x_noisy, org = sample_fn(
                        _model_fn,
                        (args.batch_size, 4, args.image_size, args.image_size), (x, cond), org=(x, cond),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=cond,
                        cond_fn=_cond_fn,
                        device=dist_util.dev(),
                        noise_level=noise_level
                    )
                    end.record()
                    th.cuda.synchronize()
                    th.cuda.current_stream().synchronize()

                    print('time for 1000', start.elapsed_time(end))

                    dst_dir = os.path.join('/yjblob/medical/output/t3softmax_woNF_noise_%03d/%d/' % (noise_level, info['study_id']))
                    os.makedirs(dst_dir, exist_ok=True)

                    img = torchvision.transforms.functional.to_pil_image(x[0, 0])
                    img.save(os.path.join(dst_dir, '%d.0_ori.jpg' % info['study_id']))

                    cls_name = [
                        'nof',
                        'car',
                        'lsn',
                        'con',
                        'atl',
                        'plr',
                        'dev'                        
                    ][args.i_cls]
                    img2 = torchvision.transforms.functional.to_pil_image(sample[0, 0])
                    img2.save(os.path.join(dst_dir, '%d.%d_%s%s.d_%d.jpg' % (info['study_id'], args.i_cls, direction, cls_name, diff_cls)))

                    with open(os.path.join(dst_dir, 'gt.json'), 'w') as f:
                        json.dump(str(np.array([cfg.neg_classes] + cfg.pos_classes)[y['condition_class'][0].bool()].tolist()), f, indent=4)

                    diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
                    diff=np.array(diff.cpu())

                    heatmap = cm.hot(diff)
                    heatmap = Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8))
                    heatmap.save(os.path.join(dst_dir, '%d.%d_%s%s.d_%d.dif.jpg' % (info['study_id'], args.i_cls, direction, cls_name, diff_cls)))

                    img = img.convert('RGB')
                    diff = (diff - diff.min()) / (diff.max() - diff.min())
                    heatmap = cm.hot(diff)
                    heatmap = Image.fromarray((heatmap[:, :, :3] * 255).astype(np.uint8))
                    combined_image = Image.blend(img, heatmap, 0.8)
                    combined_image.save(os.path.join(dst_dir, '%d.%d_%s%s.d_%d.vis.jpg' % (info['study_id'], args.i_cls, direction, cls_name, diff_cls)))

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        i_cls=0
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    torch.manual_seed(666)    
    random.seed(666)
    np.random.seed(666)    

    main()


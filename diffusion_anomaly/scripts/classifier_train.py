"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import sys
from torch.autograd import Variable
from diffusion_anomaly.guided_diffusion import dist_util2
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import blobfile as bf
import torch as th
os.environ['OMP_NUM_THREADS'] = '8'
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from visdom import Visdom
import numpy as np
viz = Visdom(port=8850)
loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='classification loss'))
val_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='validation loss'))
acc_window= viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='mAP', title='mAP'))

from guided_diffusion import logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import visualize
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
sys.path.append("./datasets")
from mimic import build_mimic_dataloader
from sklearn.metrics import average_precision_score
import pandas as pd
import random

def main():
    from omegaconf import DictConfig, OmegaConf
    cfg = OmegaConf.load('.//datasets/conf.yaml')

    args = create_argparser().parse_args()
    args.num_classes = 1 + len(cfg.pos_classes)

    dist_util2.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys()),
    )
    model.to(dist_util2.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion, maxt=1000
        )

    resume_step = 0

    ###############################################
    args.resume_checkpoint = '/yjblob/medical/ckpt/diff_classifier_multi/ep_014.pt'
    ###############################################

    if args.resume_checkpoint:
        resume_step = 15 #parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            ckpt = torch.load(args.resume_checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])

            # model.load_state_dict(
            #     dist_util.load_state_dict(
            #         args.resume_checkpoint, map_location=dist_util.dev()
            #     )
            # )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util2.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )


    logger.log("creating data loader...")

    # if args.dataset == 'brats':
    #     ds = BRATSDataset(args.data_dir, test_flag=False)
    #     datal = th.utils.data.DataLoader(
    #         ds,
    #         batch_size=args.batch_size,
    #         shuffle=True)
    #     data = iter(datal)

    # elif args.dataset == 'chexpert':
    #     data = load_data(
    #         data_dir=args.data_dir,
    #         batch_size=1,
    #         image_size=args.image_size,
    #         class_cond=True,
    #     )
    #     print('dataset is chexpert')

    # elif args.dataset == 'mimic-cxr':
    #     datal, pos_weight = build_mimic_dataloader(
    #         cfg,
    #         'train'
    #     )
    #     data = iter(datal)
    #     print('dataset is mimic-cxr')

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location='cpu')
        opt.load_state_dict(ckpt['opt'])

        # opt_checkpoint = bf.join(
        #     bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        # )
        # logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        # opt.load_state_dict(
        #     dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        # )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, step, cfg, prefix="train"):
        if args.dataset=='brats':
            batch, extra, labels,_ , _ = next(data_loader)
            print('IS BRATS')

        elif  args.dataset=='chexpert':
            batch, extra = next(data_loader)
            labels = extra["y"].to(dist_util2.dev())
            print('IS CHEXPERT')
        
        elif args.dataset=='mimic-cxr':
            batch, y, info = next(data_loader)
            labels = y['condition_class']
            #print('IS MIMIC-CXR')

        # print('labels', labels)
        batch = batch.to(dist_util2.dev())
        labels= labels.to(dist_util2.dev())
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util2.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util2.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
          
            sub_batch = Variable(sub_batch, requires_grad=True)
            logits = model(sub_batch, timesteps=sub_t)
         
            #loss = F.cross_entropy(logits, sub_labels, reduction="none")
            loss = F.binary_cross_entropy_with_logits(logits, sub_labels.float(), pos_weight=pos_weight.to(dist_util2.dev()), reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()

            losses['pred'] = logits.sigmoid().detach().cpu().numpy()
            losses['gt'] = sub_labels.detach().cpu().numpy()            

            # losses[f"{prefix}_acc@1"] = compute_top_k(
            #     logits, sub_labels, k=1, reduction="none"
            # )
            # losses[f"{prefix}_acc@2"] = compute_top_k(
            #     logits, sub_labels, k=2, reduction="none"
            # )
            # print('acc', losses[f"{prefix}_acc@1"])

            # print('ap', ap)
            log_loss_dict(diffusion, sub_t, losses)

            loss = loss.mean()
            if prefix=="train":
                viz.line(X=th.ones((1, 1)).cpu() * step, Y=th.Tensor([loss]).unsqueeze(0).cpu(),
                     win=loss_window, name='loss_cls',
                     update='append')

            # else:
            #     output_idx = logits[0].argmax()
            #     print('outputidx', output_idx)
            #     output_max = logits[0, output_idx]
            #     print('outmax', output_max, output_max.shape)
            #     output_max.backward()
            #     saliency, _ = th.max(sub_batch.grad.data.abs(), dim=1)
            #     print('saliency', saliency.shape)
            #     viz.heatmap(visualize(saliency[0, ...]))
            #     viz.image(visualize(sub_batch[0, 0,...]))
            #     viz.image(visualize(sub_batch[0, 1, ...]))
            #     th.cuda.empty_cache()


            if loss.requires_grad and prefix=="train":
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

        return losses

    for epoch in range(resume_step, args.iterations):
        datal, pos_weight = build_mimic_dataloader(
            cfg,
            'train'
        )
        data = iter(datal)
        model.train()

        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (epoch - resume_step) / (args.iterations - resume_step))

        for step in tqdm(range(len(data)), desc='Epoch %d' % epoch):
    #for step in range(args.iterations - resume_step):
            # logger.logkv("step", step + resume_step)
            # logger.logkv(
            #     "samples",
            #     (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
            # )
            if args.dataset == 'mimic-cxr':
                losses = forward_backward_log(data, step + resume_step, cfg)
            else:
                try:
                    losses = forward_backward_log(data, step + resume_step, cfg)
                except:
                    data = iter(datal)
                    losses = forward_backward_log(data, step + resume_step, cfg)

            # correct+=losses["train_acc@1"].sum()
            # total+=args.batch_size
            # acctrain=correct/total

            mp_trainer.optimize(opt)
            
            if not step % args.log_interval:
                print('epoch %d, step %d' % (epoch, step))
                logger.dumpkvs()

            dist.barrier()

        if dist.get_rank() == 0:
            logger.log("saving model...")
            save_model(mp_trainer, opt, epoch, cfg)

        if epoch % 1 == 0:
            datal, pos_weight = build_mimic_dataloader(
                cfg,
                'test'
            )
            data = iter(datal)
            
            pred = []
            gt = []
            model.eval()
            for step in tqdm(range(len(data)), desc='testing'):
                with torch.no_grad():
                    loss_dict = forward_backward_log(data, step, cfg, prefix="test")
                pred.append(loss_dict['pred'])
                gt.append(loss_dict['gt'])

            pred = np.vstack(pred)
            gt = np.vstack(gt)

            ap = average_precision_score(gt, pred, average=None)
            mAP = ap.mean()

            df = pd.DataFrame(columns=[cfg.neg_classes] + cfg.pos_classes + ['mAP'])
            df.loc[0] = ap.tolist() + [mAP]
            print(df)

            viz.line(X=th.ones((1, 1)).cpu() * epoch, Y=torch.tensor([mAP]).unsqueeze(0), win=acc_window, name='mAP', update='append')


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    print('Set lr %f' % lr)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, epoch, cfg):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if dist.get_rank() == 0:
        th.save(
            {
                'config': cfg,
                'state_dict': mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
                'opt': opt.state_dict(),
            },
            os.path.join(cfg.ckpt_dir, f"ep_{epoch:03d}.pt"),
        )

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,
        dataset='brats',
        num_classes=2
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    torch.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    main()

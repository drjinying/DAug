"""
Train a diffusion model on images.
"""
import sys
import argparse
import torch as th

sys.path.append("..")
sys.path.append(".")
#from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import logger
from guided_diffusion import dist_util2 as dist_util
#from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch.distributed as dist
import os

def main():
    args = create_argparser().parse_args()

    from omegaconf import DictConfig, OmegaConf
    cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '../datasets/conf.yaml'))   
    cfg.data_root = os.path.join(args.data_dir, 'medical')

    dist_util.setup_dist()
    dist.barrier()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")

    # if args.dataset == 'brats':
    #     ds = BRATSDataset(args.data_dir, test_flag=False)
    #     datal = th.utils.data.DataLoader(
    #         ds,
    #         batch_size=args.batch_size,
    #         shuffle=True)
    #    # data = iter(datal)

    # elif args.dataset == 'chexpert':
    #     datal = load_data(
    #         data_dir=args.data_dir,
    #         batch_size=1,
    #         image_size=args.image_size,
    #         class_cond=True,
    #     )
    #     print('dataset is chexpert')


    logger.log("training...")
    TrainLoop(
        cfg=cfg,
        model=model,
        diffusion=diffusion,
        data=None,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":    
    main()

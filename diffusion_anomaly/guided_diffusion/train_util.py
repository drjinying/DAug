import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
#from visdom import Visdom

import sys
from datasets.mimic import build_mimic_dataloader
from tqdm import tqdm
import torch
import numpy as np

import wandb

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        cfg,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
    ):        
        self.batch_size = batch_size
        self.cfg = cfg
        self.cfg.batch_size = batch_size
        self.epochs = self.cfg.epochs
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset=dataset
        # self.iterdatal = iter(data)        
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        ########################################3
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        print('World size: %d' % dist.get_world_size())

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        ########################################                
        if self.resume_step:
            print('Loading models')
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        #resume_checkpoint = os.path.join(self.cfg.data_root, 'ckpt/diff/mimic2update000011.pt')

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        #main_checkpoint =  find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = os.path.join(self.cfg.data_root, 'ckpt/diff/ema2update_0.9999_000011.pt') #find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        # main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        # opt_checkpoint = bf.join(
        #     bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        # )
        opt_checkpoint = os.path.join(self.cfg.data_root, 'ckpt/diff/optmimic2update000011.pt')
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        if dist.get_rank() == 0:
            wandb.login(key='925b95b541babecaf9f3c1b3ee75be8b6d57af2f')
            wandb.init(
                project='medical',
                group='ddpm',
                name='diffusion',
            )

            # viz = Visdom(port=8850)
            # train_loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='step', ylabel='mse', title='train mse'))
            # test_loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='mse', title='test mse'))

        for epoch in range(self.resume_step, self.epochs):
            self.step = epoch

            datal, pos_weight = build_mimic_dataloader(
                self.cfg,
                'test',
                dist=True
            )
            datal.sampler.set_epoch(epoch)

            self.ddp_model.train()

            mse = []
            for batch in tqdm(datal, desc='epoch %d' % epoch, disable=(dist.get_rank() != 0)):
                x, y, info = batch
                cond = {'y': y['condition_class'][:, 0]}
                lossmse, _ = self.run_step(x, cond)
                mse.append(lossmse.item())

                if dist.get_rank() == 0:
                    wandb.log({'train_mse': np.mean(mse)}, step=(len(mse) + epoch * len(datal)))

                    # viz.line(X=th.ones((1, 1)).cpu() * (len(mse) + epoch * len(datal)), Y=th.Tensor([np.mean(mse)]).unsqueeze(0).cpu(),
                    #     win=train_loss_window, name='mse',
                    #     update='append')

            #logger.dumpkvs()            
            self.save()

            datal, pos_weight = build_mimic_dataloader(
                self.cfg,
                'test',
                dist=True
            )

            mse = []
            self.ddp_model.eval()
            for batch in tqdm(datal, desc='test', disable=(dist.get_rank() != 0)):
                x, y, info = batch
                cond = {'y': y['condition_class'][:, 0]}
                with torch.no_grad():
                    lossmse,  sample = self.forward_backward(x, cond)
                    mse.append(lossmse.detach().item())
            if dist.get_rank() == 0:
                print(np.mean(mse))
                wandb.log({'test_mse': np.mean(mse)}, step=(len(mse) + epoch * len(datal)))

                # viz.line(X=th.ones((1, 1)).cpu() * epoch, Y=th.Tensor([np.mean(mse)]).unsqueeze(0).cpu(),
                #      win=test_loss_window, name='mse',
                #      update='append')

    def run_loop_bak(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            if self.dataset=='brats':
                try:
                    batch, cond, label = next(self.iterdatal)
                except:
                    self.iterdatal = iter(self.datal)
                    batch, cond, label, _, _ = next(self.iterdatal)
            elif self.dataset=='chexpert':
                batch, cond = next(self.datal)
                cond.pop("path", None)

            elif self.dataset=='mimic-cxr':
                batch, y, info = next(self.datal)
                labels = y['condition_class']

            self.run_step(batch, cond)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        lossmse,  sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return lossmse,  sample

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            #print('micro', micro.shape)
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
       
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]

            loss = (losses["loss"] * weights).mean()

            lossmse = (losses["mse"] * weights).mean().detach()            

            if self.ddp_model.training:
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )
                self.mp_trainer.backward(loss)                

            return lossmse.detach(),  sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        out_dir = os.path.join(self.cfg.data_root, 'ckpt/diff/')
        os.makedirs(out_dir, exist_ok=True)
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)            
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"mimic2update_1e5_{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema2update_1e5_{rate}_{(self.step+self.resume_step):06d}.pt"
                print('filename', filename)
                with bf.BlobFile(bf.join(out_dir, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(out_dir, f"optmimic2update_1e5_{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        if type(values) != th.Tensor:
            continue
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss.mean())

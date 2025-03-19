import os
from dataclasses import dataclass
import parse
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


@dataclass
class DaugDataCollatorWithPadding:
    """
    Modified from DataCollatorWithPadding
    """

    def __call__(self, features):
        x = torch.stack([x[0] for x in features])
        x_diff = torch.stack([x[1] for x in features])
        y = {
            'labels': torch.stack([x[2].pop('labels') for x in features]),
            'reports': [x[2].pop('reports') for x in features],
            'img_ids': [x[2].pop('img_ids') for x in features]
        }
        return x, x_diff, y
    
def recursive_to_device(d, device, **kwargs):
    if isinstance(d, tuple) or isinstance(d, list):
        return [recursive_to_device(x, device, **kwargs) for x in d]
    elif isinstance(d, dict):
        return dict((k, recursive_to_device(v, device)) for k, v in d.items())
    elif isinstance(d, torch.Tensor) or hasattr(d, 'to'):        
        return d.to(device, **kwargs)
    else:
        return d
    
def wandb_init(cfg: DictConfig, save_dir=None):
    wandb.init(
        project='daug',
        group=cfg.exp_group,
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_code=False
    )
    
    save_dir = cfg.ckpt_dir if save_dir is None else save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    OmegaConf.save(config=cfg, f=os.path.join(save_dir, 'conf.yaml'))


class MovingAverage:
    def __init__(self):
        self.cnt = 0
        self.val = 0
        self.total_val = 0
        self.mean = 0

    def add_val(self, val):
        self.cnt += 1
        self.total_val += val
        self.val = val
        self.mean = self.total_val / self.cnt

def save_checkpoint(epoch, model, optimizer, ckpt_dir):
    print('Saving checkpoint...')
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    out_fp = os.path.join(ckpt_dir, 'ckpt_%04d.pt' % epoch)
    torch.save(model.state_dict(), out_fp)
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict()
    # }, out_fp)

    print("Saved checkpoint to %s" % out_fp)


def load_checkpoint(ckpt_dir, device='cpu'):
    print('Searching for checkpoints in ' + ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        print('Not loading from previous checkpoint')
        return None

    ckpt_files = os.listdir(ckpt_dir)
    ckpt_files = list(filter(lambda x: x.startswith('ckpt_'), ckpt_files))

    if len(ckpt_files) == 0:
        print('Not loading from previous checkpoint')        
        return None

    ckpt_epochs = [parse.parse('ckpt_{epoch:d}.pt', x)['epoch'] for x in ckpt_files]
    ckpt_files = [f for e, f in sorted(zip(ckpt_epochs, ckpt_files), reverse=True)]
    latest_fp = os.path.join(ckpt_dir, ckpt_files[0])
    print('Loading checkpoint from ' + latest_fp)

    return torch.load(latest_fp, map_location=torch.device(device))
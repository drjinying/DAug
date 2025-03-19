from main import DistributedTrainer
from rich import pretty, print
import os
import hydra
from model import DAugModel
import torch
from utils import load_checkpoint
import sys
from torch.nn.parallel import DistributedDataParallel
from mimic import build_mimic_daug_dataloader

class DistributedTester(DistributedTrainer):
    def test(self):
        model = DAugModel(self.cfg)
        ckpt = load_checkpoint(self.cfg.ckpt_dir)
        model.load_state_dict(ckpt)
        
        model.to(self.local_rank)
        model = DistributedDataParallel(model, device_ids=[self.local_rank], find_unused_parameters=False)

        train_dataloader, class_weights = build_mimic_daug_dataloader(self.cfg, 'train_val', True)
        test_dataloader, _ = build_mimic_daug_dataloader(self.cfg, 'test', True)
        self.inference_cls(model, test_dataloader, train_dataloader)    

if __name__ == '__main__':
    pretty.install()
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    with hydra.initialize(config_path='./configs', version_base=None):
        cfg = hydra.compose(config_name="conf", overrides=['+abl=cls_hcontra'])
        cfg.wandb = False
        
        os.environ['NCCL_DEBUG'] = 'VERSION'
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

        local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        distributed_tester = DistributedTester(cfg, local_rank)
        distributed_tester.test()


# torchrun --standalone --nnodes=1 --nproc_per_node=8 test.py
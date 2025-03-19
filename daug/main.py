import os
import sys
import json

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers import get_scheduler, CLIPTokenizerFast

from mimic import build_mimic_daug_dataloader
from model import DAugModel, build_optimizer
from utils import MovingAverage, save_checkpoint, wandb_init, recursive_to_device, load_checkpoint
from sklearn.metrics import roc_auc_score

class DistributedTrainer:
    def __init__(self, cfg, local_rank) -> None:
        self.cfg = cfg
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = local_rank
        self.local_gpus = torch.cuda.device_count()
        self.global_master = self.global_rank == 0

        self.nworkers = os.cpu_count() // self.local_gpus - 1
        self.prefetch = max(cfg.data.batch_size // self.nworkers, cfg.data.batch_size)
        self.id_str = '[G %02d/%02d, L %d/%d]' % (self.global_rank, self.world_size, self.local_rank, self.local_gpus)

        self.tokenizer = CLIPTokenizerFast.from_pretrained(cfg.model.pretrain)

        if self.cfg.model.label_bce:
            self.prompts = self.tokenizer(
                [
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
                    'A healthy chest-xray image with no findings.',
                ], 
                return_tensors="pt", padding=True, max_length=cfg.data.max_text_len, truncation=True
            )
        else:
            self.prompts = self.tokenizer(
                [
                    'A chest-xray image with healthy cardiomediastinum.', 'A chest-xray image of enlarged cardiomediastinum.',
                    'A chest-xray image with healthy heart size.', 'A chest-xray image of cardiomegaly.',
                    'A chest-xray image with clear lungs.', 'A chest-xray image of lung opacity',
                    'A chest-xray image with healthy lungs and no lesions.', 'A chest-xray image of lung lesion.',
                    'A chest-xray image with healthy lungs and no edema.', 'A chest-xray image of edema',
                    'A chest-xray image with healthy lungs and no consolidation.', 'A chest-xray image of consolidation',
                    'A chest-xray image of healthy lungs and no pneumonia.', 'A chest-xray image of pneumonia',
                    'A chest-xray image of healthy lungs and no atelectasis.', 'A chest-xray image of atelectasis',
                    'A chest-xray image of healthy lungs and no pneumothorax.', 'A chest-xray image of pneumothorax',
                    'A chest-xray image of healthy lungs and no pleural effusion.', 'A chest-xray image of pleural effusion',
                    'A chest-xray image of healthy lungs and no pleural diseases.', 'A chest-xray image of other pleural diseases',
                    'A chest-xray image of healthy, normal ribs.', 'A chest-xray image of rib fracture',
                    'A chest-xray image with no support devices.', 'A chest-xray image of support devices',
                    'An unhealthy chest-xray image showing abnormalities.', 'A healthy chest-xray image with no findings.'
                ], 
                return_tensors="pt", padding=True, max_length=cfg.data.max_text_len, truncation=True
            )

        if self.global_master:
            self.log('world size: %d' % self.world_size)
            os.makedirs(cfg.ckpt_dir, exist_ok=True)

            if not cfg.wandb:
                os.environ['WANDB_MODE'] = 'dryrun'
            wandb.login(key=os.environ['WANDB_KEY'])

    def log(self, message, master_only=True):
        if (master_only and self.global_master) or (not master_only):
            print(self.id_str + ': ' + str(message))
    
    def freeze_layers(self, model, prefix_list, freeze=True):
        for n, p in model.named_parameters():
            if any(prefix in n for prefix in prefix_list):
                p.requires_grad = not freeze

    def train(self):
        if self.global_master:
            wandb_init(self.cfg)

        train_dataloader, class_weights = build_mimic_daug_dataloader(self.cfg, 'train_val', True)
        test_dataloader, _ = build_mimic_daug_dataloader(self.cfg, 'test', True)
        self.log('Length training data %d' % len(train_dataloader))

        model = DAugModel(self.cfg)
        optimizer = build_optimizer(model, self.cfg)
        
        scheduler = get_scheduler(
            name=self.cfg.scheduler.name,
            optimizer=optimizer, 
            num_warmup_steps=self.cfg.scheduler.warmup, 
            num_training_steps=len(train_dataloader)*self.cfg.epochs)
        
        model.to(self.local_rank)
        model = DistributedDataParallel(model, device_ids=[self.local_rank], find_unused_parameters=False)

        #self.evaluate(model, test_dataloader, train_dataloader, 0)

        for epoch in range(self.cfg.epochs):
            model.train()
            train_dataloader.sampler.set_epoch(epoch)
            metric = MovingAverage()

            if self.global_master:
                wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

            for it, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

                x, x_diff, y = batch
                x_diff = x_diff.cuda() if self.cfg.model.heatmap else None
                y['prompts'] = self.prompts
                y['reports'] = self.tokenizer(
                    y.pop('reports'),
                    padding='longest',
                    max_length=self.cfg.data.max_text_len,
                    truncation=True,
                    return_tensors='pt',
                )
                y = recursive_to_device(y, 'cuda')
                loss = model(x.cuda(), x_diff, y)

                metric.add_val(loss.detach().item())

                loss.backward()
                optimizer.step()
                scheduler.step()

                if it % self.cfg.print_freq == 0:
                    self.log('Epoch: %3d/%3d, iter: %4d/%4d, loss: %.3f (%.3f)' %
                        (epoch, self.cfg.epochs, it, len(train_dataloader), metric.val, metric.mean))

            if self.global_master:                
                wandb.log(data={'loss': metric.mean}, step=epoch)
                if epoch % self.cfg.save_freq == 0 or epoch+1 == self.cfg.epochs:
                    save_checkpoint(epoch, model.module, optimizer, self.cfg.ckpt_dir)

            dist.barrier()

            if epoch == 0 or epoch % self.cfg.test_freq == 0 or epoch+1 == self.cfg.epochs:
                self.evaluate(model, test_dataloader, train_dataloader, epoch+1)

            dist.barrier()

    def inference_cls(self, model, test_loader, train_loader):
        self.log('Running test')

        emb = {'src':{'x':[], 'r':[]}, 'tar':{'x':[], 'r': []}}
        labels = {'src':[], 'tar': []}        

        emb['src']['x'], emb['src']['r'], probs_test, labels['src'], img_ids_test = self.inference_embeddings(model, test_loader)
        emb['tar']['x'], emb['tar']['r'], probs_train, labels['tar'], img_ids_train = self.inference_embeddings(model, train_loader)


        if self.global_master:
            torch.save({
                'img_ids_test': img_ids_test,
                'probs_test': probs_test.cpu().numpy(),
                'img_ids_train': img_ids_train,
                'probs_train': probs_train.cpu().numpy()
            }, 'cls_out.pt')

    def evaluate(self, model, test_loader, train_loader, epoch=0):
        self.log('Running test')

        emb = {'src':{'x':[], 'r':[]}, 'tar':{'x':[], 'r': []}}
        labels = {'src':[], 'tar': []}        

        emb['src']['x'], emb['src']['r'], probs_test, labels['src'], img_ids_test = self.inference_embeddings(model, test_loader)
        emb['tar']['x'], emb['tar']['r'], probs_train, labels['tar'], img_ids_train = self.inference_embeddings(model, train_loader)

        result_x2x, result_r2x = self.evaluate_retrieval(emb, labels, epoch)
        result_cls = self.evaluate_classification(probs_test, labels['src'], epoch)

        if self.global_master:
            with open(os.path.join(self.cfg.ckpt_dir, 'eval_epoch_%01d.json' % epoch), 'w') as f:
                json.dump({
                    'x->x': result_x2x,
                    'r->x': result_r2x,
                    'classify': result_cls
                }, f, indent=4)

    def inference_embeddings(self, model, loader):
        emb_x_all, emb_y_all, probs_all, labels_all = [], [], [], []   
        img_ids_all = []     
        
        with torch.no_grad():
            model.eval()
            for batch in tqdm(loader, desc='inference', disable=not self.global_master):
                x, x_diff, y = batch
                x_diff = x_diff.cuda() if self.cfg.model.heatmap else None
                img_id = y.pop('img_ids')
                y['prompts'] = self.prompts
                y['reports'] = self.tokenizer(
                    y.pop('reports'),
                    padding='longest',
                    max_length=self.cfg.data.max_text_len,
                    truncation=True,
                    return_tensors='pt',
                )
                y = recursive_to_device(y, 'cuda')                                
                emb_x, emb_t, prob = model(x.cuda(), x_diff, y)

                gathered_emb_x = torch.zeros(self.world_size*emb_x.shape[0], emb_x.shape[1]).cuda()
                gathered_emb_t = torch.zeros(self.world_size*emb_t.shape[0], emb_t.shape[1]).cuda()
                gathered_probs = torch.zeros(self.world_size*prob.shape[0], prob.shape[1]).cuda()
                gathered_labels = torch.zeros(self.world_size*y['labels'].shape[0], y['labels'].shape[1], dtype=y['labels'].dtype).cuda()
                dist.all_gather_into_tensor(gathered_emb_x, emb_x.detach().contiguous())
                dist.all_gather_into_tensor(gathered_emb_t, emb_t.detach().contiguous())
                dist.all_gather_into_tensor(gathered_probs, prob.detach().contiguous())
                dist.all_gather_into_tensor(gathered_labels, y['labels'].contiguous())
                emb_x_all.append(gathered_emb_x)
                emb_y_all.append(gathered_emb_t)
                probs_all.append(gathered_probs)
                labels_all.append(gathered_labels)

                gathered_img_ids = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered_img_ids, img_id)
                img_ids_all.extend(gathered_img_ids)
        
        emb_x_all = torch.vstack(emb_x_all)
        emb_y_all = torch.vstack(emb_y_all)
        probs_all = torch.vstack(probs_all)
        labels_all = torch.vstack(labels_all)

        return emb_x_all, emb_y_all, probs_all, labels_all, img_ids_all

    def evaluate_retrieval(self, emb, labels, epoch=0):                
        def do_retrieval(src, tar, labels):
            sim = torch.matmul(src, tar.t())
            results = {}
            for k in [1, 3, 5, 10]:
                k_sims, k_idx = torch.topk(sim, k=k, dim=1)
                k_labels = torch.logical_and(labels['src'].unsqueeze(1), labels['tar'][k_idx]).cpu() * 1

                AP = []
                for i in range(k_labels.shape[-1]): # class
                    class_ap = []
                    for s in range(len(src)): # query
                        if labels['src'][s, i]:
                            positions = torch.arange(1, k+1)[k_labels[s, :, i] > 0]
                            ap = torch.div((torch.arange(len(positions)) + 1), positions).mean()
                            class_ap.append(torch.nan_to_num(ap, 0).item())
                    if len(class_ap) == 0:
                        class_ap = [0]
                    AP.append(class_ap)

                class_weights = torch.tensor([len(x) for x in AP])
                class_weights = class_weights / class_weights.sum()

                AP = torch.tensor([torch.tensor(x).mean() for x in AP])

                results[k] = {
                    'AP': [round(x, 3) for x in AP.tolist()],
                    'mAP': round(AP.mean().item(), 3),
                    'wmAP': round((AP * class_weights).sum().item(), 3),
                }

            return results

        result_x2x = do_retrieval(emb['src']['x'], emb['tar']['x'], labels)
        #result_x2r = do_retrieval(emb['src']['x'], emb['tar']['r'], labels)
        result_r2x = do_retrieval(emb['src']['r'], emb['tar']['x'], labels)

        if self.global_master:
            log_result = {k : {
                'x->x': result_x2x[k],
                #'x->r': result_x2r[k],
                'r->x': result_r2x[k]
            } for k in result_x2x.keys()}

            print(json.dumps(log_result, indent=4))
            wandb.log(data={
                'x->x, mAP': log_result[self.cfg.log_retrieval_k]['x->x']['mAP'],
                'x->x, wmAP': log_result[self.cfg.log_retrieval_k]['x->x']['wmAP'],
                #'x->r, mAP': log_result[self.cfg.log_retrieval_k]['x->r']['mAP'],
                #'x->r, wmAP': log_result[self.cfg.log_retrieval_k]['x->r']['wmAP'],
                'r->x, mAP': log_result[self.cfg.log_retrieval_k]['r->x']['mAP'],
                'r->x, wmAP': log_result[self.cfg.log_retrieval_k]['r->x']['wmAP'],
            }, step=epoch)

        return result_x2x, result_r2x

    def evaluate_classification(self, probs, labels, epoch=0):        
        wavg_auc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy(), average='weighted', multi_class='ovr')
        auc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy(), average=None, multi_class='ovr').tolist()
        log_result = {
            'auc': [round(x, 3) for x in auc],
            'avg_auc': torch.mean(torch.tensor(auc)).item(),
            'w_avg_auc': wavg_auc
        }
        if self.global_master:
            print(json.dumps(log_result, indent=4))
            wandb.log(data={
                'cls_ave_auc': log_result['avg_auc'],
                'cls_wave_auc': log_result['w_avg_auc']
            }, step=epoch)

        return log_result


@hydra.main(config_path='./configs', config_name='conf', version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    print(cfg.exp_name)

    # i = 0
    # ckpt_dir = cfg.ckpt_dir
    # while os.path.isdir(ckpt_dir):
    #     i += 1
    #     ckpt_dir = cfg.ckpt_dir + '_%d' % i
    # cfg.ckpt_dir = ckpt_dir

    os.environ['NCCL_DEBUG'] = 'VERSION'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    if cfg.debug:
        torch.manual_seed(cfg.random_seed)
        #torch.use_deterministic_algorithms(True)
        np.random.seed(cfg.random_seed)
        cfg.wandb=False
        cfg.data.batch_size=8

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    distributed_trainer = DistributedTrainer(cfg, local_rank)
    distributed_trainer.train()

if __name__ == '__main__':
    sys.argv = [x.replace('--local_rank=', 'local_rank=') for x in sys.argv]
    sys.argv = list(filter(lambda t: 'dummy' not in t, sys.argv))
    if 'blob_root' in sys.argv:
        idx = sys.argv.index('blob_root')
        sys.argv.pop(idx)
        sys.argv[idx] = 'blob_root=' + sys.argv[idx]
    else:
        # not AML submitted job
        pretty.install()
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    main()

    # with hydra.initialize(config_path='./configs', version_base=None):
    #     cfg = hydra.compose(config_name="phase2", overrides=sys.argv)
    #     main(cfg)

# OMP_NUM_THREADS=5 TOKENIZERS_PARALLELISM=true torchrun --standalone --nnodes=1 --nproc_per_node=8 main.py
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import os
from tqdm import tqdm
from transformers import CLIPTokenizerFast, CLIPImageProcessor
from PIL import Image
from utils import DaugDataCollatorWithPadding
import json

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
    def __init__(self, cfg, split) -> None:
        super().__init__()

        assert split in ['train', 'test', 'validate', 'train_val']
        self.cfg = cfg

        kwargs = dict(
            do_resize=True, 
            size=[cfg.data.img_size, cfg.data.img_size], 
            do_center_crop=False
        )
        self.processor = CLIPImageProcessor.from_pretrained(cfg.model.pretrain, **kwargs)

        data = torch.load(os.path.join(cfg.data.data_root, cfg.data.anno_path))
        self.meta = data['meta_data']
        print(self.meta['condition_names'])
        if split == 'train_val':
            data = list(data['train'].values()) + list(data['validate'].values())
        else:
            data = list(data[split].values())

        if self.cfg.debug:
            data = data[:32*8]

        self.image_info = []
        self.gt_report = []
        self.gt_report_split = []
        self.gt_prompt_cls = []
        self.condition_class = []
        self.info_dict = []

        if cfg.data.mode == 'by_pathology':
            input_class_names = self.meta['condition_names']
            super_classes = [[x] for x in self.cfg.data.class_labels]
            super_class_idx = [[input_class_names.index(c) for c in names] for names in super_classes]

        is_master = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        import random
        random.seed(cfg.random_seed)
        random.shuffle(data)

        for s in tqdm(data, desc='loading %s' % split, disable=not is_master):
            if cfg.data.mode == 'by_pathology':
                conditions = []
                conds_ori = np.array(s['pred_%s' % cfg.data.class_provider])
                for i in range(len(super_classes)):
                    conditions.append(conds_ori[super_class_idx[i]].max())                
                if max(conditions) < 1:
                    continue                                
            else:
                conditions = s['pred_%s' % cfg.data.class_provider]

            if cfg.data.class_provider == 'chexpert':
                conditions = np.array(conditions, dtype=np.int32)
                conditions[conditions == -2] = 0 # treat blank as negative
                if cfg.data.chexpert_uncertain == 'skip':
                    if -1 in conditions:
                        continue
                else:                        
                    conditions[conditions == -1] = 0                        
                
                conditions = conditions.tolist()

            # '', 'AP', 'AP AXIAL', 'AP LLD', 'AP RLD', 'LAO', 'LATERAL', 'LL', 'LPO', 'PA', 'PA LLD', 'RAO', 'SWIMMERS', 'XTABLE LATERAL'
            views = np.array(s['images']['view_position'])
            idx_pa = np.flatnonzero(views == 'PA').tolist()
            idx_ap = np.flatnonzero(np.logical_or(views == 'AP', views == 'AP AXIAL')).tolist()
            idx_lat = np.flatnonzero(views == 'LATERAL').tolist()
            idx_empty = np.flatnonzero(views == '').tolist()
            idx_unrecognized = list(filter(lambda i: views[i] not in ['PA', 'AP', 'LATERAL', ''], range(len(views))))
            
            img_idx.append(i)            

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
        super_classes = super_classes = [[x] for x in self.cfg.data.class_labels]
        super_class_idx = [[input_class_names.index(c) for c in names] for names in super_classes]
        output_labels = []
        for i in range(len(super_classes)):
            output_labels.append((np.array(input_labels)[super_class_idx[i]].max() == 1) * 1)
        return output_labels

    def stat_pathologies(self):
        import pandas as pd
        stats = pd.DataFrame(columns=['pathology', 'total', 'ratio', 'n_multi', 'n_single'])

        for i, pathology in enumerate(self.cfg.data.class_labels):
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
        else:
            img_fp = self.image_info[index]['path']
        
        # this is the pre-generated heatmap
        x_diff = np.load(img_fp.replace(self.cfg.data.img_dir, self.cfg.data.heatmap_dir).replace('.jpg', self.cfg.data.heatmap_name))
        x_diff = (x_diff - x_diff.min()) / (x_diff.max() - x_diff.min())
        x_diff = Image.fromarray(np.uint8(x_diff*255)).convert('RGB')        

        if 'mimic-cxr-resized' in img_fp:
            img_fp = img_fp.replace('.jpg', '.png')
        x = Image.open(img_fp).convert('RGB')
        
        x, x_diff = self.processor(images = [x, x_diff], return_tensors="pt").pixel_values
        y = {
            'labels': torch.tensor(self.condition_class[index]),
            'reports': self.gt_report[index],
            'img_ids': os.path.basename(img_fp).split('.')[0]
        }
     
        # y = {
        #     'gt_report': self.gt_report[index],
        #     'gt_report_split': self.gt_report_split[index],
        #     'gt_prompt_cls': self.gt_prompt_cls[index],
        #     'condition_class': torch.tensor(self.condition_class[index]),
        #     'view_position': self.image_info[index]['view_position'],
        #     'path': self.image_info[index]['path'],
        #     'index': index
        # }
        # info = self.info_dict[index]
    
        return x, x_diff, y

def build_mimic_daug_dataloader(cfg, split, dist=True):
    dataset = MimicDataset(cfg, split)
    
    collate_fn = DaugDataCollatorWithPadding()
    if dist:
        sampler = DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, sampler=sampler, num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=(split!='test'), num_workers=cfg.data.num_workers, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    pos_weight = dataset.get_pos_weights()

    return loader, pos_weight
    
if __name__ == '__main__':
    from omegaconf import DictConfig, OmegaConf
    cfg = OmegaConf.load('configs/conf.yaml')
    cfg.data.class_provider = ['visual_chexbert', 'chexpert'][1]
    cfg.data.chexpert_uncertain = 'negative' # 'skip' or 'negative'
    cfg.data.num_workers = 1
    
    assert cfg.data.class_provider in ['visual_chexbert', 'chexpert']
    assert cfg.data.chexpert_uncertain in ['skip', 'negative']

    dataset = MimicDataset(cfg, 'test')

    import random
    import json

    conds = np.array(['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                'Support Devices', 'No Finding'])
    sup_conds = np.array([cfg.data.neg_classes] + cfg.data.pos_classes, dtype=object)

    while True:
        i = random.randint(0, len(dataset))
        x, x_diff, y = dataset[i]
    #     x.save('test.jpg')
        print(x.size)
    #     print(y['gt_report'])
    #     # print('Classes: ')
    #     # print(conds[y['_condition_class'].astype(bool)])
    #     print('Superclasses: ')
    #     print(sup_conds[y['condition_class'].astype(bool)])
    #     break
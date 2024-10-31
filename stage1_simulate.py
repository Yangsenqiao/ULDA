import pickle
import os
import clip
import torch
import network
import torch.nn as nn
from utils.stats import calc_mean_std
import argparse
from torch.utils import data
import numpy as np
import random
import wandb
from datasets import Cityscapes, gta5
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a photo of one {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'a low resolution photo of a {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a photo of a nice {}.',
    'a blurry photo of a {}.',
    'art of a {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
]


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to dataset")
    parser.add_argument("--save_dir1", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--save_dir2", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--save_dir3", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--save_dir4", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--save_dir5", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--save_dir6", type=str, 
                        help= "path for learnt parameters saving")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','gta5'], help='Name of dataset')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default = 'RN50',
                        help= "backbone name" )
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type = int, default =100,
                        help= "total number of optimization iterations")
    # learn statistics
    parser.add_argument("--resize_feat",action='store_true',default=False,
                        help="resize the features map to the dimension corresponding to CLIP")
    # random seed
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    # target domain description
    parser.add_argument("--domain_desc1", type=str , default = "driving under rain",
                        help = "description of the target domain")
    parser.add_argument("--domain_desc2", type=str , default = "driving at night",
                        help = "description of the target domain")
    parser.add_argument("--domain_desc3", type=str , default = "driving in snow",
                        help = "description of the target domain")
    parser.add_argument("--domain_desc4", type=str , default = "driving in fog",
                        help = "description of the target domain")
    parser.add_argument("--domain_desc5", type=str , default = "driving in a game",
                        help = "description of the target domain")
    parser.add_argument("--domain_desc6", type=str , default = "driving in fog",
                        help = "description of the target domain")
    parser.add_argument("--notes", type=str , default = "debug")
    parser.add_argument("--wandb_name", type=str , default = "debug")
    parser.add_argument("--wandb_project", type=str , default = "debug")
    parser.add_argument("--wandb_entity", type=str , default = "debug")
    # projector learning rate
    parser.add_argument("--proj_lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=10)
    parser.add_argument("--pixel_rate", type=float, default=1e-1)
    parser.add_argument("--region_rate", type=float, default=1e-1)
    parser.add_argument("--cor_rate", type=float, default=1e-1)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--seg_rate", type=float, default=1.0)
    parser.add_argument("--domain_rate", type=float, default=10)


    return parser

def get_dataset(dataset,data_root,crop_size,ACDC_sub="night",data_aug=True):
    """ Dataset And Augmentation
    """
    if dataset == 'cityscapes':
        if data_aug:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform)

    if dataset == 'ACDC':
        train_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform, ACDC_sub = ACDC_sub)
        val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform, ACDC_sub = ACDC_sub)

    if dataset == "gta5":
        
        if data_aug:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(768, 768)),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(768, 768)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        val_transform = et.ExtCompose([
            et.ExtCenterCrop(size=(1046, 1914)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_train.txt',transform=train_transform)
        val_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_val.txt',transform=val_transform)

    return train_dst, val_dst

class PROJECTOR(nn.Module):
    def __init__(self, indim, outdim, shape=None):
        super(PROJECTOR, self).__init__()

        layers = [nn.Linear(indim, shape[0] if shape else outdim)] 
        if shape:
            for i in range(len(shape) - 1):
                layers.append(nn.Linear(shape[i], shape[i+1]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(shape[-1], outdim))
        self.projector = nn.Sequential(*layers)

    def forward(self, feature):
        feature = feature.permute(1, 2, 0)  
        feature =  self.projector(feature)
        feature = feature.permute(2, 0, 1) 
        return feature




def mapping(raw_class, raw_label, template='a {} in the {}'):
    class_name  = [Cityscapes.train_id_to_name[class_idx.item()] for class_idx in raw_class]
    filled_templates = [template.format(item) for item in class_name]

    max_class_id = - 1
    new_label_map = torch.full(raw_label.shape, max_class_id, dtype=torch.long)

    for new_class_id, old_class_tensor in enumerate(raw_class):
        old_class_id = old_class_tensor.item() 
        new_label_map = torch.where(raw_label == old_class_id, 
                                    torch.tensor(new_class_id, dtype=torch.long), 
                                    new_label_map)

    
    special_value = 255
    new_label_map = torch.where(raw_label == special_value, 
                                torch.tensor(special_value, dtype=torch.long), 
                                new_label_map)

    return new_label_map, filled_templates


def extract_and_pool_features(raw_class, raw_label, raw_feature, template='a {} in the {}'):
    class_name = [Cityscapes.train_id_to_name[class_idx] for class_idx in raw_class]  
    filled_templates = [template.format(item) for item in class_name]

    new_label_map = torch.zeros_like(raw_label, dtype=torch.long)

    pooled_features_list = []

    for new_class_id, old_class_id in enumerate(raw_class):
        mask = (raw_label == old_class_id)

        new_label_map = torch.where(mask, torch.tensor(new_class_id, dtype=torch.long), new_label_map)

        class_features = raw_feature[:, mask] 

        pooled_features = torch.mean(class_features, dim=1)  
        pooled_features_list.append(pooled_features)

    special_value = 255
    new_label_map = torch.where(raw_label == special_value, torch.tensor(special_value, dtype=torch.long), new_label_map)

    pooled_features_tensor = torch.stack(pooled_features_list)

    return new_label_map, filled_templates, pooled_features_tensor

class PIN(nn.Module):
    def __init__(self,shape,content_feat):
        super(PIN,self).__init__()
        self.shape = shape
        self.content_feat = content_feat.clone().detach()
        self.content_mean, self.content_std = calc_mean_std(self.content_feat)
        self.size = self.content_feat.size()
        self.content_feat_norm = (self.content_feat - self.content_mean.expand(
        self.size)) / self.content_std.expand(self.size)

        self.style_mean =   self.content_mean.clone().detach() 
        self.style_std =   self.content_std.clone().detach()

        self.style_mean = nn.Parameter(self.style_mean, requires_grad = True)
        self.style_std = nn.Parameter(self.style_std, requires_grad = True)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self):
        
        self.style_std.data.clamp_(min=0)
        target_feat =  self.content_feat_norm * self.style_std.expand(self.size) + self.style_mean.expand(self.size)
        target_feat = self.relu(target_feat)

        return target_feat
        

def main():

    opts = get_argparser().parse_args()
    opts.raw_save_dir = [opts.save_dir1, opts.save_dir2, opts.save_dir3, \
                    opts.save_dir4, opts.save_dir5, opts.save_dir6]


    wandb.init(
        name=opts.wandb_name,
        project=opts.wandb_project,
        entity=opts.wandb_entity,
        mode="online",
        save_code=True,
        config=opts,
        notes=opts.notes
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,data_aug=False)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,
        drop_last=False)  # drop_last=True to ignore single-image batches.
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    model = network.modeling.__dict__[opts.model](num_classes=19,BB= opts.BB,replace_stride_with_dilation=[False,False,False])
    model_seg = network.modeling.__dict__[opts.model](num_classes=19, BB= opts.BB,replace_stride_with_dilation=[False,False,True])
    model_seg.backbone.attnpool = nn.Identity()
    for param in model_seg.backbone.parameters():
        param.requires_grad = False
    model_seg.backbone.eval()
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model_seg.load_state_dict(checkpoint["model_state"])
        
        model_seg.to(device)
        del checkpoint  # free memory

    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    cur_itrs = 0
    

    if opts.resize_feat:
        t1 = nn.AdaptiveAvgPool2d((56,56))
    else:
        t1 = lambda x:x

    #text
    #target text
    text_target_all = []
    for desc in [opts.domain_desc1, opts.domain_desc2, opts.domain_desc3, \
                opts.domain_desc4, opts.domain_desc5, opts.domain_desc6]:
        target = compose_text_with_templates(desc, imagenet_templates)
        tokens = clip.tokenize(target).to(device)
        text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
        text_target /= text_target.norm(dim=-1, keepdim=True)
        text_target = text_target.repeat(opts.batch_size,1).type(torch.float32)  # (B,1024)
        text_target_all.append(text_target)



    fea_proj1 = PROJECTOR(256, 1024, [512]) 
    fea_proj2 = PROJECTOR(256, 1024, [512]) 
    fea_proj3 = PROJECTOR(256, 1024, [512]) 
    fea_proj4 = PROJECTOR(256, 1024, [512]) 
    fea_proj5 = PROJECTOR(256, 1024, [512]) 
    fea_proj6 = PROJECTOR(256, 1024, [512]) 


    fea_proj1.to(device)
    fea_proj2.to(device)
    fea_proj3.to(device)
    fea_proj4.to(device)
    fea_proj5.to(device)
    fea_proj6.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion_cor = torch.nn.MSELoss()

    for i,(img_id, tar_id, images, labels) in enumerate(train_loader):
        print(i)    
        labels = labels.to(dtype=torch.long)
        
        f1 = model.backbone(images.to(device),trunc1=False,trunc2=False,
        trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)  # (B,C1,H1,W1)
        
        #optimize mu and sigma of target features with CLIP
        model_pin_1 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
        model_pin_1.to(device)
        model_pin_2 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
        model_pin_2.to(device)
        model_pin_3 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
        model_pin_3.to(device)
        model_pin_4 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
        model_pin_4.to(device)
        model_pin_5 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
        model_pin_5.to(device)
        model_pin_6 = PIN([f1.shape[0],256,1,1],f1.to(device)) #  mu_T (B,C1)  sigma_T(B,C1)
        model_pin_6.to(device)

        optimizer_proj = torch.optim.SGD(params=[
            {'params': fea_proj1.parameters(), 'lr': opts.proj_lr},
            {'params': fea_proj2.parameters(), 'lr': opts.proj_lr},
            {'params': fea_proj3.parameters(), 'lr': opts.proj_lr},
            {'params': fea_proj4.parameters(), 'lr': opts.proj_lr},
            {'params': fea_proj5.parameters(), 'lr': opts.proj_lr},
            {'params': fea_proj6.parameters(), 'lr': opts.proj_lr},

        ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)



        optimizer_pin_1 = torch.optim.SGD(params=[
            {'params': model_pin_1.parameters(), 'lr': 1},
        ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)


        optimizer_pin_2 = torch.optim.SGD(params=[
            {'params': model_pin_2.parameters(), 'lr': 1},
        ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)


        optimizer_pin_3 = torch.optim.SGD(params=[
            {'params': model_pin_3.parameters(), 'lr': 1},
        ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)
        optimizer_pin_4 = torch.optim.SGD(params=[
            {'params': model_pin_4.parameters(), 'lr': 1},
        ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)
        optimizer_pin_5 = torch.optim.SGD(params=[
            {'params': model_pin_5.parameters(), 'lr': 1},
        ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)
        optimizer_pin_6 = torch.optim.SGD(params=[
            {'params': model_pin_6.parameters(), 'lr': 1},
        ], lr= 1, momentum=0.9, weight_decay=opts.weight_decay)

        if i == len(train_loader)-1 and f1.shape[0] < opts.batch_size :
            # text_target = text_target[:f1.shape[0]]
            text_target_all = [text_target[:f1.shape[0]] for text_target in text_target_all]
        cur_itrs = 0
        if i>=100:
            break
        while cur_itrs< opts.total_it: 


            cur_itrs += 1
            if cur_itrs % opts.total_it==0:
                print(cur_itrs)

            optimizer_pin_1.zero_grad()
            optimizer_pin_2.zero_grad()
            optimizer_pin_3.zero_grad()
            optimizer_pin_4.zero_grad()
            optimizer_pin_5.zero_grad()
            optimizer_pin_6.zero_grad()
            optimizer_proj.zero_grad()

            f1_hal_1 = model_pin_1()
            f1_hal_2 = model_pin_2()
            f1_hal_3 = model_pin_3()
            f1_hal_4 = model_pin_4()
            f1_hal_5 = model_pin_5()
            f1_hal_6 = model_pin_6()

            f1_hal_trans_1 = t1(f1_hal_1)
            f1_hal_trans_2 = t1(f1_hal_2)
            f1_hal_trans_3 = t1(f1_hal_3)
            f1_hal_trans_4 = t1(f1_hal_4)
            f1_hal_trans_5 = t1(f1_hal_5)
            f1_hal_trans_6 = t1(f1_hal_6)

            f1_hal_trans_all = [f1_hal_trans_1, f1_hal_trans_2, f1_hal_trans_3, \
                            f1_hal_trans_4, f1_hal_trans_5, f1_hal_trans_6]
            f1_hal_all = [f1_hal_1, f1_hal_2, f1_hal_3, f1_hal_4, f1_hal_5, f1_hal_6]

            template_all = [ '{} '+ ' '.join(opts.domain_desc1.split()[1:]), \
                             '{} '+ ' '.join(opts.domain_desc2.split()[1:]), \
                             '{} '+ ' '.join(opts.domain_desc3.split()[1:]), \
                             '{} '+ ' '.join(opts.domain_desc4.split()[1:]), \
                             '{} '+ ' '.join(opts.domain_desc5.split()[1:]), \
                             '{} '+ ' '.join(opts.domain_desc6.split()[1:]), \
                            ]
            Domain_all_loss = 0
            Domain_clip_loss = 0
            Domain_pixel_loss = 0
            Domain_region_loss = 0
            Domain_seg_loss = 0
            Domain_cor_loss = 0
            Domain_intercor_loss = 0
            # for domain_idx in range(3):
            projector_all = [fea_proj1, fea_proj2, fea_proj3, fea_proj4, fea_proj5, fea_proj6]

            for domain_idx in range(6):
                f1_hal = f1_hal_all[domain_idx]
                f1_hal_trans = f1_hal_trans_all[domain_idx]
                '''==============================ADD Semantic segmentation Loss==========================='''
                outputs_map = model_seg.forward_pin(f1_hal, labels)# correct the loss
                Domain_seg_loss += criterion(outputs_map, labels.to(device))
                #target_features (optimized)
                target_features_from_f1 = model.backbone(f1_hal_trans,trunc1=True,trunc2=False,trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=False)
                target_features_from_f1 /= target_features_from_f1.norm(dim=-1, keepdim=True).clone().detach()
        
                #loss CLIP
                Domain_clip_loss += (1- torch.cosine_similarity(text_target_all[domain_idx], target_features_from_f1, dim=1)).mean()
            Domain_clip_loss = Domain_clip_loss /6
            Domain_seg_loss = Domain_seg_loss / 6
            for index in range(len(labels)):
                cur_label_raw = labels[index]
                cur_label = torch.nn.functional.interpolate(cur_label_raw.to(dtype=torch.float).unsqueeze(0).unsqueeze(0), \
                                    size=[f1_hal.shape[-2],f1_hal.shape[-1]] , mode='nearest') 
                cur_label = cur_label.to(dtype=torch.long)
                cur_label = cur_label.squeeze(0).squeeze(0)

                del cur_label_raw 
                unique_classes = list(cur_label.unique())
                if 255 in unique_classes:
                    unique_classes.remove(255)

                loss_region = 0
                loss_pixel = 0
                loss_cor = 0
                Domain_text = []
                Domain_proto = []

                for domain_idx in range(6):

                    cur_features = projector_all[domain_idx](f1_hal_all[domain_idx][index])

                    cur_gt, cur_text, cur_prototype = extract_and_pool_features(unique_classes, cur_label, cur_features, \
                                                                template = template_all[domain_idx] )

                    encoded_vectors = []
                    for input_txt in cur_text:
                        target_txt = compose_text_with_templates(input_txt, imagenet_templates)
                        tokens_txt = clip.tokenize(target_txt).to(device)
                        encoded_text = clip_model.encode_text(tokens_txt).mean(axis=0, keepdim=True).detach()
                        encoded_text = encoded_text / encoded_text.norm(dim=-1, keepdim=True)
                        encoded_vectors.append(encoded_text.cpu())
                    # combine
                    text_matrix = torch.cat(encoded_vectors, dim=0)
                    text_matrix = text_matrix.to(device)
                    del encoded_vectors 

                    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>The Region Level Loss>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
                    cur_prototype = cur_prototype.permute(1,0)
                    text_matrix = text_matrix.to(dtype=torch.float32)
                    text_matrix = F.normalize(text_matrix, 2, -1)  
                    cur_prototype = F.normalize(cur_prototype, 2, 0)  

                    Domain_text.append(text_matrix)
                    Domain_proto.append(cur_prototype.permute(1,0))



                    logits_pro = torch.einsum('ij,jk->ik', text_matrix, cur_prototype) * opts.temperature
                    pro_gt = torch.arange(0, len(logits_pro), dtype = torch.long).to(device)
                    cl_pro = criterion(logits_pro.unsqueeze(0), pro_gt.unsqueeze(0))
                    loss_region += cl_pro
                    
                    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>The Region Level Correlation loss>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
                    cor_gt = torch.matmul(text_matrix, text_matrix.permute(1, 0))
                    cor_feat = torch.matmul(cur_prototype.permute(1, 0), cur_prototype)
                    cor_loss = criterion_cor(cor_feat, cor_gt)
                    loss_cor += cor_loss

                    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>The Pixel Level Loss>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
                    
                    cur_features = cur_features.to(dtype=torch.float32)
                    cur_features = F.normalize(cur_features, 2, 0)  
                    logits = torch.einsum('ij,jkl->ikl', text_matrix, cur_features) * opts.temperature
                    # del text_matrix 
                    class_num = logits.shape[0]
                    assert class_num==len(unique_classes)
                    cur_gt = cur_gt.to(device)
                    cl_loss = criterion(logits.view(class_num, -1).unsqueeze(0), cur_gt.\
                                        view(-1).unsqueeze(0))
                    loss_pixel += cl_loss
                    # del logits   
   

                '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>The Domain Level Correlation loss>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

                Domain_text = torch.cat(Domain_text, dim=0)
                Domain_proto = torch.cat(Domain_proto, dim=0)
                domain_cor_gt = torch.matmul(Domain_text, Domain_text.permute(1, 0))
                domain_cor_feat = torch.matmul(Domain_proto, Domain_proto.permute(1, 0))
                Domain_cor_loss += criterion_cor(domain_cor_feat, domain_cor_gt)

                Domain_pixel_loss += loss_pixel  / 6
                Domain_region_loss += loss_region/ 6
                Domain_intercor_loss += loss_cor / 6
                '''==============================Finish============================='''
            Domain_pixel_loss /= len(labels)
            Domain_region_loss  /= len(labels)
            Domain_intercor_loss /= len(labels)
            Domain_cor_loss /= len(labels)

            #loss All
            Domain_all_loss = Domain_clip_loss + Domain_pixel_loss * opts.pixel_rate + \
                Domain_region_loss * opts.region_rate + Domain_seg_loss * opts.seg_rate + \
                 Domain_intercor_loss * opts.cor_rate + Domain_cor_loss*opts.domain_rate

            Domain_all_loss.backward()
            optimizer_pin_1.step()
            optimizer_pin_2.step()
            optimizer_pin_3.step()
            optimizer_pin_4.step()
            optimizer_pin_5.step()
            optimizer_pin_6.step()
            optimizer_proj.step()


                
            ##>>>>>>>>>>>>>>>>>>>>>>>>>>>SAVE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  
            model_pins = [model_pin_1, model_pin_2, model_pin_3, \
                        model_pin_4, model_pin_5, model_pin_6]

            for save_idx in range(len(text_target_all)):              
                cur_raw_save_dir = opts.raw_save_dir[save_idx]
                cur_save_dir = cur_raw_save_dir 
                if not os.path.isdir(cur_save_dir):
                    os.mkdir(cur_save_dir)
                for name, param in model_pins[save_idx].named_parameters():
                    if param.requires_grad and name == 'style_mean':
                        learnt_mu_f1 = param.data
                    elif param.requires_grad and name == 'style_std':
                        learnt_std_f1 = param.data

                for k in range(learnt_mu_f1.shape[0]):
                    learnt_mu_f1_ = torch.from_numpy(learnt_mu_f1[k].detach().cpu().numpy())
                    learnt_std_f1_ = torch.from_numpy(learnt_std_f1[k].detach().cpu().numpy())

                    stats = {}
                    stats['mu_f1'] = learnt_mu_f1_
                    stats['std_f1'] = learnt_std_f1_

                    with open(cur_save_dir+'/'+img_id[k].split('/')[-1]+'.pkl', 'wb') as f:
                        pickle.dump(stats, f)

    


main()
from re import L
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import Cityscapes, gta5
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pickle
from utils.utils import denormalize
from torchvision.utils import save_image
import wandb
import clip
import torch.nn.functional as F
from utils.stats import *
import copy


def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','ACDC','gta5'], help='Name of dataset')
    parser.add_argument("--ACDC_sub", type=str, default="rain",
                        help = "specify which subset of ACDC  to use")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type = str, default = "RN50",
                        help = "backbone of the segmentation network")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 1)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")

    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--freeze_BB", action='store_true',default=False,
                        help="Freeze the backbone when training")
    parser.add_argument("--ckpts_path", type = str ,
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=False)
    #validation
    parser.add_argument("--val_results_dir", type=str,help="Folder name for validation results saving")
    #Augmented features
    parser.add_argument("--train_aug",action='store_true',default=False,
                        help="train on augmented features using CLIP")
    parser.add_argument("--mix", action='store_true',default=False,
                        help="mix statistics")
    ## Wandb Information
    parser.add_argument("--notes", type=str , default = "debug",
                        help = "description of the target domain")
    parser.add_argument("--wandb_name", type=str , default = "debug",
                        help = "description of the target domain")
    parser.add_argument("--wandb_project", type=str , default = "debug")
    parser.add_argument("--wandb_entity", type=str , default = "debug")
    parser.add_argument("--test_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--testset", type=str, default='cityscapes',
                        choices=['cityscapes','ACDC','gta5'], help='Name of dataset')
    
    parser.add_argument("--proj_lr", type=float, default=1e-2)
    ## Path of the pin
    parser.add_argument("--path_mu_sig", type=str)
    parser.add_argument("--path_mu_sig2", type=str)
    parser.add_argument("--path_mu_sig3", type=str)
    parser.add_argument("--path_mu_sig4", type=str)
    parser.add_argument("--path_mu_sig5", type=str)
    parser.add_argument("--path_mu_sig6", type=str)
    # Target Domain description
    parser.add_argument("--domain_desc1", type=str,
                        help = "description of the target domain")
    parser.add_argument("--domain_desc2", type=str,
                        help = "description of the target domain")
    parser.add_argument("--domain_desc3", type=str,
                        help = "description of the target domain")
    parser.add_argument("--domain_desc4", type=str,
                        help = "description of the target domain")
    parser.add_argument("--domain_desc5", type=str,
                        help = "description of the target domain")
    parser.add_argument("--domain_desc6", type=str,
                        help = "description of the target domain")
    parser.add_argument("--scale", type=float, default=1e-1)


    return parser



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

class PROJECT(nn.Module):
    def __init__(self, input_dim= 1024, output_dim=1024):
        super(PROJECT,self).__init__()
        # self.std_rate = 0.5
        # self.mean_rate = 0.5
        projector = []
        projector.append(nn.Linear(input_dim, output_dim))
        projector.append(nn.ReLU())
        projector.append(nn.Linear(output_dim, output_dim))
        self.projector = nn.Sequential(*projector)


    def forward(self, feature):
        return self.projector(feature)

        
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



def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists(opts.val_results_dir):
            os.mkdir(opts.val_results_dir)
        img_id = 0

    with torch.no_grad():

        for i, (im_id, tg_id, images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            new_feature = model.forward_low(images)
            
            outputs = model.forward_head(new_feature, images.shape)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
           
            metrics.update(targets, preds)
            
            if opts.save_val_results:
                for j in range(len(images)):

                    target = targets[j]
                    pred = preds[j]

                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(target).save(opts.val_results_dir+'/%d_target.png' % img_id)
                    Image.fromarray(pred).save(opts.val_results_dir+'/%d_pred.png' % img_id)

                    images[j] = denormalize(images[j],mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
                    save_image(images[j],opts.val_results_dir+'/%d_image.png' % img_id)

                    fig = plt.figure()
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    #plt.savefig(opts.val_results_dir+'/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score



def main(opts):

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
  
    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,opts.ACDC_sub,
                                    data_aug=opts.data_aug)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
    drop_last=True)  # drop_last=True to ignore single-image batches.

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)


    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=19, BB= opts.BB,replace_stride_with_dilation=[False,False,True])
    model.backbone.attnpool = nn.Identity()

    #fix the backbone
    if opts.freeze_BB:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()

    # Set up metrics
    metrics = StreamSegMetrics(19)

    model_project = PROJECT(input_dim=256)
    model_alpha_mean_project = PROJECT(input_dim=1024, output_dim=256)
    model_alpha_std_project = PROJECT(input_dim=1024, output_dim=256)
    initial_scale = torch.tensor(opts.scale) * torch.ones(256)
    learnable_scale = nn.Parameter(initial_scale, \
                                requires_grad = True)
    model_project.train()
    model_alpha_mean_project.train()
    model_alpha_std_project.train()

    model_project.to(device)
    model_alpha_mean_project.to(device)
    model_alpha_std_project.to(device)
    model_parameters_best=None
    # Set up optimizer
    if opts.freeze_BB:
        optimizer = torch.optim.SGD(params=[
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            {'params': model_project.parameters(), 'lr': opts.proj_lr},
            {'params': model_alpha_mean_project.parameters(), 'lr': opts.proj_lr},
            {'params': model_alpha_std_project.parameters(), 'lr': opts.proj_lr},
            {'params': [learnable_scale], 'lr': opts.proj_lr*0.1},

            ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.001 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            {'params': model_project.parameters(), 'lr': opts.proj_lr},
            {'params': model_alpha_mean_project.parameters(), 'lr': opts.proj_lr},
            {'params': model_alpha_std_project.parameters(), 'lr': opts.proj_lr},

            ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    # if not opts.test_only:
    #     utils.mkdir(opts.ckpts_path)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model.to(device)
    
    # ==========   Train Loop   ==========#

    if opts.test_only:
       
        model.eval()

        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)

        print(metrics.to_str(val_score))
        print(val_score["Mean IoU"])
        print(val_score["Class IoU"])
        return

    interval_loss = 0
    intervalproj_loss = 0
    clip_model, preprocess = clip.load(opts.BB, device, jit=False)

    if opts.train_aug:
        files1 = [opts.path_mu_sig+'/'+f for f in os.listdir(opts.path_mu_sig+'/')]
        files2 = [opts.path_mu_sig2+'/'+f for f in os.listdir(opts.path_mu_sig2+'/')]
        files3 = [opts.path_mu_sig3+'/'+f for f in os.listdir(opts.path_mu_sig3+'/')]
        files4 = [opts.path_mu_sig4+'/'+f for f in os.listdir(opts.path_mu_sig4+'/')]
        files5 = [opts.path_mu_sig5+'/'+f for f in os.listdir(opts.path_mu_sig5+'/')]
        files6 = [opts.path_mu_sig6+'/'+f for f in os.listdir(opts.path_mu_sig6+'/')]

        files_all = [(file, 0) for file in files1] + \
                     [(file, 1) for file in files2] + \
                     [(file, 2) for file in files3] + \
                    [(file, 3) for file in files4] + \
                    [(file, 4) for file in files5] + \
                    [(file, 5) for file in files6]

    text_target_all = []
    for desc in [opts.domain_desc1, opts.domain_desc2, opts.domain_desc3, \
                 opts.domain_desc4, opts.domain_desc5, opts.domain_desc6]:
        target = compose_text_with_templates(desc, imagenet_templates)
        tokens = clip.tokenize(target).to(device)
        text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
        text_target = F.normalize(text_target, 2, -1)

        text_target_all.append(text_target)
    text_target_all = torch.cat(text_target_all, dim=0)
    text_target_all = text_target_all.to(dtype=torch.float32)

    
        

    relu = nn.ReLU(inplace=True)

    while True: 
    # =====  Train  =====
    
        if opts.freeze_BB:
            model.classifier.train()
        else:
            model.train()

        cur_epochs += 1

        for (im_id, tg_id, images, labels) in train_loader:
            
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            if opts.train_aug:
                mu_t_f1 = torch.zeros([opts.batch_size,256,1,1])
                std_t_f1 = torch.zeros([opts.batch_size,256,1,1])
                domain_label = torch.zeros([opts.batch_size], dtype=torch.long)
                

                random_files  = random.sample(files_all, opts.batch_size)
                mu_k = 0
                for cur_file, cur_domain in random_files:
                    with open(cur_file, 'rb') as f:
                        loaded_dict = pickle.load(f)
                        mu_t_f1[mu_k] = loaded_dict['mu_f1']
                        std_t_f1[mu_k] = loaded_dict['std_f1']
                        domain_label[mu_k] = torch.tensor(cur_domain)
                    mu_k+=1

                _, newfeature,_,_ = model.forward_low(images,mu_t_f1.to(device),std_t_f1.to(device),
                                    transfer=True,mix=opts.mix,activation=relu)


                input_text_emb = text_target_all[domain_label]
                incre_std =  model_alpha_std_project(input_text_emb)
                incre_mean =  model_alpha_mean_project(input_text_emb)
                incre_std = incre_std.unsqueeze(-1).unsqueeze(-1)
                incre_mean = incre_mean.unsqueeze(-1).unsqueeze(-1)
                incre_std.clamp_(min=0)

                new_mean, raw_new = calc_mean_std(newfeature)

                newfeature_norm = (newfeature - new_mean.expand( 
                    newfeature.shape)) / raw_new.expand( newfeature.shape)
                incre_feature = incre_std.expand(newfeature_norm.shape) * newfeature_norm \
                                    + incre_mean.expand(newfeature_norm.shape)
                revised_feature = newfeature + incre_feature * (learnable_scale).unsqueeze(-1).\
                    unsqueeze(-1).expand(newfeature_norm.shape).to(device)
                outputs = model.forward_head(revised_feature, images.shape)


            else:
                print('Please train aug')
            loss_project = 0
            loss = criterion(outputs, labels)
            loss_all = loss+ loss_project
            loss_all.backward()
            optimizer.step()
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                wandb.log(
                    {
                        'cur_epochs': cur_epochs,
                        'cur_iters': cur_itrs,
                        'segment loss': interval_loss,
                    }
                )
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0 and not opts.train_aug:

                print("validation...")
                model.eval()
               
                val_score = validate(
                    opts=opts, model=model, loader=val_loader,device=device, metrics=metrics
                    )

                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']

                if opts.freeze_BB:
                    model.classifier.train()
                else:
                    model.train()
            '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
            if (cur_itrs) % opts.val_interval == 0 and opts.train_aug:

                print("validation...")
                model.eval()
                model_project.eval()
                model_alpha_mean_project.eval()
                model_alpha_std_project.eval()
                val_score = validate(
                    opts=opts, model=model, loader=val_loader,device=device, metrics=metrics
                    )#rain

                cur_mean = val_score['Mean IoU'] 
                if cur_mean > best_score:  # save best model
                    best_score = cur_mean
                    save_ckpt(opts.ckpts_path+'/best_%s_%s.pth' %
                            (opts.model, opts.dataset))
                    model_parameters_best = copy.deepcopy(model.state_dict())
                    # Moving the model parameters to CPU (in case they are on a GPU)
                    for param_tensor in model_parameters_best:
                        model_parameters_best[param_tensor] = model_parameters_best[param_tensor].to('cpu')


                wandb.log(
                    {
                        'Train IoU':cur_mean,
                        'best':best_score,
                    }
                )

                if opts.freeze_BB:
                    model.classifier.train()
                    model_project.train()
                    model_alpha_mean_project.train()
                    model_alpha_std_project.train()

                else:
                    model.train()
                    
            if opts.train_aug and cur_itrs == opts.total_itrs:
                save_ckpt(opts.ckpts_path+'/adapted_%s_%s.pth' %
                        (opts.model, opts.dataset))
            
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                param_model_last =  copy.deepcopy(model.state_dict())
                # Moving the model parameters to CPU (in case they are on a GPU)
                for param_tensor in param_model_last:
                    param_model_last[param_tensor] = param_model_last[param_tensor].to('cpu')

                return model_parameters_best, param_model_last
    
            
def test(opts, param_best, param_last):
    metrics = StreamSegMetrics(19)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _,val_dst = get_dataset(opts.testset,opts.test_root,opts.crop_size,'rain',
                                    data_aug=opts.data_aug)
    _,val_dst2 = get_dataset(opts.testset,opts.test_root,opts.crop_size,'night',
                                    data_aug=opts.data_aug)
    _,val_dst3 = get_dataset(opts.testset,opts.test_root,opts.crop_size,'snow',
                                    data_aug=opts.data_aug)
    _,val_dst4 = get_dataset(opts.testset,opts.test_root,opts.crop_size,'fog',
                                    data_aug=opts.data_aug)


    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    val_loader2 = data.DataLoader(
        val_dst2, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    val_loader3 = data.DataLoader(
        val_dst3, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    val_loader4 = data.DataLoader(
        val_dst4, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=19, BB= opts.BB,replace_stride_with_dilation=[False,False,True])
    model.backbone.attnpool = nn.Identity()
    # cur_path = opts.ckpts_path+'/best_%s_%s.pth' % (opts.model, opts.dataset)
    # checkpoint = torch.load(cur_path, map_location=torch.device('cpu'))
    model.load_state_dict(param_best)
    model.to(device)
    model.eval()
    val_score = validate(
        opts=opts, model=model, loader=val_loader,device=device, metrics=metrics
        )#rain
    val_score_2 = validate(
                opts=opts, model=model, loader=val_loader2,device=device, metrics=metrics
        )#night
    val_score_3 = validate(
                opts=opts, model=model, loader=val_loader3,device=device, metrics=metrics
        )#snow
    val_score_4 = validate(
                opts=opts, model=model, loader=val_loader4,device=device, metrics=metrics
        )#fog
    print(metrics.to_str(val_score))
    cur_mean_4 = (val_score['Mean IoU'] +  val_score_2['Mean IoU'] + val_score_3['Mean IoU'] + val_score_4['Mean IoU'])/4

    print('#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#')

    model.load_state_dict(param_last)
    model.to(device)
    model.eval()
    val_score_last_1 = validate(
        opts=opts, model=model, loader=val_loader,device=device, metrics=metrics
        )#rain
    val_score_last_2 = validate(
                opts=opts, model=model, loader=val_loader2,device=device, metrics=metrics
        )#night
    val_score_last_3 = validate(
                opts=opts, model=model, loader=val_loader3,device=device, metrics=metrics
        )#snow
    val_score_last_4 = validate(
                opts=opts, model=model, loader=val_loader4,device=device, metrics=metrics
        )#fog

    last_mean_4 = (val_score_last_1['Mean IoU'] +  val_score_last_2['Mean IoU'] + val_score_last_3['Mean IoU'] + val_score_last_4['Mean IoU'])/4

    wandb.log(
        {
            'Best 4 mean ':cur_mean_4,
            'Best rain':val_score['Mean IoU'],
            'Best night':val_score_2['Mean IoU'],
            'Best snow':val_score_3['Mean IoU'],
            'Best fog':val_score_4['Mean IoU'],
            'Last 4 mean':last_mean_4,
            'Last rain':val_score_last_1['Mean IoU'],
            'Last night':val_score_last_2['Mean IoU'],
            'Last snow':val_score_last_3['Mean IoU'],
            'Last Fog':val_score_last_4['Mean IoU'],
        }
    )
if __name__ == '__main__':
    opts = get_argparser().parse_args()
    if not opts.test_only:
        utils.mkdir(opts.ckpts_path)
    wandb.init(
        name=opts.wandb_name,
        project=opts.wandb_project,
        entity=opts.wandb_entity,
        mode="online",
        save_code=True,
        config=opts,
        notes=opts.notes
    )
    param_best, param_last = main(opts)
    test(opts, param_best,param_last)
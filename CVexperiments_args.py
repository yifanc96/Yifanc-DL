######
# dataset, dataloader, data aug
    # name, batch size
# model, load checkpoint
    # model parameters, size
# trainer
    # optim, scheduler (warmup)
# evaluation
# entry to distributed training
# logger, SummaryWriter, checkpoint
######

import argparse
import logging
import os
import numpy as np
import random
import torch
import datetime
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import models.cct as models
from tqdm import tqdm
import math
from tensorboardX import SummaryWriter


def get_logger(level = 'INFO'):
    logging.getLogger().setLevel(logging.__dict__[level])
    
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch framework for NNs')
    # fundamental
    parser.add_argument("--randomseed", type=int, default=9999)
    parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
    
    # data
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10","cifar100"])
    parser.add_argument("--datafolder", type=str, default='./data/dataset/cifar100')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_aug", type=bool, default=True)
    
    # model: cct, ViT
    parser.add_argument("--model", type=str, default="CCT")
    parser.add_argument("--conv_size", type=int, default=3)
    parser.add_argument("--conv_layer", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=7)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=2.)
    parser.add_argument("--layerscale", type=float, default=0.0)
    parser.add_argument("--attn_dropout_rate", type=float, default=0.1)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    
    # optim
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--kinetic_lambda", type=float, default = 0.0)
    parser.add_argument("--optim_loss", type=str, default="smooth_cross_entropy", choices=["cross_entropy", "smooth_cross_entropy"])
    parser.add_argument("--optim_loss_smoothing", type=float, default=0.1)
    parser.add_argument("--optim_alg", type=str, default="AdamW", choices = ["Adam","AdamW"])
    parser.add_argument("--optim_wd", type=float, default=3e-2)
    parser.add_argument("--optim_lr", type=float, default=0.0005)
    parser.add_argument('--optim_cosine', type = bool, default = True)
    parser.add_argument("--optim_warmup", type=int, default = 0)
    
    # log
    parser.add_argument("--summarywriter", type=bool, default=True)
    parser.add_argument("--writer_logroot", type=str, default='./tblogs/')
    parser.add_argument("--checkpoint_epochs", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str, default="./results/")
    
    args = parser.parse_args()
    return args

def set_random_seeds(random_seed=0, log = True):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    if log: logging.info(f"[Seeds] random seeds: {args.randomseed}")
    
def set_device(args):
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu': args.num_gpus = 0
    args.distributed = args.num_gpus > 1
    args.local_rank = -1
    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return args

class set_Summarywriter(object):
    def __init__(self, args):
        self.log_root = args.writer_logroot
        self.log_name = ''
        date = str(datetime.datetime.now())
        self.log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(self.log_root, self.log_name, self.log_base)
    
    def get_writer(self, args):
        if not args.distributed or args.local_rank == 1:
            writer = SummaryWriter(self.log_dir)
        return writer
    
    def set_writer(self, args, **kwargs):
        # specific utility of writer
        return None

class set_data(object):
    def __init__(self):
        pass
    def data_normalize_augment(self, args, log = True):
        DATASETS = {
            'cifar10': {
                'num_classes': 10,
                'img_size': 32,
                'mean': [0.4914, 0.4822, 0.4465],
                'std': [0.2470, 0.2435, 0.2616]
            },
            'cifar100': {
                'num_classes': 100,
                'img_size': 32,
                'mean': [0.5071, 0.4867, 0.4408],
                'std': [0.2675, 0.2565, 0.2761]
            }
        }
        args.img_size = DATASETS[args.dataset]['img_size']
        args.channels = 3
        args.num_classes = DATASETS[args.dataset]['num_classes']
        args.img_mean, args.img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']
        
        normalize = [transforms.Normalize(mean=args.img_mean, std=args.img_std)]
        augmentations = []
        
        if args.data_aug:
            from utils.autoaug import CIFAR10Policy
            augmentations += [
                CIFAR10Policy()
            ]
        augmentations += [
            transforms.RandomCrop(args.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            *normalize,
            ]

        augmentations = transforms.Compose(augmentations)
        if log: logging.info(f"[Data] Dataset: {args.dataset}, path: {args.datafolder}, img_size: {args.img_size}, num_class: {args.num_classes}")
        return args, augmentations
    
    def get_trainloader(self, args, augmentations, log = True):
        train_dataset = datasets.__dict__[args.dataset.upper()](root = args.datafolder, train = True, download = True, transform = augmentations)
        
        if args.distributed:
            train_sampler = DistributedSampler(dataset=train_dataset)
            trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
        else:
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        if log: logging.info(f"[DataLoader] batch size: {args.batch_size}, augmentation: {args.data_aug}, num_workers: {args.num_workers}")
        return trainloader
    
    def get_testloader(self, args):
        normalize = [transforms.Normalize(mean=args.img_mean, std=args.img_std)]
        val_dataset = datasets.__dict__[args.dataset.upper()](
            root=args.datafolder, train=False, download=False, transform=transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                *normalize,
            ]))
        testloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)
        return testloader
    

def get_model(args, log = True):
    # for CCT
    model = models.__dict__[args.model](img_size=args.img_size, kernel_size=args.conv_size, n_input_channels=args.channels, num_classes=args.num_classes, embeding_dim=args.embed_dim, num_layers=args.num_layers,num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, n_conv_layers=args.conv_layer, drop_rate=args.dropout_rate, attn_drop_rate=args.attn_dropout_rate, drop_path_rate=args.drop_path_rate, layerscale = args.layerscale, positional_embedding='learnable')
    
    if log: logging.info(f"[Model] name: {args.model}, conv-size: {args.conv_size}, conv-layer: {args.conv_layer}, embed_dim: {args.embed_dim}, num_layers: {args.num_layers}, num_heads: {args.num_heads}, mlp_ratio: {args.mlp_ratio}, layerscale:{args.layerscale}, attn_dropout_rate: {args.attn_dropout_rate}, dropout_rate: {args.dropout_rate}, drop_path_rate: {args.drop_path_rate}")
    
    # additional info log
    tokenizer_params = sum(p.numel() for p in model.tokenizer.parameters() if p.requires_grad)
    if log: logging.info(f'[Model] num of tokenizer parameters: {tokenizer_params}')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if log: logging.info(f'[Model] num of total trainable parameters: {total_params}')
    sequence_length=model.tokenizer.sequence_length(n_channels=args.channels,
                                                           height=args.img_size,
                                                           width=args.img_size)
    if log: logging.info(f"[Model] token numbers: {sequence_length}")
    return model

def get_loss(args, log=True):
    if args.optim_loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        if log: logging.info(f"[Loss] {args.optim_loss}")
    elif args.optim_loss == "smooth_cross_entropy":
        from utils.losses import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=args.optim_loss_smoothing).to(args.device)
        if log: logging.info(f"[Loss] {args.optim_loss}, smoothing: {args.optim_loss_smoothing}")
    return criterion

def get_optimizer(args, model, log=True):
    if args.optim_alg == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr)
        if log: logging.info(f"[Optimizer] {args.optim_alg}, lr: {args.optim_lr}")
    elif args.optim_alg == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr,
                                  weight_decay=args.optim_wd)
        if log: logging.info(f"[Optimizer] {args.optim_alg}, lr: {args.optim_lr}, wd: {args.optim_wd}")
    return optimizer

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.optim_lr
    if hasattr(args, 'optim_warmup') and epoch < args.optim_warmup:
        lr = lr / (args.optim_warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.optim_warmup) / (args.train_num_epochs - args.optim_warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train_one_epoch():
    return None

def evaluate():
    return None

def store_checkpoint():
    return None


if __name__ == '__main__':
    
    ## get logger
    get_logger(level = 'INFO')
    
    ## get argument parser
    args = get_parser()
    
    ## get device
    args = set_device(args)
    log = not args.distributed or args.local_rank == 1
    logging.info(f"[Device] device: {args.device}, num_gpus: {args.num_gpus}, distributed: {args.distributed}") # log for all devices
    
    ## set random seed
    set_random_seeds(args.randomseed, log)
    
    ## get dateset and loader
    data = set_data()
    args, augmentations = data.data_normalize_augment(args, log)
    trainloader = data.get_trainloader(args, augmentations, log)
    testloader = data.get_testloader(args)
    
    ## get model
    model = get_model(args, log)
    model = model.to(args.device)
    
    ## get loss
    criterion = get_loss(args, log)
    
    ## get optimizer, scheduler
    optimizer = get_optimizer(args, model, log)
    
    ## training
    if log:
        train_accs = []
        test_accs = []
    
    depth = model.classifier.depth if hasattr(model, 'classifier') else model.module.classifier.depth 
    v_collect = [torch.torch.empty(0).cuda(args.device) for i in range(2*depth)]
    # save the norm of outputs of each layer
    x_norm_collect = [torch.torch.empty(0).cuda(args.device) for i in range(2*depth)]
    cosine_similarity = [torch.torch.empty(0).cuda(args.device) for i in range(2*depth)]
    def save_outputs_hook(layer_id):
        def fn(_, input, output):
            v_collect[layer_id] = output - input[0]
            x_norm_collect[layer_id] = torch.norm(output, dim=(1,2)).mean()
            inres_sim = (input[0]*v_collect[layer_id]).sum(dim=(1,2))/(torch.norm(input[0], dim=(1,2))*torch.norm(output - input[0], dim=(1,2))+ 1e-5)
            # inres_sim_all = (input[0]*v_collect[layer_id]).sum()/(torch.norm(input[0])*torch.norm(output - input[0])+ 1e-5)
            cosine_similarity[layer_id] = inres_sim.mean()
        return fn
    for iter_i in range(depth):
        if hasattr(model, 'classifier'):
            model.classifier.blocks_attn[iter_i].register_forward_hook(save_outputs_hook(iter_i))
            model.classifier.blocks_MLP[iter_i].register_forward_hook(save_outputs_hook(iter_i+depth))
        else:
            model.module.classifier.blocks_attn[iter_i].register_forward_hook(save_outputs_hook(iter_i))
            model.module.classifier.blocks_MLP[iter_i].register_forward_hook(save_outputs_hook(iter_i+depth))
    
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args)
        running_loss, running_accuracy = train(model, train_dataloader, criterion, optimizer, local_rank, device, v_collect, kinetic_lambda = kinetic_lambda)
        if local_rank == 0: 
            print(f"Epoch : {epoch+1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
            train_accs.append(running_accuracy)
            writer.add_scalar('training/training loss', running_loss, epoch)
            writer.add_scalar('training/training accuracy', running_accuracy, epoch)

        if test_dataloader is not None and local_rank == 0:
            test_loss, test_accuracy = evaluation(model, test_dataloader, criterion, device, x_norm_collect, cosine_similarity, writer, epoch, nepochs_save)
            print(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")
            test_accs.append(test_accuracy)
            writer.add_scalar('test/test loss', test_loss, epoch)
            writer.add_scalar('test/test accuracy', test_accuracy, epoch)
        if (epoch+1)%nepochs_save == 0 and local_rank == 0:
            # torch.save(model.state_dict(), save_path)
            torch.save({
                'epoch': epoch,
                'train_acc': train_accs,
                'test_acc': test_accs,
                'lambda': kinetic_lambda
                }, save_path) 
    
    
    ## get SummaryWriter
    meter = set_Summarywriter(args)
    writer = meter.get_writer(args)
    if log: logging.info(f"[SummaryWriter] Directory: {meter.log_dir}")
    
    
    
    






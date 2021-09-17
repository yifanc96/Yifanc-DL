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



    
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch framework for NNs')
    # fundamental
    parser.add_argument("--randomseed", type=int, default=9999)
    parser.add_argument("--local_rank", type=int, default=0,
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
    parser.add_argument("--optim_warmup", type=int, default = 10)
    
    # log
    parser.add_argument("--summarywriter", type=bool, default=True)
    parser.add_argument("--writer_logroot", type=str, default='./tblogs/')
    parser.add_argument("--checkpoint_epochs", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str, default="./results/")
    
    args = parser.parse_args()
    return args

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)
    if args.log: logging.info(f"[Seeds] random seeds: {args.randomseed}")
    
def set_device(args):
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu': args.num_gpus = 0
    args.distributed = args.num_gpus > 1
    args.local_rank = 0
    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    logging.info(f"[Device] device: {args.device}, num_gpus: {args.num_gpus}, distributed: {args.distributed}") # log for all devices
    return args

class set_data(object):
    def __init__(self):
        pass
    def data_normalize_augment(self, args):
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
        if args.log: logging.info(f"[Data] Dataset: {args.dataset}, path: {args.datafolder}, img_size: {args.img_size}, num_class: {args.num_classes}")
        return args, augmentations
    
    def get_trainloader(self, augmentations, args):
        train_dataset = datasets.__dict__[args.dataset.upper()](root = args.datafolder, train = True, download = True, transform = augmentations)
        
        if args.distributed:
            train_sampler = DistributedSampler(dataset=train_dataset)
            trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
        else:
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        if args.log: logging.info(f"[DataLoader] batch size: {args.batch_size}, augmentation: {args.data_aug}, num_workers: {args.num_workers}")
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
    
def get_model(args):
    # for CCT
    model = models.__dict__[args.model](img_size=args.img_size, kernel_size=args.conv_size, n_input_channels=args.channels, num_classes=args.num_classes, embeding_dim=args.embed_dim, num_layers=args.num_layers,num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, n_conv_layers=args.conv_layer, drop_rate=args.dropout_rate, attn_drop_rate=args.attn_dropout_rate, drop_path_rate=args.drop_path_rate, layerscale = args.layerscale, positional_embedding='learnable')
    
    if args.log: logging.info(f"[Model] name: {args.model}, conv-size: {args.conv_size}, conv-layer: {args.conv_layer}, embed_dim: {args.embed_dim}, num_layers: {args.num_layers}, num_heads: {args.num_heads}, mlp_ratio: {args.mlp_ratio}, layerscale:{args.layerscale}, attn_dropout_rate: {args.attn_dropout_rate}, dropout_rate: {args.dropout_rate}, drop_path_rate: {args.drop_path_rate}")
    
    # additional info log
    tokenizer_params = sum(p.numel() for p in model.tokenizer.parameters() if p.requires_grad)
    if args.log: logging.info(f'[Model] num of tokenizer parameters: {tokenizer_params}')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.log: logging.info(f'[Model] num of total trainable parameters: {total_params}')
    sequence_length=model.tokenizer.sequence_length(n_channels=args.channels,
                                                           height=args.img_size,
                                                           width=args.img_size)
    if args.log: logging.info(f"[Model] token numbers: {sequence_length}")
    return model

def get_loss(args):
    if args.optim_loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        if args.log: logging.info(f"[Loss] {args.optim_loss}")
    elif args.optim_loss == "smooth_cross_entropy":
        from utils.losses import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=args.optim_loss_smoothing).to(args.device)
        if args.log: logging.info(f"[Loss] {args.optim_loss}, smoothing: {args.optim_loss_smoothing}")
    return criterion

def get_optimizer(model, args):
    if args.optim_alg == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr)
        if args.log: logging.info(f"[Optimizer] {args.optim_alg}, lr: {args.optim_lr}")
    elif args.optim_alg == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr,
                                  weight_decay=args.optim_wd)
        if args.log: logging.info(f"[Optimizer] {args.optim_alg}, lr: {args.optim_lr}, wd: {args.optim_wd}")
    return optimizer

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.optim_lr
    if hasattr(args, 'optim_warmup') and epoch < args.optim_warmup:
        lr = lr / (args.optim_warmup - epoch)
    elif args.optim_cosine:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.optim_warmup) / (args.train_num_epochs - args.optim_warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class set_logger(object):
    def __init__(self):
        pass
    
    def logger(self, level = 'INFO'):
        logging.getLogger().setLevel(logging.__dict__[level])
        
    def get_writer(self, args):
        log_root = args.writer_logroot
        log_name = args.model + '_' + args.dataset + '_' + args.optim_alg + '_'+ args.optim_loss + '_'+'b'+ str(args.batch_size)+ 'd' + str(args.num_layers) + 'e' + str(args.embed_dim) + 'h' + str(args.num_heads) + 'm' + str(int(args.mlp_ratio)) + 'sd' + str(args.drop_path_rate).replace(".","")
        if args.kinetic_lambda > 0.0:
            log_name += '_k'+str(int(args.kinetic_lambda))
        if args.layerscale > 0.0:
            log_name += '_ls'+str(args.layerscale).replace(".","")
        date = str(datetime.datetime.now())
        log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
        self.log_dir = os.path.join(log_root, log_name, log_base)
        
        if not args.distributed or args.local_rank == 0:
            writer = SummaryWriter(self.log_dir)
            if args.log: logging.info(f"[SummaryWriter] Directory: {meter.log_dir}")
        return writer
    
    def set_hook(self, model, args):
        # specific utility of writer
        depth = args.num_layers
        self.v_collect = [torch.empty(0, device = args.device) for i in range(2*depth)]
        # save the norm of outputs of each layer
        self.x_norm_collect = [torch.empty(0, device = args.device) for i in range(2*depth)]
        self.cosine_similarity = [torch.empty(0, device = args.device) for i in range(2*depth)]
        def save_outputs_hook(layer_id):
            def fn(_, input, output):
                self.v_collect[layer_id] = output - input[0]
                self.x_norm_collect[layer_id] = torch.norm(output, dim=(1,2)).mean()
                inres_sim = (input[0]*self.v_collect[layer_id]).sum(dim=(1,2))/(torch.norm(input[0], dim=(1,2))*torch.norm(output - input[0], dim=(1,2))+ 1e-5)
                self.cosine_similarity[layer_id] = inres_sim.mean()
            return fn
        for iter_i in range(depth):
            if hasattr(model, 'classifier'):
                model.classifier.blocks_attn[iter_i].register_forward_hook(save_outputs_hook(iter_i))
                model.classifier.blocks_MLP[iter_i].register_forward_hook(save_outputs_hook(iter_i+depth))
            else:
                model.module.classifier.blocks_attn[iter_i].register_forward_hook(save_outputs_hook(iter_i))
                model.module.classifier.blocks_MLP[iter_i].register_forward_hook(save_outputs_hook(iter_i+depth))
        return model
    
    
def train_one_epoch(model, trainloader, criterion, optimizer, meter, args):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    if args.log: trainloader = tqdm(trainloader)
    
    for data, target in trainloader:
        data = data.to(args.device)
        target = target.to(args.device)

        output = model(data)
        
        transport = args.kinetic_lambda * sum([torch.mean(v ** 2) for v in meter.v_collect])
        loss = criterion(output, target) + transport
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        acc = (output.argmax(dim=1) == target).float().mean()
        running_accuracy += acc / len(trainloader)
        running_loss += loss.item() / len(trainloader)

    return running_loss, running_accuracy

def evaluate(model, testloader, criterion, meter, writer, args):
    model.eval()
    depth = args.num_layers
    x_norm = np.zeros(2*depth)
    cos_similarity = np.zeros(2*depth)
    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        for data, target in tqdm(testloader):
            data = data.to(args.device)
            target = target.to(args.device)

            output = model(data)  
            for i in range(2*depth):
                x_norm[i] += meter.x_norm_collect[i].item()/len(testloader)
                cos_similarity[i] += meter.cosine_similarity[i].item()/len(testloader)
            loss = criterion(output, target)
            acc = (output.argmax(dim=1) == target).float().mean()
            test_accuracy += acc / len(testloader)
            test_loss += loss.item() / len(testloader)
            
    for i in range(depth):
        writer.add_scalar(f"attn_norm/depth{i}", x_norm[i],epoch)
        writer.add_scalar(f"attn_cosim/depth{i}", cos_similarity[i],epoch)
        writer.add_scalar(f"MLP_norm/depth{i}", x_norm[i+depth],epoch)
        writer.add_scalar(f"MLP_cosim/depth{i}", cos_similarity[i+depth],epoch)
    return test_loss, test_accuracy

def store_checkpoint():
    return None


if __name__ == '__main__':
    
    ## set logger
    meter = set_logger()
    meter.logger(level = 'INFO')
    
    ## get argument parser
    args = get_parser()
    
    ## get device
    args = set_device(args)
    args.log = not args.distributed or args.local_rank == 0
    
    ## set random seed
    set_random_seeds(args.randomseed)
    
    ## get dateset and loader
    data = set_data()
    args, augmentations = data.data_normalize_augment(args)
    trainloader = data.get_trainloader(augmentations, args)
    testloader = data.get_testloader(args)
    
    ## get model
    model = get_model(args)
    model = model.to(args.device)
    
    ## get loss
    criterion = get_loss(args)
    
    ## get optimizer, scheduler
    optimizer = get_optimizer(model, args)
    
    ## get SummaryWriter
    writer = meter.get_writer(args)
    model = meter.set_hook(model, args)
    
    ## training
    if args.log:
        train_accs = []
        test_accs = []

    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        running_loss, running_accuracy = train_one_epoch(model, trainloader, criterion, optimizer, meter, args)
        if args.log: 
            logging.info(f"Epoch : {epoch+1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
            train_accs.append(running_accuracy)
            writer.add_scalar('training/training loss', running_loss, epoch)
            writer.add_scalar('training/training accuracy', running_accuracy, epoch)

        if testloader is not None and args.log:
            test_loss, test_accuracy = evaluate(model, testloader, criterion, meter, writer, args)
            logging.info(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")
            test_accs.append(test_accuracy)
            writer.add_scalar('test/test loss', test_loss, epoch)
            writer.add_scalar('test/test accuracy', test_accuracy, epoch)
            
        if (epoch+1)% args.checkpoint_epochs == 0 and args.log:
            # torch.save(model.state_dict(), save_path)
            torch.save({
                'epoch': epoch,
                'train_acc': train_accs,
                'test_acc': test_accs,
                }, args.checkpoint_path) 
    
    
    
    
    
    
    






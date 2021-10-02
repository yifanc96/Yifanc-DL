import os

os.system("CUDA_VISIBLE_DEVICES=0 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model ViT --conv_size 2 &")

os.system("CUDA_VISIBLE_DEVICES=1 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --dataset cifar100 --model ViT --conv_size 2 &")

os.system("CUDA_VISIBLE_DEVICES=2 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model ViT --layerscale 0.1 --train_scale --conv_size 2 &")

os.system("CUDA_VISIBLE_DEVICES=3 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model ViT --layerscale 0.1 --conv_size 2 &")
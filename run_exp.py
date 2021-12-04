import os

# os.system("CUDA_VISIBLE_DEVICES=0 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --seq_pool &")

# os.system("CUDA_VISIBLE_DEVICES=1 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --dataset cifar100 --model CCT --seq_pool &")

# os.system("CUDA_VISIBLE_DEVICES=2 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.1 --train_scale --seq_pool &")

os.system("CUDA_VISIBLE_DEVICES=0 nohup python CVexperiments.py --num_epochs 1000 --optim_alg Adam --optim_warmup 0 --optim_cosine --dataset cifar10 --model ViT --norm_lambda 1.0 &")

os.system("CUDA_VISIBLE_DEVICES=1 nohup python CVexperiments.py --num_epochs 1000 --optim_alg Adam --optim_warmup 0 --optim_cosine --dataset cifar10 --model ViT --norm_lambda 10.0 &")

os.system("CUDA_VISIBLE_DEVICES=2 nohup python CVexperiments.py --num_epochs 1000 --optim_alg Adam --optim_warmup 0 --optim_cosine --dataset cifar10 --model ViT --norm_lambda 0.1 &")

os.system("CUDA_VISIBLE_DEVICES=3 nohup python CVexperiments.py --num_epochs 1000 --optim_alg Adam --optim_warmup 0 --optim_cosine --dataset cifar10 --model ViT &")

# os.system("CUDA_VISIBLE_DEVICES=2 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.01 &")

# os.system("CUDA_VISIBLE_DEVICES=3 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.01 --train_scale &")
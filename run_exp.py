import os

# os.system("CUDA_VISIBLE_DEVICES=0 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --seq_pool &")

# os.system("CUDA_VISIBLE_DEVICES=1 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --dataset cifar100 --model CCT --seq_pool &")

# os.system("CUDA_VISIBLE_DEVICES=2 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.1 --train_scale --seq_pool &")

os.system("CUDA_VISIBLE_DEVICES=0 nohup python CVexperiments.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.1 --linearscale 10 &")

os.system("CUDA_VISIBLE_DEVICES=1 nohup python CVexperiments.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.1 --linearscale 20 &")

os.system("CUDA_VISIBLE_DEVICES=2 nohup python CVexperiments.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.01 --linearscale 10 &")

os.system("CUDA_VISIBLE_DEVICES=3 nohup python CVexperiments.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.01 --linearscale 20 &")

# os.system("CUDA_VISIBLE_DEVICES=2 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.01 &")

# os.system("CUDA_VISIBLE_DEVICES=3 nohup python CVexperiments_args.py --num_epochs 1000 --optim_alg AdamW --optim_warmup 0 --optim_cosine --dataset cifar100 --model CCT --layerscale 0.01 --train_scale &")
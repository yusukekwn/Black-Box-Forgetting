#!/bin/bash

# custom config
dataset='cub' #select dataset 'cifar10', 'cifar100', 'cub', 'imagenet30', 'imagenet'

prompt_type=$1 # select prompt type 'lcs', 'bbt', 'ind (ours w/o lcs)'

# dimension of latent context
dim_l_ctx=$2 # 500 when lcs, 125 when BBT and ours w/o lcs

n_ctx=16 # number of context

# to independently optimize unique latent context
# cma for shared latent context is specified separately internally
total_number_of_cma=$3 # 1 when 'bbt', number of context when 'lcs' and 'ours w/o lcs'. 

pop_size=20 # population size for CMA-ES
iteration=800
kshot=1

export CUDA_VISIBLE_DEVICES=0

# only when prompt_type='lcs', must be specified
dim_slc=400 # dimension of shared latent context
dim_ulc=100 # dimension of unique latent context
  
mkdir ./result-${dataset}/
mkdir ./result-${dataset}/${prompt_type}-nctx-${n_ctx}/
save_dir=./result-${dataset}/${prompt_type}-nctx-${n_ctx}/num-fgt-cls-80/
mkdir ${save_dir}
python main.py \
 --dataset ${dataset} \
 --dim_l_ctx ${dim_l_ctx} \
 --dim_slc ${dim_slc} \
 --dim_ulc ${dim_ulc} \
 --total_number_of_cma ${total_number_of_cma} \
 --n_ctx ${n_ctx} \
 --popsize ${pop_size} \
 --iteration ${iteration} \
 --save_dir ${save_dir} \
 --prompt_type ${prompt_type} \
 --kshot ${kshot}

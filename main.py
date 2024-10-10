import argparse
import time
import random
import torch
import numpy as np
import cma
import pickle
from dataloader import CIFAR10Loader, CIFAR100Loader, CUBLoader, Imagenet30Loader, ImagenetLoader
from torch.utils.data import DataLoader
from train import LMForwardAPI
from utils import construct_true_few_shot_data
from collections import OrderedDict


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    save_dir = args.save_dir
    path_file = save_dir + 'log.txt'

    class_token_position = 'end'

    sigma = 1
    
    popsize = args.popsize
    kshot = args.kshot
    
    iteration = args.iteration
    dataset = args.dataset
    
    # forgetting classes list
    if args.fgt_cls_list: # If you want to change forgetting classes, you can be set by argument.
        fgt_cls_list = args.fgt_cls_list
    else:
        fgt_cls_list = []
    
    if dataset == 'cifar10':
        classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        train_data, test_data = CIFAR10Loader.load()
        train_data, valid_data = construct_true_few_shot_data(train_data, k_shot=kshot)

        if len(fgt_cls_list) == 0:
            # forgetting classes list
            for i in range(4):
                fgt_cls_list.append(i)

    elif dataset == 'cifar100':
        with open('./data/cifar-100/cifar100-clsnames.txt', 'rb') as p:
            classnames = pickle.load(p)

        train_data, test_data = CIFAR100Loader.load()
        train_data, valid_data = construct_true_few_shot_data(train_data, k_shot=kshot)
        
        if len(fgt_cls_list) == 0:
            # forgetting classes list
            for i in range(40):
                fgt_cls_list.append(i)

    elif dataset == 'cub':
        classnames =[]
        with open("./data/cub/cub-classes.txt", encoding="UTF-8") as f:
            for line in f:
                line_tmp = line.split()[1][4:]
                line_tmp = line_tmp.replace('_', ' ')
                classnames.append(line_tmp)
        
        train_dataset = CUBLoader(train=True, download=True)
        test_data = CUBLoader(train=False, download=True)  
        new_test_data = []
        for i in range(len(test_data)):
            new_test_data.append(test_data.__getitem__(i))
        test_data = new_test_data
        train_data, valid_data = construct_true_few_shot_data(train_dataset, k_shot=kshot)
        
        if len(fgt_cls_list) == 0:
            # forgetting classes list
            for i in range(80):
                fgt_cls_list.append(i)
                
    elif dataset == 'imagenet30':
        classnames = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock', 'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover', 'mosque', 'nail', 'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile', 'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
        
        train_data, test_data = Imagenet30Loader.load()
        train_data, valid_data = construct_true_few_shot_data(train_data, k_shot=kshot)
        
        if len(fgt_cls_list) == 0:
            # forgetting classes list
            for i in range(12):
                fgt_cls_list.append(i)

    elif dataset == 'imagenet':
        classnames = OrderedDict()
        with open('./data/imagenet/classnames.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        classnames = list(classnames.values())
        
        train_data, test_data = ImagenetLoader.load()
        train_data, valid_data = construct_true_few_shot_data(train_data, k_shot=kshot)

        if len(fgt_cls_list) == 0:
            # forgetting classes list
            for i in range(400):
                fgt_cls_list.append(i)           

    batch_size = args.batch_size

    trainloader = DataLoader(train_data,batch_size=batch_size,shuffle=True, num_workers=2,pin_memory=True)
    validloader = DataLoader(valid_data,batch_size=batch_size,shuffle=True, num_workers=2,pin_memory=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True)

    dim_l_ctx = args.dim_l_ctx
    dim_slc = args.dim_slc
    dim_ulc=args.dim_ulc
    number_of_context = args.n_ctx
    total_number_of_cma = args.total_number_of_cma
    
    prompt_type = args.prompt_type
    if prompt_type == 'lcs':
        output_of_cma = dim_ulc
    elif prompt_type == 'bbt':
        output_of_cma = dim_l_ctx * number_of_context
    elif prompt_type == 'ind': # when 'w/o lcs'
        output_of_cma = dim_l_ctx
        
    eval_only = args.eval_only
    
    model_forward = LMForwardAPI(
        save_dir=save_dir,
        path_file=path_file,
        device=device,
        class_token_position=class_token_position,
        dim_l_ctx=dim_l_ctx,
        number_of_context=number_of_context,
        dim_slc=dim_slc,
        dim_ulc=dim_ulc,
        classnames=classnames,
        fgt_cls_list=fgt_cls_list,
        batch_size=batch_size,
        testloader=testloader,
        trainloader=trainloader,
        validloader=validloader,
        prompt_type=prompt_type,
        iteration=iteration
    )

    cma_opts = {
        'seed': args.seed,
        'popsize': popsize,
        'maxiter': iteration,
        'verbose': -1,
    }

    es_list = [cma.CMAEvolutionStrategy(output_of_cma * [0], sigma, inopts=cma_opts) for _ in range(total_number_of_cma)]
    if prompt_type == 'lcs':   
        # for shared latent context 
        es_slc = cma.CMAEvolutionStrategy(dim_slc * [0], sigma ,inopts=cma_opts)
        es_list.append(es_slc)
    print('Population Size: {}'.format(es_list[0].popsize))
    print('{} Evaluation.'.format('Serial'))
    
    if eval_only == 'yes':
        load_epoch = args.load_epoch
        prompt_embedding = torch.load(save_dir + str(load_epoch) + '.pt')
        test_cls_acc, test_harmonic_mean, err_for, acc_mem = model_forward.eval(prompt_embedding=prompt_embedding, test_data=testloader)
        show_test_acc = "Test Acc: |"
        for clsacc in test_cls_acc:
            show_test_acc += f' {clsacc:.4f} |'
        print(show_test_acc + '\n')
        print('Test acc: {}'.format(test_harmonic_mean, 4) + '\n')
        print('for acc: {}'.format(err_for, 4) + '\n')
        print('mem acc: {}'.format(acc_mem, 4) + '\n\n\n')
        
        with open(save_dir + '/' + 'eval_only_' + str(load_epoch) + '.txt', mode='a') as f:
            f.write(str(fgt_cls_list) + '\n')
            f.write('\nEvaluate on test data')
            f.write('\n' + show_test_acc + '\n')
            f.write('Test acc: {}'.format(test_harmonic_mean, 4) + '\n')
            f.write('for acc: {}'.format(err_for, 4) + '\n')
            f.write('mem acc: {}'.format(acc_mem, 4) + '\n\n\n')
        
    else:
        start_time = time.time()
        while not es_list[0].stop():
            solutions_list = []
            fitness = []
            for es in es_list:
                solutions_list.append(es.ask())
            for epch in range(popsize):
                learnable_ctx = []
                for ctx_vec in solutions_list:
                    learnable_ctx = np.append(learnable_ctx, ctx_vec[epch])
                fitness.append(model_forward.eval(learnable_ctx))
            for es, solutions in zip(es_list, solutions_list): 
                es.tell(solutions, fitness) 

        end_time = time.time()
        print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
        print('Evaluate on test data...')

        test_cls_acc, test_harmonic_mean, err_for, acc_mem = model_forward.eval(test_data=testloader)
        show_test_acc = "Test Acc: |"
        for clsacc in test_cls_acc:
            show_test_acc += f' {clsacc:.4f} |'
        print(show_test_acc + '\n')
        print('Test harmonic mean: {}'.format(test_harmonic_mean, 4) + '\n')
        print('err for: {}'.format(err_for, 4) + '\n')
        print('acc mem: {}'.format(acc_mem, 4) + '\n\n\n')

        with open(path_file, mode='a') as f:
            f.write(str(fgt_cls_list) + '\n')
            f.write(str(model_forward.best_epoch[-1]) + '\n')
            f.write('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
            f.write('\nEvaluate on test data')
            f.write('\n' + show_test_acc + '\n')
            f.write('Test harmonic mean: {}'.format(test_harmonic_mean, 4) + '\n')
            f.write('err for: {}'.format(err_for, 4) + '\n')
            f.write('acc mem: {}'.format(acc_mem, 4) + '\n\n\n')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fgt_cls_list", type=int, nargs='*', help='list of forgetting class')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--kshot", type=int)    
    parser.add_argument("--dim_l_ctx", type=int, help='dimension of latent context')
    parser.add_argument("--dim_slc", type=int, help='dimension of shared latent context')
    parser.add_argument("--dim_ulc", type=int, help='dimension of unique latent context')
    parser.add_argument("--total_number_of_cma", type=int)
    parser.add_argument("--n_ctx", type=int)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--prompt_type", type=str, help='type of method')
    parser.add_argument("--eval_only", type=str, default='no')
    parser.add_argument("--load_epoch", type=str, default=None)
    args = parser.parse_args()
    main(args)

import os
import copy
import clip
from tqdm import tqdm
import torch
import numpy as np
from transformers import CLIPConfig
from clip import clip
from src.customclip import CustomCLIP
from utils import crossentropyloss_max

class LMForwardAPI:
    @torch.no_grad()
    def __init__(
        self,
        save_dir,
        path_file,
        device, 
        class_token_position, 
        dim_l_ctx,
        number_of_context,
        dim_slc,
        dim_ulc,
        classnames, 
        fgt_cls_list,
        batch_size,
        testloader,
        trainloader,
        validloader,
        prompt_type,
        iteration
    ):  
        self.save_dir = save_dir
        self.path_file = path_file
        self.iteration = iteration
        self.device = device

        backbone_name = "ViT-B/16"
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, root=os.path.expanduser("~/.cache/clip"))

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        self.clip_model = clip.build_model(state_dict or model.state_dict())
        self.config = CLIPConfig()
        
        with torch.no_grad():
            vocab = torch.cat([self.clip_model.token_embedding(torch.tensor([i])) for i in range(self.clip_model.vocab_size)])
            vocab_mean = torch.mean(vocab, dim=0)
            self.linear = torch.nn.Linear(dim_l_ctx, self.config.projection_dim, bias=False)
            mu_hat = torch.mean(vocab).item()
            self.std_hat = torch.std(vocab).item()
            
        mu = 0
        print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, self.std_hat, mu, self.std_hat))
        # A
        for p in self.linear.parameters():
            torch.nn.init.normal_(p, mu, self.std_hat)
        #
                
        self.best_train_harmonic_mean = 0.0
        self.best_valid_harmonic_mean = 0.0
        self.best_prompt = None
        self.num_call = 0
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_prompt = vocab_mean.type(torch.float16)
        self.classnames = classnames
        self.fgt_cls_list = fgt_cls_list
        self.best_epoch = []
        self.batch_size = batch_size
        self.class_token_position = class_token_position
        self.dim_l_ctx = dim_l_ctx
        self.n_ctx = number_of_context
        self.dim_slc = dim_slc
        self.dim_ulc = dim_ulc
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.prompt_type = prompt_type
        
    def evl_acc_test(
        self, 
        testloader
    ):
        pr_cls = np.zeros(len(self.classnames))
        gt_cls = np.zeros(len(self.classnames))
        
        err_for = 0
        acc_mem = 0
        gt_for = 0
        gt_mem = 0
                            
        with torch.no_grad():    
            for images, labels in tqdm(testloader):
                self.model.eval()
                images = images.to(self.device)
                labels = labels.to(self.device)
    
                similarity, _ = self.model(images)
                
                similarity = self.softmax(similarity)
                    
                for idx_pr, idx_gt in zip(similarity,labels):
                    if torch.argmax(idx_pr).item()  == idx_gt:
                        pr_cls[int(idx_gt)] += 1
                    if idx_gt in self.fgt_cls_list:
                        gt_for += 1
                        if torch.argmax(idx_pr).item()  != idx_gt:
                            err_for += 1
                    else:
                        gt_mem += 1
                        if torch.argmax(idx_pr).item()  == idx_gt:
                            acc_mem += 1
                    gt_cls[int(idx_gt)] += 1
            
            err_for = err_for / gt_for
            acc_mem = acc_mem / gt_mem
            harmonic_mean = (2 * err_for * acc_mem) / (err_for + acc_mem)
            
        return pr_cls / gt_cls, harmonic_mean, err_for, acc_mem
    
    def eval(
        self, 
        prompt_embedding=None,
        test_data=None,
    ):
        with torch.no_grad():       
            tmp_prompt = copy.deepcopy(prompt_embedding)            
            
            self.num_call += 1
            if prompt_embedding is None:
                prompt_embedding = self.best_prompt
                torch.save(self.best_prompt, self.save_dir + 'best_prompt' + '.pt')
            
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32) # z
            
            if self.prompt_type == "lcs":
                l_ctx_embedding = torch.zeros(self.n_ctx, self.dim_l_ctx) 
                slc_embegging = prompt_embedding[-self.dim_slc:]
                ulc_embedding = prompt_embedding[:self.dim_ulc*self.n_ctx]
                ulc_embedding = ulc_embedding.reshape(self.n_ctx, self.dim_ulc)
                
                for i in range(self.n_ctx):
                    l_ctx_embedding[i] = torch.cat((slc_embegging,ulc_embedding[i]), dim=0)   
                
                # Az
                prompt_embedding = self.linear(l_ctx_embedding).type(torch.float16)                
                
            else: # bbt and ours w/o lcs
                prompt_embedding = prompt_embedding.reshape(self.n_ctx, self.dim_l_ctx)
                
                # Az
                prompt_embedding = self.linear(prompt_embedding).type(torch.float16)
            
            #           
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
                prompt_embedding = prompt_embedding.type(torch.float16)
            #

            self.model = CustomCLIP(classnames=self.classnames, clip_model=self.clip_model, n_ctx=self.n_ctx, class_token_position=self.class_token_position, learnable_z=prompt_embedding)
            
            self.model.to(self.device)
            
            if test_data is not None:
                test_cls_acc, harmonic_mean, err_for, acc_mem = self.evl_acc_test(testloader=self.testloader)
                
                return test_cls_acc, harmonic_mean, err_for, acc_mem

            else:                
                losses = []
                gt_for = 0
                gt_mem = 0
                err_for = 0
                acc_mem = 0
                gt_cls = np.zeros(len(self.classnames))
                pr_cls = np.zeros(len(self.classnames))
                
                for images, labels in self.trainloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    for_lis = []
                    mem_lis = []    
                    
                    outputs, _ = self.model(images)
                    soft_out = self.softmax(outputs)
                    
                    max_entropy = 1/ len(self.classnames)
                    
                    for out, (idx, lbl) in zip(soft_out, enumerate(labels)):
                        if lbl in self.fgt_cls_list:
                            gt_for += 1
                            for_lis = np.append(for_lis,idx)
                            if torch.argmax(out).item() != lbl :
                                err_for += 1
                        else:
                            gt_mem += 1
                            mem_lis = np.append(mem_lis,idx)
                            if torch.argmax(out).item() == lbl :
                                acc_mem += 1
                        
                        if torch.argmax(out).item() == lbl :
                            pr_cls[int(lbl)] += 1
                        gt_cls[int(lbl)] += 1

                    for_lis = torch.tensor(for_lis).type(torch.long)
                    mem_lis = torch.tensor(mem_lis).type(torch.long)
                                            
                    loss = self.ce_loss(outputs[mem_lis], labels[mem_lis])
                    
                    if len(for_lis) != 0:
                        err_loss = crossentropyloss_max(soft_out[for_lis], max_entropy)
                        loss = loss + err_loss
                    
                    losses = np.append(losses, loss.item())    
                Loss = losses.sum() / len(losses)
                
                err_for = err_for / gt_for
                acc_mem = acc_mem / gt_mem

                train_harmonic_mean = (2 * err_for * acc_mem) / (err_for + acc_mem)
                cls_accs = pr_cls / gt_cls
                                
                if train_harmonic_mean > self.best_train_harmonic_mean:
                    self.best_train_harmonic_mean = train_harmonic_mean
                train_acc_show = "Class acc: |"
                for cls_acc in cls_accs:
                    train_acc_show += f" {cls_acc:.4f} |"                    

                print('[# API Calls {}] loss: {}. Current harmonic mean: {}. Best harmonic mean so far: {}. train err for: {}. train acc mem {}.'.format(
                    self.num_call,
                    round(float(Loss), 4),
                    round(float(train_harmonic_mean), 4),
                    round(float(self.best_train_harmonic_mean), 4),
                    round(float(err_for), 4),
                    round(float(acc_mem), 4)))
                print(train_acc_show)
                
                with open(self.path_file, mode='a') as f:
                    f.write('[# API Calls {}] loss: {}. Current harmonic mean: {}. Best harmonic mean so far: {}. train err for: {}. train acc mem {}.'.format(
                    self.num_call,
                    round(float(Loss), 4),
                    round(float(train_harmonic_mean), 4),
                    round(float(self.best_train_harmonic_mean), 4),
                    round(float(err_for), 4),
                    round(float(acc_mem), 4)))
                    f.write('\n' + train_acc_show)
                    

                print('********* Evaluated on valid set *********')
                with open(self.path_file, mode='a') as f:
                    f.write('\n********* Evaluated on valid set *********\n')
            
                dev_losses = []
                gt_for = 0
                gt_mem = 0
                err_for = 0
                acc_mem = 0
                gt_cls = np.zeros(len(self.classnames))
                pr_cls = np.zeros(len(self.classnames))
                
                for images, labels in self.validloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs, _ = self.model(images)
                    
                    for_lis = []
                    mem_lis = []
                    
                    soft_out = self.softmax(outputs)
                    
                    max_entropy = 1 / len(self.classnames)
                
                    for out, (idx, lbl) in zip(soft_out, enumerate(labels)):
                        if lbl in self.fgt_cls_list:
                            gt_for += 1
                            for_lis = np.append(for_lis,idx)
                            if torch.argmax(out).item() != lbl:
                                err_for += 1
                        else:
                            gt_mem += 1
                            mem_lis = np.append(mem_lis,idx)
                            if torch.argmax(out).item() == lbl :
                                acc_mem += 1
                        
                        if torch.argmax(out).item() == lbl :
                            pr_cls[int(lbl)] += 1
                        gt_cls[int(lbl)] += 1
                        
                    for_lis = torch.tensor(for_lis).type(torch.long)
                    mem_lis = torch.tensor(mem_lis).type(torch.long)
                
                    dev_loss = self.ce_loss(outputs[mem_lis], labels[mem_lis])

                    if len(for_lis) != 0:
                        dev_err_loss = crossentropyloss_max(soft_out[for_lis], max_entropy)
                        dev_loss = dev_loss + dev_err_loss
                    dev_losses = np.append(dev_losses, dev_loss.item())                          
                dev_Loss = dev_losses.sum() / len(dev_losses)

                cls_accs = pr_cls / gt_cls
                
                err_for = err_for / gt_for
                acc_mem = acc_mem / gt_mem

                # harmonic mean of err_for and acc_mem
                valid_harmonic_mean = (2 * err_for * acc_mem) / (err_for + acc_mem)
                
                if valid_harmonic_mean > self.best_valid_harmonic_mean:
                    self.best_valid_harmonic_mean = valid_harmonic_mean
                    self.best_prompt = tmp_prompt
                    self.best_epoch = np.append(self.best_epoch, self.num_call)
                    torch.save(self.best_prompt, self.save_dir + 'best' + str(self.num_call) + '.pt')
                    
                dev_acc_show = "Class acc: |"
                for cls_acc in cls_accs:
                    dev_acc_show += f" {cls_acc:.4f} |"
                print('Valid loss: {}. valid harmonic mean: {}. Best valid harmonic mean: {}. valid err for: {}. valid acc mem {}.'.format(
                    round(float(dev_Loss), 4),
                    round(float(valid_harmonic_mean), 4),
                    round(float(self.best_valid_harmonic_mean), 4),
                    round(float(err_for), 4),
                    round(float(acc_mem), 4)))
                print(dev_acc_show)
                print('********* Done *********')
                
                with open(self.path_file, mode='a') as f:
                    f.write('Valid loss: {}. valid harmonic mean: {}. Best valid harmonic mean: {}. valid err for: {}. valid acc mem {}.'.format(
                    round(float(dev_Loss), 4),
                    round(float(valid_harmonic_mean), 4),
                    round(float(self.best_valid_harmonic_mean), 4),
                    round(float(err_for), 4),
                    round(float(acc_mem), 4)))
                    f.write('\n' + dev_acc_show)
                    f.write('\n********* Done *********\n') 
            return Loss

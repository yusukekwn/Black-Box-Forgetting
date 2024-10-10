import torch
import numpy as np

def construct_true_few_shot_data(train_data, k_shot):
    train_label_count = {}
    valid_label_count = {}
    new_train_data = []
    new_valid_data = []
    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        label = train_data[index][1]
        if label < 0:
            continue

        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in valid_label_count:
            valid_label_count[label] = 0

        if train_label_count[label] < k_shot:
            new_train_data.append(train_data[index])
            train_label_count[label] += 1
        elif valid_label_count[label] < k_shot:
            new_valid_data.append(train_data[index])
            valid_label_count[label] += 1
    
    return new_train_data, new_valid_data

def crossentropyloss_max(outputs, maxentropy):
    loss = 0
    
    eps = 1e-7
    
    # 1 / (the number of the classes to be forgotten)
    t = maxentropy
    
    loss = torch.sum((t * torch.log(outputs + eps)), 1)
    
    loss = torch.sum(loss) / len(loss)
    
    return -loss
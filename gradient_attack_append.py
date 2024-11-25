import os
import time
import csv
import sys
import yaml
import numpy as np
import pandas as pd
from src.util import ExeDataset, write_pred
from src.model import MalConv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
import time


# Load config file for experiment

config_path = 'config/example.yaml' #needs to modify to point to a new list of valid label
seed = int(123)
conf = yaml.load(open(config_path, 'r'), Loader = yaml.SafeLoader)

use_gpu = conf['use_gpu']
use_cpu = conf['use_cpu']
exp_name = conf['exp_name'] + '_sd_' + str(seed)
batch_size = conf['batch_size']

data_path = '/home/user/Desktop/CodeCaveFinal-main/CodeCaveMalware/UPX9_Cave8192_UPX1/'
# checkpoint_dir = conf['checkpoint_dir']
# chkpt_acc_path = checkpoint_dir + exp_name + '.model'


validloader = DataLoader(ExeDataset(data_path),
                         batch_size=1, shuffle=False, num_workers=12)

malconv = torch.load('/home/user/Desktop/Retraining_Malconv/checkpoint/Retrain_all_samples_sd_850.model', map_location=torch.device('cuda') if use_gpu else 'cpu')
malconv = malconv.cuda() if use_gpu else malconv

print("Loading MalConv model successful")

history = {}
history['val_loss'] = []
history['val_acc'] = []
history['val_pred'] = []
bce_loss = nn.BCEWithLogitsLoss().cuda() if use_gpu else nn.BCEWithLogitsLoss()
total=0
evade = 0
changes=[]
temp_df = pd.DataFrame()

for _, val_batch_data in enumerate(validloader):
    total+=1
    cur_batch_size = val_batch_data[0].size(0)
    print("cur batch size:", total)

    exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]

    data = exe_input[0].cpu().numpy()

    caveStart = data[-2]
    caveSize = data[-1]
    caveEnd = int(caveStart) + int(caveSize)
    length = data[-3]

    data = data[:length]
    data = np.concatenate([data, np.random.randint(0, 256, 2000000 - length)])

    init_prob = 0
 
    label = val_batch_data[1].cuda() if use_gpu else val_batch_data[1]
 
    label = Variable(label.float(), requires_grad=False)

    embed = malconv.embed
    sigmoid = nn.Sigmoid()
    count_j = 0
    t=0

    exe_input = torch.from_numpy(np.array([data])).long().cuda() if use_gpu else torch.from_numpy(np.array([data])).long()
    exe_input = Variable(exe_input.long(), requires_grad=False)
    pred = malconv(exe_input)
    prob = sigmoid(pred).cpu().data.numpy()[0][0]

    for t in range(5):                                     
        exe_input = torch.from_numpy(np.array([data])).long().cuda() if use_gpu else torch.from_numpy(np.array([data])).long()
        
        # exe_input = Variable(exe_input.long(), requires_grad=False)

        pred = malconv(exe_input)
        
        prob = sigmoid(pred).cpu().data.numpy()[0][0]
        
        print("prob: ", prob)
        if t==0:
            init_prob = prob
        if prob < 0.5:
            print("prob<0.5,success.")
            evade+=1
            print("evading rate:",evade/float(total))
            break

        loss = bce_loss(pred, label)

        if (caveStart+8192) >=2000000:
            print("Perturbtation out of bound")
            continue
        loss.backward()
        w = malconv.embed_x.grad[0].data
        z = malconv.embed_x.data[0]

        print("Total malware size: ",length)
        end = int(8192+length)
        print("Inserting the perturbation of size: ",int(end-length))
 

        for j in range(caveStart, caveEnd):
            if j % 1024 == 0:
                exe_input = torch.from_numpy(np.array([data])).long().cuda() if use_gpu else torch.from_numpy(np.array([data])).long()
                #exe_input = Variable(exe_input.long(), requires_grad=False)
                pred = malconv(exe_input)
                prob = sigmoid(pred).cpu().data.numpy()[0][0]
                print("prob: ", prob)
                count_j = j
                if prob < 0.5:
                    break
                print("changing " + str(j) + "th byte")

            try:
                min_index = -1
                min_di = int(end)
                wj = -w[j:j + 1, :]
                nj = wj / torch.norm(wj, 2)
                zj = z[j:j + 1, :]
                for i in range(1, 256):
                    mi = embed(Variable(torch.from_numpy(np.array([i])))).data
                    si = torch.matmul((nj), torch.t(mi - zj))
                    di = torch.norm(mi - (zj + si * nj))
                    si = si.cpu().numpy()
                    if si > 0 and di < min_di:
                        min_di = di
                        min_index = i
                if min_index != -1:
                    data[j] = min_index
                    changes.append(min_index)
            except:
                continue
        print("finish ", t) 

    with open("Codecave_UPX9_8192.csv", 'a') as file:
        writer = csv.writer(file)
        data = [val_batch_data, length, t, count_j, init_prob, prob ]
        writer.writerow(data)


    print("Reported result to a file")

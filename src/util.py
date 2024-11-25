import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import random

def write_pred(test_pred,test_idx,file_path):
    test_pred = [item for sublist in test_pred for item in sublist]
    with open(file_path,'w') as f:
        for idx,pred in zip(test_idx,test_pred):
            print(idx.upper()+','+str(pred[0]))

def load_malware_data(csv_path):
    # Load CSV file only once and cache the relevant columns
    df = pd.read_csv(csv_path)
    df.columns = ['Malware', 'Starting_address', 'Size', 'Status', 'Flag']
    
    # Create a dictionary for fast lookups
    malware_dict = {
        row['Malware']: (int(row['Starting_address']), int(row['Size']))
        for _, row in df.iterrows()
    }
    
    return malware_dict

def check_malware(malware, malware_dict):
    # Look up malware in the dictionary for fast access
    return malware_dict.get(malware, (0, 0))


class ExeDataset(Dataset):
    def __init__(self, data_path,csv_path):
        self.data_path = data_path
        self.fp_list = [
            f for f in os.listdir(data_path) 
            if os.path.isfile(os.path.join(data_path, f)) and 
            os.path.getsize(os.path.join(data_path, f)) < 2 * 1000 * 1000  # 2 MB in bytes
        ]
        self.first_n_byte = 2000000
        self.csv_path = csv_path
        self.malware_dict = load_malware_data(self.csv_path)

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):

        file_path = os.path.join(self.data_path, self.fp_list[idx])
        # print("Wroking to load file")
        try:
            with open(file_path,'rb') as f:
                tmp = [i+1 for i in f.read()]
                length=len(tmp)
                tmp=tmp+[0]*(self.first_n_byte-len(tmp)-1)
                tmp=tmp+[length]
                starting_address, size = check_malware(self.fp_list[idx], self.malware_dict)
                tmp = tmp + [starting_address] + [size]
        except Exception as e:
            print(f"Error handling {file_path}: {e}")
            tmp=[]
            length=0

        return np.array(tmp),np.array([1])
 

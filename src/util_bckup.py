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

def check_malware(malware):
    # Load CSV file
    df = pd.read_csv(r"/home/user/Desktop/CodeCaveFinal-main/KkrunchyCodeCave/Cave4096_kkrunchy2_caves.csv")
    df.columns = ['Malware', 'Starting_address', 'Size', 'Status', 'Flag']
    
    # Try to find the row that matches the malware name
    match_row = df[df['Malware'] == malware]
    
    if not match_row.empty:
        # Extract the starting address and size
        starting_address = int(match_row['Starting_address'].values[0])
        size = int(match_row['Size'].values[0])
        return (starting_address, size)
    else:
        # Return (0, 0) if no match is found
        return (0, 0)

class ExeDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.fp_list = [
            f for f in os.listdir(data_path) 
            if os.path.isfile(os.path.join(data_path, f)) and 
            os.path.getsize(os.path.join(data_path, f)) < 2 * 1000 * 1000  # 2 MB in bytes
        ]
        self.first_n_byte = 2000000

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
                # a,b = check_malware(self.fp_list[idx])
                # tmp = tmp + [a] + [b]
        except Exception as e:
            print(f"Error handling {file_path}: {e}")
            tmp=[]
            length=0

        return np.array(tmp),np.array([1])
 

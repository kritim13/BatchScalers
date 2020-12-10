from faker import Factory
import pandas as pd
from tqdm.notebook import tqdm
import os
import numpy as np

class BatchStandardScaler():
    """
    Standard scaler for batch processing. 
    
    Needs to be declared before the batch processing starts
    """
    """https://stats.stackexchange.com/questions/133138/will-the-mean-of-a-set-of-means-always-be-the-same-as-the-mean-obtained-from-the"""
    def __init__(self,
                 cols,
                 batch_size=1000):
        self.batch_size=batch_size
        self.cols=cols
        self.batch_params={x:{'sum':[],'count':[]} for x in self.cols}
        self.std_params={x:{'sum':[],'count':[]} for x in self.cols}
    
    def split_file(path):
        return
    
    def get_mean_params(self,path,fname):
        """
        run this only on the training batches
        """

        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            
            sum_=batch[self.cols].sum()
            count_=batch[self.cols].count()     
            for x,y,z in zip(sum_.index,sum_,count_):
                self.batch_params[x]['sum'].append(y)
                self.batch_params[x]['count'].append(z)
            
            del batch, sum_, count_
        
    def get_pop_mean(self):
        self.mean_=[]
        for x in self.cols:  
            pop_sum_=np.sum(self.batch_params[x]['sum'])
            pop_count_=np.sum(self.batch_params[x]['count'])
            self.mean_.append(pop_sum_/pop_count_)
     
    def get_sd_params(self,path,fname):
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            sq_diff_= (batch[self.cols]-self.mean_)**2
            sum_=sq_diff_.sum()    
            count_=sq_diff_.count()            
            for x,y,z in zip(sum_.index,sum_,count_):
                self.std_params[x]['sum'].append(y)
                self.std_params[x]['count'].append(z)
            
            del batch, sum_, count_, sq_diff_
            
    def get_pop_std(self):
        self.sd_=[]
        for x in self.cols:
            pop_sq_sum_=np.sum(self.std_params[x]['sum'])
            pop_n_=np.sum(self.std_params[x]['count'])
            self.sd_.append(np.sqrt(pop_sq_sum_/pop_n_))       
    
    def fit(self,path,fname):
        
        print('computing mean...')
        self.get_mean_params(path,fname)
        self.get_pop_mean()
        print('MEAN COMPUTED')
        
        print('comuting std dev....')
        self.get_sd_params(path,fname)
        self.get_pop_std()
        print('STD DEV COMPUTED')
        
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            batch[self.cols]=(batch[self.cols]-self.mean_)/self.sd_
            batch[self.cols].to_csv(os.path.join(path,f'{fname}.csv'),index=False)
            
    def transform(self,path,fname):
        for batch in tqdm(pd.read_csv(os.path.join(path,fname), chunksize=self.batch_size)):
            batch[self.cols]=(batch[self.cols]-self.mean_)/self.sd_
            batch[self.cols].to_csv(os.path.join(path,f'{fname}.csv'),index=False)
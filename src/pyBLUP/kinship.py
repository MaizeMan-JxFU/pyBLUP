import numpy as np
from .QC import QC
class KIN:
    def __init__(self,SNP:np.ndarray,method:str='VanRanden'):
        '''
        rowname: indv
        colname: SNP
        '''
        # print(f'Initializing of kinship (Method:{method})...')
        # print(f'Number of SNP: {SNP.shape[1]}')
        qc = QC(SNP)
        SNP = qc.simple_QC()
        self.SNP_retain = qc.SNP_retain
        # print(f'Number of effective SNP: {SNP.shape[1]}')
        self.sample_size = SNP.shape[0]
        p_i:np.ndarray = SNP.sum(axis=0)/(2*self.sample_size) # 每个SNP的次等位基因频率
        SNP[:,p_i>.5] = 2 - SNP[:,p_i>.5] # 矫正每个SNP的次等位基因
        p_i[p_i>.5] = 1-p_i[p_i>.5] # 矫正每个SNP的次等位基因
        self.p_i = p_i.astype(np.float32)
        self.SNP_mean = SNP.mean(axis=0)
        self.SNP_std = SNP.std(axis=0)
        self.method = method
        self.SNP = SNP
        pass
    def kinship(self, SNP:np.ndarray=None) -> np.ndarray:
        SNP = self.SNP.astype(np.float32) if SNP is None else SNP.astype(np.float32)
        method = self.method
        p_i = self.p_i
        if method == 'VanRanden':
            Z:np.ndarray = SNP - 2*p_i
            p_sum = 2*np.sum(p_i*(1-p_i))
            return Z@Z.T/p_sum
        elif method == 'gemma1':
            Z:np.ndarray = SNP - self.SNP_mean
            return Z@Z.T/Z.shape[1]
        elif method == 'gemma2':
            Z:np.ndarray = (SNP - self.SNP_mean)/self.SNP_std
            return Z@Z.T/Z.shape[1]
        elif method == 'pearson':
            return np.corrcoef(SNP)
    def chunk_kinship(self,split_num:int=4) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        # o = int(split_num*(split_num-1)/2)
        # print(f'Runing by {split_num} matrix(O={o})...')
        SNP = self.SNP
        chunks = np.linspace(0,self.sample_size,split_num,dtype=int)
        kin = np.zeros(shape=(self.sample_size,self.sample_size),dtype=np.float32)
        for ind1 in range(len(chunks)-1):
            for ind2 in range(ind1,len(chunks)-1):
                SNP_sub = np.concatenate([SNP[chunks[ind1]:chunks[ind1+1],:],SNP[chunks[ind2]:chunks[ind2+1],:]],axis=0,dtype=np.float32) # 分块计算 kinship
                kin[chunks[ind1]:chunks[ind1+1],chunks[ind2]:chunks[ind2+1]] = self.kinship(SNP_sub)[:chunks[ind1+1]-chunks[ind1],chunks[ind1+1]-chunks[ind1]:]
        return np.triu(kin,k=0)+np.triu(kin,k=1).T
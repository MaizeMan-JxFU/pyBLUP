import numpy as np

class QC:
    def __init__(self,M:np.ndarray):
        self.SNP = M
        pass
    def simple_QC(self,missf = 0.05,maff = 0.02):
        SNP = self.SNP
        missr = (SNP<0).sum(axis=0)/SNP.shape[0] # 统计每个SNP的缺失率
        maf = SNP.sum(axis=0)/(2*SNP.shape[0]) # 统计每个SNP的maf
        self.SNP_retain = (missr<=missf)&(maf>=maff)&(maf<=(1-maff)) # 保留缺失率低于5%且maf大于2%的SNP
        SNP = SNP[:,self.SNP_retain]
        missr = (SNP<0).sum(axis=0)/SNP.shape[0] # 每个SNP的缺失率
        if (missr>0).sum()>0: # 存在缺失值进行简单均值填充
            for i,mr in enumerate(missr):
                if mr > 0:
                    SNP_col = SNP[:,i]
                    SNP[SNP_col<0,i] = np.sum(SNP_col[SNP_col>=0])/np.sum(SNP_col>=0) # 为每列SNP 填充缺失填
        return SNP
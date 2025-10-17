import numpy as np
import time
np.random.seed(2025)
def mlm_test() -> None:
    from pyBLUP import BLUP
    snp_num = 10000
    sample_num = 2000
    pve = 0.5
    sigmau = 1
    x = np.zeros(shape=(sample_num,snp_num)) # 0,1,2 of SNP
    for i in range(snp_num):
        maf = np.random.uniform(0.05,0.5)
        x[:,i] = np.random.binomial(2,maf,size=sample_num)
    u = np.random.normal(0,sigmau,size=(snp_num,1)) # effect of SNP 服从正态分布
    g = x @ u
    e = np.random.normal(0,np.sqrt((1-pve)/pve*(g.var())),size=(sample_num,1))
    y = g + e
    for i in [None,'pearson','VanRanden','gemma1','gemma2']:
        _ = []
        _hat = []
        t = time.time()
        model = BLUP(y,None,x,kinship=i)
        model.fit()
        print((time.time()-t)/60,'mins')
        # break
        y_hat = model.predict(None,x)
        _+=y.tolist()
        _hat+=y_hat.tolist()
        real_pred = np.concatenate([np.array(_),np.array(_hat)],axis=1)
        print(f'{i}({round(model.pve,3)})',np.corrcoef(real_pred,rowvar=False)[0,1])

if __name__ == "__main__":
    pass
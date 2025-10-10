import numpy as np

def rsqr(M:np.ndarray):
    '''
    :param M: Marker matrix: rownames are individuals and colnames are markers
    :returns rsqr: LD matrix of r-square
    
    D = P(AB)-P(A)P(B)
    
    rsqr = D^2/(P(A)(1-P(A))P(B)(1-P(B)))
    '''
    n = M.shape[0]
    Pab = M.T@M/(4*n)
    Pa:np.ndarray = M.sum(axis=0,keepdims=True)/(2*n)
    D = Pab - Pa.T@Pa
    PaM1minusPa = Pa*(1-Pa)
    cov = PaM1minusPa.T@PaM1minusPa
    return np.power(D,2)/cov


if __name__ == "__main__":
    np.random.seed(88)
    snp_num = 3000
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
    print(rsqr(x))
import numpy as np
from .QC import QC
def rsqr(M:np.ndarray,phased:bool=False):
    '''
    :param M: Marker matrix: rownames are individuals and colnames are markers
    :returns rsqr: LD matrix of r-square (digenic LD)
    
    Phased method for bialleles(Heterozygous)
    
    Vens, M., Ziegler, A. (2017). Estimating Disequilibrium Coefficients. In: Elston, R. (eds) Statistical Human Genetics.
    Methods in Molecular Biology, vol 1666. Humana Press, New York, NY. https://doi.org/10.1007/978-1-4939-7274-6_7

    Gaunt, T.R., Rodríguez, S. & Day, I.N. Cubic exact solutions for the estimation of pairwise haplotype frequencies: implications for linkage disequilibrium analyses and a web tool 'CubeX'. 
    BMC Bioinformatics 8, 428 (2007). https://doi.org/10.1186/1471-2105-8-428
    '''
    M = M.astype(np.float32)
    qc = QC(M)
    M = qc.simple_QC()
    if phased:
        # n = M.shape[0]
        # Pab = M.T@M/(4*n)
        # Pa:np.ndarray = M.sum(axis=0,keepdims=True)/(2*n)
        # D = Pab - Pa.T@Pa
        # PaM1minusPa = Pa*(1-Pa)
        # cov = PaM1minusPa.T@PaM1minusPa
        return None # np.power(D,2)/cov
    else:
        return np.corrcoef(M,rowvar=False)**2

if __name__ == "__main__":
    np.random.seed(32)
    snp_num = 3000
    sample_num = 2000
    pve = 0.5
    sigmau = 1
    x = np.zeros(shape=(sample_num,snp_num)) # 0,1,2 of SNP
    for i in range(snp_num):
        maf = np.random.uniform(0.02,0.5)
        x[:,i] = np.random.binomial(2,maf,size=sample_num)
    u = np.random.normal(0,sigmau,size=(snp_num,1)) # effect of SNP 服从正态分布
    g = x @ u
    e = np.random.normal(0,np.sqrt((1-pve)/pve*(g.var())),size=(sample_num,1))
    y = g + e
    print(np.tril(rsqr(x,),-1)[:4,:4])
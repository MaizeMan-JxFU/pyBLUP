import numpy as np
from scipy.optimize import minimize_scalar
import scipy

class GWAS:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None,M:np.ndarray=None,kinship:np.ndarray=None):
        '''Fast Solve of Mixed Linear Model by Brent.
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param Z: Designed matrix for random effect nxq\n
        :param M: Marker matrix (0,1,2 of SNP)\n
        :param kinship: Calculation method of kinship matrix ('VanRanden','pearson','gemma1','gemma2')
        '''
        X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1)) # 设计矩阵 或 n1 向量
        self.X = X
        self.y = y
        self.M = M
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.G = kinship+1e-6*np.eye(self.n) # 添加正则项 确保矩阵正定
        # 简化G矩阵求逆
        D,S,Dh = np.linalg.svd(self.G)
        self.Dh = Dh
        self.X = Dh@self.X
        self.y = Dh@self.y
        self.S = np.diag(S)
        self._fit()
        pass
    def _REML(self,lbd: float):
        '''暂时不写 ML'''
        n,p = self.n,self.p
        V = self.S+lbd*np.eye(n)
        V_inv = np.diag(1/np.diag(V))
        XTV_invX = self.X.T@V_inv@self.X
        XTV_invy = self.X.T@V_inv@self.y
        self.beta = np.linalg.solve(XTV_invX,XTV_invy)
        r = self.y - self.X@self.beta
        rTV_invr = r.T@V_inv@r
        c = (n-p)*(np.log(n-p)-1-np.log(2*np.pi))/2
        log_detV = np.sum(np.log(np.diag(V)))
        sign, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
        total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX
        self.V,self.V_inv,self.r,self.sigma2 = V,V_inv,r,rTV_invr/(n-p)
        return c - total_log / 2
    def _fit(self,):
        self.X = self.X
        self.result = minimize_scalar(lambda lbd: -self._REML(lbd),bounds=(1e-6,1e6),method='bounded') # 寻找lbd 最大化似然函数
        self.lbd = self.result.x
        Vg = np.trace(self.S)/self.n
        Ve = self.lbd
        self.pve = Vg/(Vg+Ve)
        self.V_inv = self.V_inv/self.sigma2
    def gwas(self,snp:np.ndarray):
        self.snp = np.concatenate([self.X,self.Dh@snp],axis=1)
        XTV_invX = self.snp.T@self.V_inv@self.snp + 1e-6*np.eye(self.p+1)
        XTV_invy = self.snp.T@self.V_inv@self.y
        beta = np.linalg.solve(XTV_invX,XTV_invy)[-1,0]
        snp_se = np.sqrt(np.linalg.inv(XTV_invX)[-1,-1])
        return beta,snp_se
    
if __name__ == '__main__':
    pass
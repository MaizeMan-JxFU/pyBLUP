import numpy as np
from scipy.optimize import minimize_scalar
from .kinship import KIN
__info__ = ' a package of solving mixed linear model based on numpy. '
xx = len(__info__)+2
__info__ = '#'*xx+f'''\n#{' '*(xx-2)}#\n#{__info__}#\n#{' '*(xx-2)}#\n'''+'#'*xx



class MLM:
    def __init__(self,X:np.ndarray,y:np.ndarray):
        self.X_train,self.y_train = X, y
        pass
    def fit_lm(self,):
        '''
        Simple linear model
        '''
        X,y = self.X_train,self.y_train
        assert X.shape[0] == y.shape[0]
        X = np.concatenate([X,np.ones((X.shape[0],1))],axis=1)
        self.n = X.shape[0]
        self.factor = X.shape[1]
        XTX = X.T@X
        XTy = X.T@y
        self.beta = np.linalg.solve(XTX,XTy) # the effects of factors
        r = y-X@self.beta
        rTr = r.T@r
        self.SE = np.sqrt(rTr*np.diag(np.linalg.solve(XTX,np.eye(self.factor)))/(self.n-self.factor))
        self.R2 = 1-(rTr/self.n)/np.var(y)
    def fit_mlm(self,):
        
        return 
    def pred(self, X:np.ndarray):
        X = np.concatenate([X,np.ones(X.shape)],axis=1)
        return X@self.beta
    
    
class BLUP:
    def __init__(self,y:np.ndarray=None,X:np.ndarray=None,M:np.ndarray=None,Z:np.ndarray=None, kinship:str=None):
        '''Fast Solve of Mixed Linear Model by Brent.
        
        :param y: Phenotype nx1\n
        :param X: Designed matrix for fixed effect nxp\n
        :param Z: Designed matrix for random effect nxq\n
        :param M: Marker matrix (0,1,2 of SNP)\n
        :param kinship: Calculation method of kinship matrix ('VanRanden','pearson','gemma1','gemma2')
        '''
        Z = Z if Z is not None else np.eye(y.shape[0]) # 设计矩阵 或 单位矩阵(一般没有重复则采用单位矩阵)
        X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1) if X is not None else np.ones((y.shape[0],1)) # 设计矩阵 或 n1 向量
        assert M.shape[0] == Z.shape[1] # 随机效应和效应值相同
        self.X = X
        self.y = y
        self.M = M
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.kinship = kinship # 确保训练和预测的kinship方法一致
        if self.kinship is not None:
            kmodel = KIN(M,self.kinship)
            self.G = kmodel.chunk_kinship()
            self.G+=1e-6*np.eye(self.G.shape[0]) # 添加正则项 确保矩阵正定
            self.Z = Z
        else:
            self.G = np.eye(M.shape[1])
            self.Z = Z@M
        # 简化G矩阵求逆
        D,S,Dh = np.linalg.svd(self.Z@self.G@self.Z.T)
        self.X = Dh@self.X
        self.y = Dh@self.y
        self.Z = Dh@self.Z
        self.S = np.diag(S)
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
        signX, log_detXTV_invX = np.linalg.slogdet(XTV_invX)
        total_log = (n-p)*np.log(rTV_invr) + log_detV + log_detXTV_invX
        self.V,self.V_inv,self.r,self.sigma2 = V,V_inv,r,rTV_invr/(n-p)
        return c - total_log / 2
    def fit(self,):
        self.X = self.X
        self.result = minimize_scalar(lambda lbd: -self._REML(lbd),bounds=(1e-6,1e6),method='bounded') # 寻找lbd 最大化似然函数
        Vg = np.trace(self.S)/self.n
        Ve = self.result.x[0,0]
        self.pve = Vg/(Vg+Ve)
        self.u = self.G@self.Z.T@self.V_inv@self.r
        return
    def predict(self,X:np.ndarray=None,M:np.ndarray=None):
        X = np.concatenate([np.ones((X.shape[0],1)),X],axis=1) if X is not None else np.ones((M.shape[0],1)) # 设计矩阵 或 n1 向量
        if self.kinship is not None:
            kmodel = KIN(np.concatenate([self.M, M]),self.kinship)
            G = kmodel.chunk_kinship()
            G+=1e-6*np.eye(G.shape[0]) # 添加正则项 确保矩阵正定
            return X@self.beta+G[self.n:, :self.n]@np.linalg.solve(self.G,self.u)
        else:
            return X@self.beta+M@self.u
    
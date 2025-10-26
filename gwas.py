from pyBLUP import QK,GWAS
from gfreader import breader,vcfreader
from bioplotkit import GWASPLOT,sci_set
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys
import os
def format_dataframe_for_export(df:pd.DataFrame, scientific_cols=None, float_cols=None):
    """
    Parameters:
    - df: raw DataFrame
    - scientific_cols: 科学计数法列
    - float_cols: 浮点数列
    """
    df_export = df.copy()
    # 科学计数法
    if scientific_cols:
        for col in scientific_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.int64]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4e}")
    # 浮点数
    if float_cols:
        for col in float_cols:
            if col in df_export.columns and df_export[col].dtype in [np.float64, np.int64]:
                df_export[col] = df_export[col].apply(lambda x: f"{x:.4f}")
    return df_export
t = time.time()

gfile,phenofile,outfolder = sys.argv[1],sys.argv[2],sys.argv[3]
kinship_method = sys.argv[4] if len(sys.argv)>=5 else 'VanRanden'# {'VanRanden', 'gemma1', 'gemma2', 'pearson'}
qdim = int(sys.argv[5]) if len(sys.argv)>=6 else 3
HighAC = bool(sys.argv[6]) if len(sys.argv)>=7 else False
if not os.path.exists(outfolder):
    os.makedirs(outfolder,mode=0o755)
prefix = gfile.replace('.vcf','').replace('.gz','')
print(f'Loading genotype from {gfile}...')
geno = vcfreader(rf'{gfile}').iloc[:,2:].T if gfile[-4:] == '.vcf' or gfile[-7:] == '.vcf.gz' else breader(rf'{gfile}').iloc[:,2:].T # PLINK格式
geno.index = geno.index.astype(str)
snp_chrloc = geno.columns
famid = geno.index
print(f'Loading phenotype from {phenofile}...')
pheno = pd.read_csv(rf'{phenofile}',sep='\t') # 第一列是样本ID, 第一行是表型名
pheno = pheno.groupby(pheno.columns[0]).mean() # 重复样本表型取均值
pheno.index = pheno.index.astype(str)
print('Geno and Pheno are ready!')

if not os.path.exists(f'{prefix}.k.{kinship_method}.txt') or not os.path.exists(f'{prefix}.q.{qdim}.txt'):
    qkmodel = QK(geno.values,low_memory=True,log=True)
    print('Samples and SNP:',geno.shape)
if os.path.exists(f'{prefix}.k.{kinship_method}.txt'):
    print(f'* Loading K matrix from {prefix}.k.{kinship_method}.txt...')
    kmatrix = pd.read_csv(f'{prefix}.k.{kinship_method}.txt',sep=r'\s+',header=None).values
else:    
    print(f'* Calculation method of kinship matrix is {kinship_method}')
    kmatrix = qkmodel.kinship(method=kinship_method)
    np.savetxt(f'{prefix}.k.{kinship_method}.txt',kmatrix,fmt='%.6f')
print(kmatrix[:5,:5])
print(kmatrix.shape)

if os.path.exists(f'{prefix}.q.{qdim}.txt'):
    print(f'* Loading Q matrix from {prefix}.q.{qdim}.txt...')
    qmatrix = pd.read_csv(f'{prefix}.q.{qdim}.txt',sep=r'\s+',header=None).values
else:    
    print(f'* Dimension of PC for q matrix is {qdim}')
    qmatrix,eigenval = qkmodel.rpca(dim=qdim)
    np.savetxt(f'{prefix}.q.{qdim}.txt',qmatrix,fmt='%.6f')
print(qmatrix[:5,:5]) if qdim > 5 else print(qmatrix[:5,:qdim])
print(qmatrix.shape)

# sci_set()
for i in pheno.columns:
    print('*'*50)
    p = pheno[i].dropna()
    famid_pheno = [i for i in famid if i in p] # 对齐样本 以geno顺序为准
    famid_geno = [i for i in range(len(famid)) if famid[i] in famid_pheno] # 对齐样本 以geno顺序为准
    p = p.loc[famid_pheno].values.reshape(-1,1)
    if len(p)>0:
        gwasmodel = GWAS(y=p,X=qmatrix[famid_geno,:],kinship=kmatrix[famid_geno,:][:,famid_geno])
        print(f'''phenotype:{i}, Number of samples:{len(famid_geno)}, Number of SNP:{geno.shape[1]}, pve of null:{round(gwasmodel.pve,3)}, high AC model: {HighAC}''')
        if HighAC:
            results = gwasmodel.gwasHAC(snp=geno.values[famid_geno,:],chunksize=200_000) # gwas running...
            np.savetxt(f'{outfolder}/{i}.lbd',np.array(gwasmodel.lbd),fmt='%.4f')
        else:
            results = gwasmodel.gwas(snp=geno.values[famid_geno,:],chunksize=200_000) # gwas running...
        results = pd.DataFrame(results,columns=['beta','se','p'],index=snp_chrloc[gwasmodel.snp_retain]).reset_index()
        results_save = format_dataframe_for_export(results, scientific_cols=['p'], float_cols=['beta','se'])
        results_save.to_csv(f'{outfolder}/{i}.tsv',sep='\t',index=False)
        print(f'Saved in {outfolder}/{i}.tsv')
        
        manhan = GWASPLOT(results,'#CHROM','POS','p')
        plt.figure(figsize=(12,4),dpi=300)
        ax1 = plt.subplot(1,2,1)
        manhan.manhattan(-np.log10(0.05/results.shape[0]),ax=ax1)
        ax2 = plt.subplot(1,2,2)
        manhan.qq(ax=ax2)
        plt.tight_layout()
        print('Visualizing...')
        plt.savefig(f'{outfolder}/{i}.png')
        print(f'Saved in {outfolder}/{i}.png\n')
        del results,results_save,manhan,gwasmodel,p,famid_pheno,famid_geno
    else:
        print(f'Phenotype {i} has no overlapping samples with genotype, please check sample id. skipped.\n')
lt = time.localtime()
print(f'\nFinished, Total time: {round(time.time()-t,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}')

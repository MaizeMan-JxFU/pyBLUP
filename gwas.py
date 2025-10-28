from pyBLUP import QK,GWAS
from gfreader import breader,vcfreader
from bioplotkit import GWASPLOT,sci_set
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import time
import socket
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
def green_output(string:str):
    return f'\033[92m{string}\033[0m'
def main(log:bool=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docfile = os.path.join(script_dir,'doc','demo.txt')
    doc = open(docfile, 'r',).read()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=doc
    )
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument('--vcf', type=str, 
                           help='Input genotype file in VCF format (.vcf or .vcf.gz)')
    geno_group.add_argument('--bfile', type=str, 
                           help='Input genotype files in PLINK binary format (prefix for .bed, .bim, .fam)')
    required_group.add_argument('--pheno', type=str, required=True,
                               help='Phenotype file (tab-delimited with sample IDs in first column)')
    required_group.add_argument('--out', type=str, required=True,
                               help='Output directory for results')
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument('--grm', type=str,
                               default='VanRanden',
                               help='Kinship matrix calculation method or path to pre-calculated GRM file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--qcov', type=str, default='3',
                               help='Number of principal components for Q matrix or path to covariate matrix file '
                                   '(default: %(default)s)')
    optional_group.add_argument('--thread', type=int, default=-1,
                               help='Number of CPU threads to use (-1 for all available cores, default: %(default)s)')
    optional_group.add_argument('--AC', action='store_true', default=True,
                               help='Enable HighAC mode for GWAS (default: %(default)s)')
    optional_group.add_argument('--no-AC', action='store_false', dest='AC',
                               help='Disable HighAC mode')
    args = parser.parse_args()
    # Determine genotype file
    gfile = args.vcf if args.vcf else args.bfile
    # Build argument list for the original script
    sys.argv = [
        sys.argv[0],  # script name
        gfile,
        args.pheno,
        args.out,
        args.grm,
        args.qcov,
        str(args.AC)
    ]
    # Print configuration summary
    if log:
        print("\n" + "="*60)
        print("GWAS LMM SOLVER CONFIGURATION")
        print("="*60)
        print(f"Genotype file:    {gfile}")
        print(f"Phenotype file:   {args.pheno}")
        print(f"Output directory: {args.out}")
        print(f"GRM method:       {args.grm}")
        print(f"Q matrix:         {args.qcov}")
        print(f"Threads:          {args.thread} ({'All cores' if args.thread == -1 else 'User specified'})")
        print(f"HighAC mode:      {args.AC}")
        print("="*60 + "\n")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.out):
        os.makedirs(args.out, mode=0o755)
        if log:
            print(f"Created output directory: {args.out}")
    return gfile,args

t = time.time()
print(green_output('High Performance Linear Mixed Model Solver for Genome-Wide Association Studies'))
print(green_output(f'Host: {socket.gethostname()}'),end='\n\n')
gfile,args = main()
phenofile,outfolder = args.pheno,args.out
kinship_method = args.grm
qdim = args.qcov
HighAC = args.AC
threads = args.thread
kcal = True if kinship_method in ['VanRanden', 'gemma1', 'gemma2', 'pearson'] else False
qcal = True if qdim in np.arange(20).astype(str) else False
if not os.path.exists(outfolder):
    os.makedirs(outfolder,mode=0o755)
prefix = gfile.replace('.vcf','').replace('.gz','')

if args.vcf:
    print(f'Loading genotype from {gfile}...')
    geno = vcfreader(rf'{gfile}').iloc[:,2:].T 
else:
    print(f'Loading genotype from {gfile}.bed...')
    geno = breader(rf'{gfile}').iloc[:,2:].T # PLINK格式
geno.index = geno.index.astype(str)
snp_chrloc = geno.columns
famid = geno.index
print(f'Loading phenotype from {phenofile}...')
pheno = pd.read_csv(rf'{phenofile}',sep='\t') # 第一列是样本ID, 第一行是表型名
pheno = pheno.groupby(pheno.columns[0]).mean() # 重复样本表型取均值
pheno.index = pheno.index.astype(str)
print('Geno and Pheno are ready!')

if qcal or kcal:
    if not os.path.exists(f'{prefix}.k.{kinship_method}.txt') or not os.path.exists(f'{prefix}.q.{qdim}.txt'):
        qkmodel = QK(geno.values,low_memory=True,log=True)
        print('Samples and SNP:',geno.shape)
    if os.path.exists(f'{prefix}.k.{kinship_method}.txt'):
        print(f'* Loading GRM from {prefix}.k.{kinship_method}.txt...')
        kmatrix = pd.read_csv(f'{prefix}.k.{kinship_method}.txt',sep=r'\s+',header=None).values
    else:    
        print(f'* Calculation method of kinship matrix is {kinship_method}')
        kmatrix = qkmodel.kinship(method=kinship_method)
        np.savetxt(f'{prefix}.k.{kinship_method}.txt',kmatrix,fmt='%.6f')

    if os.path.exists(f'{prefix}.q.{qdim}.txt'):
        print(f'* Loading Q matrix from {prefix}.q.{qdim}.txt...')
        qmatrix = pd.read_csv(f'{prefix}.q.{qdim}.txt',sep=r'\s+',header=None).values
    else:    
        print(f'* Dimension of PC for q matrix is {qdim}')
        qmatrix,eigenval = qkmodel.rpca(dim=int(qdim))
        np.savetxt(f'{prefix}.q.{qdim}.txt',qmatrix,fmt='%.6f')

else:
    if not qcal and os.path.exists(qdim):
        print(f'* Loading Q matrix from {qdim}...')
        qmatrix = np.genfromtxt(qdim)
    else:
        print(f'{qdim} is not a number and a file')
    if not kcal and os.path.exists(kinship_method):
        print(f'* Loading GRM from {kinship_method}...')
        kmatrix = np.genfromtxt(kinship_method)
    else:
        print(f'{qdim} is not a calculation method of kinship and a file')
print(f'kmatrix {kmatrix.shape}:')
print(kmatrix[:5,:5])
print(f'qmatrix {qmatrix.shape}:')
print(qmatrix[:5,:5])

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
            results = gwasmodel.gwasHAC(snp=geno.values[famid_geno,:],chunksize=200_000,threads=threads) # gwas running...
            # np.savetxt(f'{outfolder}/{i}.lbd',np.array(gwasmodel.lbd),fmt='%.4f')
        else:
            results = gwasmodel.gwas(snp=geno.values[famid_geno,:],chunksize=200_000,threads=threads) # gwas running...
        results = pd.DataFrame(results,columns=['beta','se','p'],index=snp_chrloc[gwasmodel.snp_retain]).reset_index()
        results_save = format_dataframe_for_export(results, scientific_cols=['p'], float_cols=['beta','se'])
        results_save.to_csv(f'{outfolder}/{i}.tsv',sep='\t',index=False)
        print(f'Saved in {outfolder}/{i}.tsv'.replace('//','/'))
        
        manhan = GWASPLOT(results,'#CHROM','POS','p')
        plt.figure(figsize=(12,4),dpi=300)
        ax1 = plt.subplot(1,2,1)
        manhan.manhattan(-np.log10(0.05/results.shape[0]),ax=ax1)
        ax2 = plt.subplot(1,2,2)
        manhan.qq(ax=ax2)
        plt.tight_layout()
        print('Visualizing...')
        plt.savefig(f'{outfolder}/{i}.png')
        print(f'Saved in {outfolder}/{i}.png\n'.replace('//','/'))
        del results,results_save,manhan,gwasmodel,p,famid_pheno,famid_geno
    else:
        print(f'Phenotype {i} has no overlapping samples with genotype, please check sample id. skipped.\n')
lt = time.localtime()
print(green_output(f'\nFinished, Total time: {round(time.time()-t,2)} secs\n{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'))

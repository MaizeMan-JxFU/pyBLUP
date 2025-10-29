# pyBLUP：高效的全基因组关联分析及全基因组选择工具

## 引言

&emsp;&emsp;近期需要分析海量基因型和表型数据，由于不满足于现有全基因组关联分析（GWAS）软件（GEMMA、GCTA、rMVP、TASSEL）的速度，因此尝试复现GWAS的混合线性模型算法优化计算速度。现有软件的问题主要在于跨平台困难（例如 GEMMA没有windows版本）、多核调用不积极（俗称1核有难，8核围观）、计算Q矩阵依赖其他软件等，因此我在算法复现中逐一解决上述问题。

## 算法解析

&emsp;&emsp;GWAS中的混合线性模型如公式(1)所示，通常简写成公式(2)的形式。其中$X$是固定因子矩阵，包含1列全为1的固定截距向量以及多列固定因子向量（例如 群体结构、基因型），$\beta$是固定因子的效应值；$g$是个体随机因子，$g \sim N(0,\sigma_{g}^2 G)$，$G$是亲缘关系矩阵，可通过系谱关系或基因型计算获得；$\epsilon$是残差，$\epsilon \sim N(0,\sigma_{\epsilon}^2 I)$；y是表型向量，$y \sim N(X\beta,\sigma_{g}^2 G+\sigma_{\epsilon}^2 I)$

```math
y=\mu+X_{cov}\beta_{cov}+X_{snp}\beta_{snp}+g+\epsilon \tag{1}
```

```math
y=X\beta+g+\epsilon \tag{2}
```

&emsp;&emsp;其中需要估计的参数包括$\beta、\sigma_{g}^2、\sigma_{\epsilon}^2$。

&emsp;&emsp;首先，我们设：$\sigma_{\epsilon}^2=\lambda \sigma_{g}^2$, $V=G+\lambda I $, 则 $y\sim N(X\beta,\sigma_{g}^2 V)$

&emsp;&emsp;基于广义最小二乘法，我们可以用$V$估计$\beta$,公式如下：

$$\hat{\beta}=(X'V^{-1}X)^{-1}X'V^{-1}y \tag{5}$$

>**广义最小二乘法的推导**
*定理1* 任意正定矩阵A都存在$A=LL'$（Cholesky分解）
当 $y=X\beta+\epsilon$ ，$\epsilon \sim N(0,\sigma^{2} I)$，其中$I$是单位矩阵。则$\beta$的最小二乘估计公式 $\hat{\beta}=(X'X)^{-1}X'y$
我们可以把混合线性模型同样视为 $y=X\beta+\epsilon$ 的形式，但此时 $\epsilon \sim N(0,\sigma^{2} \Sigma)$，其中$\Sigma$是随机因子的协方差矩阵，那么我们只需要对方程进行线性变换将$\Sigma$转变为单位矩阵则可以套用最小二乘估计的公式。根据定理1，可以将正定矩阵$\Sigma$分解成可逆的上下三角矩阵$L$和$L'$，随后对混合线性模型公式同乘$L^{-1}$进行线性变换，即可将$\Sigma$转换为单位矩阵。推导如下：
$$
\Sigma=LL' \\
L^{-1}y=L^{-1}X\beta+L^{-1}\epsilon, L^{-1}\epsilon \sim N(0,\sigma^2 I) \\
\hat{\beta}=((L^{-1}X)'(L^{-1}X))^{-1}(L^{-1}X)'L^{-1}y \\
\hat{\beta}=(X'(L^{-1})'L^{-1}X)^{-1}X'(L^{-1})'L^{-1}y \\
\because (L^{-1})'L^{-1}=\Sigma^{-1},\therefore \hat{\beta}=(X'\Sigma^{-1}X)^{-1}X'\Sigma^{-1}y
$$

&emsp;&emsp;此时，需要估计的参数包括$\sigma_{g}^2、\lambda$，我们采用最大似然法对其进行估计，或者说我们将表型值向量$y$的多元正态分布的似然函数作为损失函数估计这两个未知参数。多元正态分布的限制性似然函数公式（推导较为复杂，直接上公式）如下：

设$$r=(y-X\beta)'(y-X\beta)$$

则$$x$$

$$\sigma_{g}^2=\frac{r'V^{-1}r}{n-p} \tag{6}$$

$$C_{reml}=\frac{n-p}{2}(ln(n-p)-ln(2\pi)-1) \tag{8}$$

$$logL_{reml}=C_{reml}-(n-p)ln(r'V^{-1}r)+ln|V|+ln|X'V^{-1}X| \tag{7}$$

## 使用方法

### 安装

首先需要环境中需要包含 [python](https://www.python.org/downloads/release/python-3139/) (3.9~3.13)

如果有git基础，以下几行代码即可完成安装啦~

```bash
# 网络顺畅的情况
git clone https://github.com/MaizeMan-JxFU/pyBLUP.git
# 不能科学上网可以选择国内代理
git clone https://gh-proxy.com/https://github.com/MaizeMan-JxFU/pyBLUP.git
# 进入目标文件夹
cd pyBLUP
# 执行pip安装依赖
pip install -r gwas.requirements.txt
```

没有git基础，作者也提供了[pyBLUP包文件](https://pan.baidu.com/s/1EibqB_xkuJSlnDM2LhArBA?pwd=TEMP)，解压后在终端进入pyBLUP文件夹，执行下列代码安装依赖

```bash
# 执行pip安装依赖
pip install -r gwas.requirements.txt
```

### 功能1: 全基因组关联分析

必须参数1：--vcf [vcf文件] 或 --bfile [plink文件]
必须参数2：--pheno [表型文件]
必须参数3：--out [结果文件输出文件夹（不存在则自动创建）]
默认参数：计算 VanRanden 亲缘关系矩阵、基于基因型的前3个主成分，生成文件于vcf或bfile文件目录

#### 多平台使用

```bash
python gwas.py --vcf example/mouse_hs1940.vcf.gz --pheno example/mouse_hs1940.pheno --out test
```

#### unix使用

GWAS [模块名] [模块命令]（后续增加 coloc、gs 等模块）

```bash
chmod +755 GWAS # 可将 GWAS 所在文件夹加入环境变量
GWAS gwas -h # 查看帮助
GWAS gwas --vcf example/mouse_hs1940.vcf.gz --pheno example/mouse_hs1940.pheno --out test # 用法和 python gwas.py [参数] 一致
```

使用[测试数据](https://doi.org/10.1038/ng.3609)来源于输出结果如下所示：
![GWAStest](./fig/test0.png "GWAS test of pyBLUP")
*(The above image depicts physiological and behavioral trait loci identified in CFW mice using GEMMA, from Parker et al, Nature Genetics, 2016.)

### 功能2: GBLUP & rrBLUP

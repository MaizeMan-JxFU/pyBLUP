library(rrBLUP)
#random population of 200 lines with 1000 markers
snp_num = 10000
sample_num = 10000
M <- matrix(rep(0,sample_num*snp_num),sample_num,snp_num)
for (i in 1:sample_num) {
    M[i,] <- ifelse(runif(snp_num)<0.5,-1,1)
    }
#random phenotypes
u <- rnorm(snp_num)
g <- as.vector(crossprod(t(M),u))
h2 <- 0.5 #heritability
y <- g + rnorm(sample_num,mean=0,sd=sqrt((1-h2)/h2*var(g)))
#predict marker effects
t = Sys.time()
ans <- mixed.solve(y,Z=M) #By default K = I
print(Sys.time()-t)
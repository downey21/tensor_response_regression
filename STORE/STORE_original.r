## Main function for Sparse Tensor Outcome REgression (STORE) by Sun and Li (2017, JMLR)
## Maintenance: Will Wei Sun
## Last edition: Sep 1, 2017


library(igraph)
# in case there is an error in "library(remMap)": install.packages("~/Research/STOR/remMap_0.2-0.tar.gz")
library(remMap)
# in case there is an error in "library(caret)": install.packages("~/pbkrtest_0.4-5.tar.gz")
library(caret)
library(rTensor)

#####################
## Utility Functions
#####################
EPS = 1e-4

rotate <- function(x) t(apply(x, 2, rev))


fnorm <- function(T){
	### Tensor Frobenius Norm \| T \|_F ###
	return(sqrt(sum(T^2)))
}


myadj = function(A){
	#calculate the adjacent matrix of a matrix A
	connect.mat = A
	p = ncol(A)
	for(i in 1:p){
		for(j in 1: p){
			connect.mat[i,j] = ifelse(abs(A[i,j])>1e-5, 1, 0)
		}
	}
	connect.mat
}


mysd = function(x){
	if(norm(x,"2") == 0){
		return(x)
	}else{
		return(x/sqrt(t(x)%*%x))
	}	
}

mynorm = function(x){ 
	(x-mean(x))/sd(x)
}

mycenter = function(x){
	x - mean(x)
}

mytensorcenter = function(y){
	# y is a list of tensors
	n = length(y)
	y.center = Reduce("+", y) / length(y)	
	y.new = y
	for(i in 1:n){
		y.new[[i]] = y[[i]] - y.center
	}
	y.new
}

mytruncate = function(x,s){
	xtruncate = rep(0, length(x))
	xtruncate[ which(rank(-abs(x))<= s)] = x[which(rank(-abs(x))<= s)]
	xtruncate
}

myerrorlist = function(beta.old, beta.new){
	nlist = length(beta.old)
	error = 0
	for(i in 1:nlist){
		error = error + min(fnorm(beta.old[[i]] - beta.new[[i]]), fnorm(beta.old[[i]] + beta.new[[i]]))
	}
	error/nlist
}

Ta = function(T, a){
	# For a tensor T \in R^{d_1, d_2, d_3, p} and a vector a \in R^p, compute mode-4 tensor vector product T \times_{4} a.
	
	DD = dim(T)
	a = as.numeric(a)
	p = length(a)
	if(DD[length(DD)] != p){
		error("Dimensions of T and a do not fit")
	}else{
		
		tmp = array(0,DD[1:(length(DD) - 1)])
		for(i in 1:p){
			if(length(DD) == 3){
				tmp = tmp + T[,,i] * a[i]
			}else if(length(DD) == 4){
				tmp = tmp + T[,,,i] * a[i]
			}
		}
	}
	tmp
}


TA = function(T, A){
	# For a tensor T \in R^{d_1, d_2, d_3, p} and a matrix A \in R^{n * p} , compute mode-4 tensor vector product T \times_{4} A_i for each i = 1, ...,n
	n = nrow(A)
	out = list()
	for(i in 1:n){
		out[[i]] = Ta(T,A[i,])
	}
	out
}


Tabc = function(T, a, b, c){
	#compute tensor vector product T(a,b,c) for tensor T and vectors, a,b,c with different dimensions.
	
	d1 = dim(T)[1]	
	d2 = dim(T)[2]
	d3 = dim(T)[3]
	tmp = 0
	for(i in 1:d1){
		for(j in 1:d2){	
			for(k in 1:d3){		
				tmp = tmp + a[i]*b[j]*c[k]*T[i,j,k]
			}
		}
	}
	tmp
}


lasso_operator <- function(x, lambda){
	### Soft-thresholding operator ###
	k = length(x)
	temp = rep(0, k)
	for(i in 1:k){
	temp[i] = sign(x[i]) * (max(abs(x[i]) - lambda, 0))
	}
	return(temp)
}


listnorm <- function(X,Y){
	### List Frobenius Norm n^{-1}\sum_i^n \| X[[i]] - Y[[i]] \|_F ###

	n = length(Y)
	if(length(X) !=n){
		error("sizes of X and Y do not match!")
	}else{
		value = rep(0, n)
		for(i in 1:n){
			value[i] = fnorm(X[[i]] - Y[[i]])
		}
	}
	mean(value)
}


sumnorm <- function(T, A){
	### Calculate sum(T * A) with * the Hadamard product (element-wise product) ###
	
	DD = dim(T)
	m = length(DD)
	temp = 0	
	if(m == 2){
		for(i in 1 : DD[1]){
			for(j in 1 : DD[2]){
				temp = temp + T[i, j] * A[i, j]
			}
		}
	}else if(m == 3){
		for(i in 1 : DD[1]){
			for(j in 1 : DD[2]){
			  for(k in 1 : DD[3]){
				temp = temp + T[i, j, k] * A[i, j, k]
			  }
			}
		}
	}
	return(temp)
}


dist_sign = function(a.est, a){
	#compute distance min{ \|a.est - a\|, \|a.est + a\| }
	
	a.est = as.matrix(a.est)
	a = as.matrix(a)	
	if(ncol(a.est) == 1){
		out = min(norm(a - a.est,type="2"), norm(a + a.est,type="2"))
	}else{
		out = min(norm(a - a.est,type="F"), norm(a + a.est,type="F"))
	}
	out
}


mytprfpr = function(Beta.hat, Beta.true, is.symmetric = FALSE){
	#compute the variable selection error of our method.
	# Beta.hat, Beta.true: list of {\Beta_1, ..., \Beta_m, \Beta_{m+1}} with \Beta_j \in R^{K \times d_j}

	K = dim(Beta.true[[1]])[1]
	m = length(Beta.true) - 1

	if(m == 2 && is.symmetric == TRUE){
	# symmetric matrix case in simulation 3

		A.est = t(Beta.hat[[1]])
		A = t(Beta.true[[1]])
		ERROR = matrix(NA,K,K)
		j.best = rep(0,K)
		for(i in 1:K){
			for(j in 1:K){	
				ERROR[i,j] = min(norm(A.est[,i] - A[,j],type="2"),norm(A.est[,i] + A[,j],type="2"))
			}
			j.best[i] = which.min(ERROR[i,])	
		}

		TPR.vec = rep(0,K)
	   	FPR.vec = rep(0,K)
	    IND_hat = Beta.hat[[1]]
	    IND_true = Beta.true[[1]]
	    IND_hat[which(Beta.hat[[1]]!=0)] = 1
	    IND_true[which(Beta.true[[1]]!=0)] = 1
	    for(i in 1:K){
	        if(length(which(Beta.hat[[1]] == 0)) == 0){
	            TPR.vec[i] = 1
	            FPR.vec[i] = 1
	        }else{
	            TPR.vec[i] = confusionMatrix(IND_hat[i,],IND_true[j.best[i],],positive="1")$byClass[1]
	            FPR.vec[i] = 1 - confusionMatrix(IND_hat[i,],IND_true[j.best[i],],positive="1")$byClass[2]
	        }
	    }
	
	}else{

		A.est = t(Beta.hat[[1]])
		B.est = t(Beta.hat[[2]])
		C.est = t(Beta.hat[[3]])
		A = t(Beta.true[[1]])
		B = t(Beta.true[[2]])
		C = t(Beta.true[[3]])

		## find the corresponding index in Beta.hat and Beta.true
		ERROR = matrix(NA,K,K)
		j.best = rep(0,K)
		for(i in 1:K){
			for(j in 1:K){	
				ERROR[i,j] = min(norm(A.est[,i] - A[,j],type="2"),norm(A.est[,i] + A[,j],type="2"))
			}
			j.best[i] = which.min(ERROR[i,])	
		}

		## variable selection error
		IA = A; IB = B; IC = C
		IA[which(A!=0)] = 1
		IB[which(B!=0)] = 1
		IC[which(C!=0)] = 1
		
		IA.est = A.est; IB.est = B.est; IC.est = C.est
		IA.est[which(IA.est != 0)] = 1
		IB.est[which(IB.est != 0)] = 1
		IC.est[which(IC.est != 0)] = 1
		
		TPR.vec = rep(0,K)
		FPR.vec = rep(0,K)
		for(ii in 1:K){

			if(length(unique(IA.est[,ii]))==1 && IA.est[1,ii]==1){
				tpr.a = 1
				fpr.a = 1
			}else if(length(unique(IA.est[,ii]))==1 && IA.est[1,ii]==0){
				tpr.a = 0
				fpr.a = 0
			}else{
				tpr.a = confusionMatrix(IA.est[,ii],IA[,j.best[ii]],positive="1")$byClass[1]
				fpr.a = 1 - confusionMatrix(IA.est[,ii],IA[,j.best[ii]],positive="1")$byClass[2]
			}
			
			if(length(unique(IB.est[,ii]))==1 && IB.est[1,ii]==1){
				tpr.b = 1
				fpr.b = 1
			}else if(length(unique(IB.est[,ii]))==1 && IB.est[1,ii]==0){
				tpr.b = 0
				fpr.b = 0
			}else{
				tpr.b = confusionMatrix(IB.est[,ii],IB[,j.best[ii]],positive="1")$byClass[1]
				fpr.b = 1 - confusionMatrix(IB.est[,ii],IB[,j.best[ii]],positive="1")$byClass[2]
			}
			
			if(length(unique(IC.est[,ii]))==1 && IC.est[1,ii]==1){
				tpr.c = 1
				fpr.c = 1
			}else if(length(unique(IC.est[,ii]))==1 && IC.est[1,ii]==0){
				tpr.c = 0
				fpr.c = 0
			}else{
				tpr.c = confusionMatrix(IC.est[,ii],IC[,j.best[ii]],positive="1")$byClass[1]
				fpr.c = 1 - confusionMatrix(IC.est[,ii],IC[,j.best[ii]],positive="1")$byClass[2]		
			}			
			TPR.vec[ii] = mean(c(tpr.a,tpr.b,tpr.c))
			FPR.vec[ii] = mean(c(fpr.a,fpr.b,fpr.c))	
		}
	}

	c(mean(TPR.vec), mean(FPR.vec))
}


mytprfpr_B = function(B.hat, B.true){
	#compute the variable selection error of the whole tensor coefficient B. output (TPR, FPR, f1)
	# B.hat, B.true: \in R^{d_1 *...* d_{m+1}}


	m = length(dim(B.true)) - 1
	p = dim(B.true)[m+1]

	alldim = prod(dim(B.true))
	IND_hat = rep(0, alldim)
	IND_true = rep(0, alldim)
	IND_hat[which(B.hat!=0)] = 1
	IND_true[which(B.true!=0)] = 1

	if(length(which(B.hat == 0)) == 0){
		## all non-zeros
	    TPR.vec = 1
	    FPR.vec = 1
	    precision.val = length(which(B.true!=0)) / alldim
	    recall.val = 1
	    f1 = as.numeric(2 * precision.val * recall.val / (precision.val + recall.val) )

	}else if(length(which(B.hat != 0)) == 0){
		## all zeros
		TPR.vec = 0
	    FPR.vec = 0
	    precision.val = length(which(B.true!=0)) / alldim
	    recall.val = 0
	    f1 = as.numeric(2 * precision.val * recall.val / (precision.val + recall.val) )

	}else{
	    TPR.vec = confusionMatrix(IND_hat,IND_true,positive="1")$byClass[1]
	    FPR.vec = 1 - confusionMatrix(IND_hat,IND_true,positive="1")$byClass[2]
		confusion.table = confusionMatrix(IND_hat,IND_true,positive="1")$table
		TN = confusion.table[1,1]
		TP = confusion.table[2,2]
		FN = confusion.table[1,2]
		FP = confusion.table[2,1]
		if(TP == 0){
			## avoid NaN case in f1 score. 
			TP = 0.1
		}
		precision.val = TP / (TP + FP)
		recall.val = TP/ (TP + FN)
		f1 = as.numeric(2 * precision.val * recall.val / (precision.val + recall.val) )

	}

	as.numeric(c(TPR.vec, FPR.vec, f1))
}


### lasso type rank-1 sparse tensor decomposition (Allen, 2012) ###
stdlasso <- function(T, Para, niter = 20) {
	# T: a matrix or a three-order tensor
	# Para = c(lambda_1,..., lambda_m): number of non-zero slements in each columns of components
	# niter: number of iterations 
	
	DD = dim(T)
	num_m = length(DD)
	set.seed(1)
	beta.new = list()
	for(j in 1:num_m){
		beta.new[[j]] = rnorm(DD[j])
	}
	t = 0
	while(t < niter){
		t = t + 1
		beta.old = beta.new
		beta.tmp = list()
		for(j in 1:num_m){
			beta.tmp[[j]] = rep(0, DD[j])
		}
		if(num_m == 2){
			## beta_1 ##
			for (j in 1 : DD[2]) {
				beta.tmp[[1]] = beta.tmp[[1]] + beta.old[[2]][j] * T[, j]
			}
			beta.tmp[[1]] = beta.tmp[[1]] / norm(beta.tmp[[1]], type = "2")
			if (norm(lasso_operator(beta.tmp[[1]], Para[1]), type = "2") > 0) {
				beta.new[[1]] <- lasso_operator(beta.tmp[[1]], Para[1]) / norm(lasso_operator(beta.tmp[[1]], Para[1]), type = "2")
			}else{
				beta.new[[1]] = rep(0, DD[1])
				break
			}
			## beta_2 ##
			for (i in 1 : DD[1]) {
				beta.tmp[[2]] = beta.tmp[[2]] + beta.old[[1]][i] * T[i, ]
			}
			beta.tmp[[2]] = beta.tmp[[2]] / norm(beta.tmp[[2]], type = "2")
			if (norm(lasso_operator(beta.tmp[[2]], Para[2]), type = "2") > 0) {
				beta.new[[2]] <- lasso_operator(beta.tmp[[2]], Para[2]) / norm(lasso_operator(beta.tmp[[2]], Para[2]), type = "2")
			}else{
				beta.new[[2]] = rep(0, DD[2])
				break
			}
		}else if(num_m == 3){
			## beta_1 ##
			for (j in 1 : DD[2]) {
				for (k in 1 : DD[3]) {
				  beta.tmp[[1]] = beta.tmp[[1]] + beta.old[[2]][j] * beta.old[[3]][k] * T[, j, k]
				}
			}
			beta.tmp[[1]] = beta.tmp[[1]] / norm(beta.tmp[[1]], type = "2")
			if (norm(lasso_operator(beta.tmp[[1]], Para[1]), type = "2") > 0) {
				beta.new[[1]] <- lasso_operator(beta.tmp[[1]], Para[1]) / norm(lasso_operator(beta.tmp[[1]], Para[1]), type = "2")
			}else{
				beta.new[[1]] = rep(0, DD[1])
				break
			}
			## beta_2 ##
			for (i in 1 : DD[1]) {
				for (k in 1 : DD[3]) {
				  beta.tmp[[2]] = beta.tmp[[2]] + beta.old[[1]][i] * beta.old[[3]][k] * T[i, , k]
				}
			}
			beta.tmp[[2]] = beta.tmp[[2]] / norm(beta.tmp[[2]], type = "2")
			if (norm(lasso_operator(beta.tmp[[2]], Para[2]), type = "2") > 0) {
				beta.new[[2]] <- lasso_operator(beta.tmp[[2]], Para[2]) / norm(lasso_operator(beta.tmp[[2]], Para[2]), type = "2")
			}else{
				beta.new[[2]] = rep(0, DD[2])
				break
			}
			## beta_3 ##
			for (i in 1 : DD[1]) {
				for (j in 1 : DD[2]) {
				  beta.tmp[[3]] = beta.tmp[[3]] +  beta.old[[1]][i] * beta.old[[2]][j] * T[i, j,]
				}
			}
			beta.tmp[[3]] = beta.tmp[[3]] / norm(beta.tmp[[3]], type = "2")
			if (norm(lasso_operator(beta.tmp[[3]], Para[3]), type = "2") > 0) {
				beta.new[[3]] <- lasso_operator(beta.tmp[[3]], Para[3]) / norm(lasso_operator(beta.tmp[[3]], Para[3]), type = "2")
			}else{
				beta.new[[3]] = rep(0, DD[3])
				break
			}	
		}
	}	
	for(j in 1:num_m){
		beta.new[[j]] = as.matrix(beta.new[[j]])
	}
	w = ifelse(num_m ==2, as.numeric(t(beta.new[[1]]) %*% T %*% beta.new[[2]]), Tabc(T, beta.new[[1]], beta.new[[2]], beta.new[[3]]))
	OUT = list()
	OUT$beta = beta.new
	OUT$w = w
	OUT$t = t
	return(OUT)
}


### truncation type rank-1 sparse tensor decomposition (Sun et al., 2016) ###
stdtruncate <- function(T, Para, is.symmetric = FALSE, real_3d = FALSE, niter = 20) {
	# T: a three-order tensor
	# Para = c(s1,...,sm): number of non-zero slements in each columns of components
	# niter: number of iterations 
	DD = dim(T)
	num_m = length(DD)

	if(is.symmetric == TRUE){
		## special case with beta_1 = ... = beta_m
		if(length(unique(DD))>1){
			error("Input tensor is not symmetric. Can not apply symmetric tensor decomposition!")
			break
		}else{
			dd = DD[1]
			if(num_m == 2){
				set.seed(2)
				a.tmp = svd(T)$u[,1]
				a.new = mysd(mytruncate(mysd(a.tmp),Para[1]))
			}else{
				set.seed(2)
				a.new = rnorm(dd)
				a.old = rep(0,dd)
				TER.ERROR = NULL
				t = 0
				while(norm(a.old - a.new,type="2") >= EPS && t < niter)
				{
					TER.ERROR = append(TER.ERROR, norm(a.old - a.new,type="2"))
					t = t + 1
					a.old = a.new				
					a.tmp = rep(0, dd)
					for (j in 1 : dd) {
						for (k in 1 : dd) {
							a.tmp = a.tmp + a.old[j] * a.old[k] * T[, j, k]
						}
					}
					a.new = mysd(mytruncate(mysd(a.tmp),Para[1]))
				}
			}
			beta.new = list()
			for(j in 1:num_m){
				beta.new[[j]] = as.matrix(a.new)
			}
		}
	}else{

		set.seed(1)
		beta.new = list()
		beta.old = list()
		for(j in 1:num_m){
			beta.new[[j]] = rnorm(DD[j])
			beta.old[[j]] = rep(0, DD[j])
		}
		t = 0
		w.tmp = NULL
		TER.ERROR = NULL
		while(myerrorlist(beta.old, beta.new) >= EPS && t < niter)
		{
			TER.ERROR = append(TER.ERROR, myerrorlist(beta.old, beta.new))
			t = t + 1
			beta.old = beta.new
			beta.tmp = list()
			for(j in 1:num_m){
				beta.tmp[[j]] = rep(0, DD[j])
			}
			if(num_m == 2){
				## beta_1 ##
				for (j in 1 : DD[2]) {
					beta.tmp[[1]] = beta.tmp[[1]] + beta.old[[2]][j] * T[, j]
				}
				beta.new[[1]] = mysd(mytruncate(mysd(beta.tmp[[1]]),Para[1]))
				## beta_2 ##
				for (i in 1 : DD[1]) {
					beta.tmp[[2]] = beta.tmp[[2]] + beta.old[[1]][i] * T[i, ]
				}
				beta.new[[2]] = mysd(mytruncate(mysd(beta.tmp[[2]]),Para[2]))
			}else if(num_m == 3){
				## beta_1 ##
				for (j in 1 : DD[2]) {
					for (k in 1 : DD[3]) {
					  beta.tmp[[1]] = beta.tmp[[1]] + beta.old[[2]][j] * beta.old[[3]][k] * T[, j, k]
					}
				}
				beta.new[[1]] = mysd(mytruncate(mysd(beta.tmp[[1]]),Para[1]))
				## beta_2 ##
				for (i in 1 : DD[1]) {
					for (k in 1 : DD[3]) {
						if(real_3d == TRUE){
							beta.tmp[[2]] = beta.tmp[[2]] + beta.new[[1]][i] * beta.old[[3]][k] * T[i, , k]
						}else{
							beta.tmp[[2]] = beta.tmp[[2]] + beta.old[[1]][i] * beta.old[[3]][k] * T[i, , k]
						}
					  
					}
				}
				beta.new[[2]] = mysd(mytruncate(mysd(beta.tmp[[2]]),Para[2]))
				## beta_3 ##
				for (i in 1 : DD[1]) {
					for (j in 1 : DD[2]) {
						if(real_3d == TRUE){
							beta.tmp[[3]] = beta.tmp[[3]] +  beta.new[[1]][i] * beta.new[[2]][j] * T[i, j,]
						}else{
							beta.tmp[[3]] = beta.tmp[[3]] +  beta.old[[1]][i] * beta.old[[2]][j] * T[i, j,]							
						}
					  
					}
				}
				beta.new[[3]] = mysd(mytruncate(mysd(beta.tmp[[3]]),Para[3]))	
			}
			#debug
			w.tmp = append(w.tmp, ifelse(num_m ==2, as.numeric(t(as.matrix(beta.new[[1]])) %*% T %*% as.matrix(beta.new[[2]])), Tabc(T, as.matrix(beta.new[[1]]), as.matrix(beta.new[[2]]), as.matrix(beta.new[[3]]))))
		}	
		for(j in 1:num_m){
			beta.new[[j]] = as.matrix(beta.new[[j]])
		}
	}

	w = ifelse(num_m ==2, as.numeric(t(beta.new[[1]]) %*% T %*% beta.new[[2]]), Tabc(T, beta.new[[1]], beta.new[[2]], beta.new[[3]]))
	OUT = list()
	OUT$beta = beta.new
	OUT$w = w
	return(OUT)
}


myobj = function(Y, X, W, B){
	# objective function to be minimized
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * d2 * d3
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	# W: a K-dim vector of weights W = (w_1, ..., w_K)
	# B: a list of parameters B = {B_1, ..., B_m, B_{m+1}} wit B[[j]] = B_j \in R^{K * d_j} for j =1,...,m+1. here d_{m+1} = p.
	
	n = nrow(X)
	K = length(W)
	m = length(B) - 1	
	A = list()
	for(k in 1:K){
		A[[k]] = B[[1]][k,]
		for(j in 2:m){
			A[[k]] = outer(A[[k]], B[[j]][k,])
		}	
		A[[k]] = W[k] * A[[k]]		
	}		
	obj = 0
	for(i in 1:n){
		Diff = array(0, dim(Y[[1]]))	
		for(k in 1:K){
			Diff = Diff + A[[k]] * as.numeric(t(B[[m+1]][k,]) %*% X[i,])
		}
		obj = obj + fnorm( Y[[i]] - Diff)^2
	}
	obj = obj/n
	obj
}



#####################
## Main Algorithms
#####################
mystor <- function(Y, X, Para, K, stdmethod = "truncate", is.symmetric = FALSE, niter = 20, allow.restart = FALSE, real_3d = FALSE, is.center = FALSE) {
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * d2 * d3
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	# Para = (lambda_1, ..., lambda_m) or (s_1, ..., s_m): tuning parameters for components A, B, C, ...
	# K: Rank
	# niter: number of iterations 
	
	m = length(dim(Y[[1]]))
	X = as.matrix(X)
	n = nrow(X)
	p = ncol(X)
	DD = c(lapply(Y,dim)[[1]], p) 	# DD = (d_1, ..., d_m, p)

	if(is.center == TRUE){
		X = apply(X,2,mycenter)
		Y = mytensorcenter(Y)
	}

	if(allow.restart == FALSE){
		i.restart = 0
		Beta = list()	
		for(j in 1: length(DD)){
			Beta[[j]] = matrix(0, K, DD[j])
			for(k in 1:K){
				set.seed(j+10*k+(i.restart-1)*100)
				Beta[[j]][k,] = mysd(rnorm(DD[j]))
			}
		}
		W = rep(1, K)	

		## Step 2: Alternative update	
		t = 0
		error = 1	
		TRUE.ERROR = NULL
		TERMINATE.ERROR = NULL
		OBJ = NULL
		W.mat = W
		
		while(error > EPS && t <= niter){	
			W.old = W	
			Beta.old = Beta
			t = t + 1

			## step 2_1: update beta_k^{(1)}, ..., beta_k^{(m+1)}; w_k for k =1,..., K.
			tmp = list()
			for(k in 1:K){
				tmp[[k]] = Beta[[1]][k,]
				for(j in 2:m){
					tmp[[k]] = outer(tmp[[k]], Beta[[j]][k,])
				}
				tmp[[k]] = W[k] * tmp[[k]]	
			}
					
			alpha = matrix(0, n, K)
			for(k in 1:K){
				for(i in 1:n){
					alpha[i,k] = as.numeric(t(Beta[[m+1]][k,]) %*% X[i,])	
				}
			}

			for(k in 1:K){	
				R = list()			
				for(i in 1:n){	
					Diff = array(0, dim(Y[[1]]))	
					for(k_prime in 1:K){
						if(k_prime != k & alpha[i,k_prime] != 0){
							Diff = Diff + tmp[[k_prime]] * alpha[i,k_prime]
						}
					}
					if(alpha[i,k]!= 0){
						R[[i]] = (Y[[i]] - Diff) / alpha[i,k]
					}else{
						R[[i]] = array(0, dim(Y[[i]]))
					}
				}
				
				R_bar = array(0, dim(Y[[1]]))
				index = which(alpha[,k] !=0)
				for(j in 1:length(index)){
					R_bar = R_bar + alpha[index[j], k]^2 * R[[index[j]]]
				}
				R_bar = R_bar/length(index)
		
				if(stdmethod == "lasso"){
					# if use lasso penalized sparse tensor decomposition
					OUT_std = stdlasso(R_bar, Para)
				}else{
					# if use truncated sparse tensor decomposition
					OUT_std = stdtruncate(R_bar, Para, is.symmetric,real_3d)
				}
				if(is.center == TRUE){
					W[k] = OUT_std$w / mean(alpha[,k]^2)
				}else{
					W[k] = OUT_std$w
				}
				for(j in 1:m){
					Beta[[j]][k,] = OUT_std$beta[[j]]
				}
				
				tmp[[k]] = Beta[[1]][k,]
				for(j in 2:m){
					tmp[[k]] = outer(tmp[[k]], Beta[[j]][k,])
				}
				tmp[[k]] = W[k] * tmp[[k]]			
			}
		
			
			## step 2_2: update beta_k^{(m+1)}, for k = 1, ..., K		
			A = list()
			for(k in 1:K){
				A[[k]] = Beta[[1]][k,]
				for(j in 2:m){
					A[[k]] = outer(A[[k]], Beta[[j]][k,])
				}	
				A[[k]] = W[k] * A[[k]]		
			}

			for(k in 1:K){				
				Omega = solve(n^{-1} * t(X) %*% X) 
				T = list()
				numerator = rep(0, p)
				for(i in 1:n){
					Diff = array(0, dim(Y[[1]]))
					for(k_prime in 1:K){
						if(k_prime != k){
							Diff = Diff + A[[k_prime]] * as.numeric(t(Beta[[m+1]][k_prime,]) %*% X[i,])
						}
					}
					T[[i]] = Y[[i]] - Diff 	
					numerator = numerator + sumnorm(T[[i]], A[[k]]) * X[i,]
				}	
				numerator = numerator / n
				Beta[[m+1]][k,] = Omega %*% numerator / (fnorm(A[[k]]))^2	
				Beta[[m+1]][k,] = Beta[[m+1]][k,] / norm(Beta[[m+1]][k,], type="2")		
			}		
			
			# step 2_3: update error \|B.hat - B.hat.old\|_F	
			B.hat = array(0, DD)
			for(k in 1:K){  
				if(m == 2){
					B.hat = B.hat + W[k] * outer(outer(Beta[[1]][k,],Beta[[2]][k,]),Beta[[3]][k,])
				}else if(m ==3){
					B.hat = B.hat + W[k] * outer(outer(outer(Beta[[1]][k,],Beta[[2]][k,]),Beta[[3]][k,]), Beta[[4]][k,])
				}
			}
			B.hat.old = array(0, DD)
			for(k in 1:K){  
				if(m == 2){
					B.hat.old = B.hat.old + W.old[k] * outer(outer(Beta.old[[1]][k,],Beta.old[[2]][k,]),Beta.old[[3]][k,])
				}else if(m ==3){
			  		B.hat.old = B.hat.old + W.old[k] * outer(outer(outer(Beta.old[[1]][k,],Beta.old[[2]][k,]),Beta.old[[3]][k,]), Beta.old[[4]][k,])
				}
			}
			error = fnorm(B.hat - B.hat.old)/fnorm(B.hat)
			TERMINATE.ERROR = append(TERMINATE.ERROR, error)	
			OBJ = append(OBJ, myobj(Y,X,W,Beta))
			W.mat = cbind(W.mat,W)

			if(myobj(Y,X,W,Beta) > 1e+50 || sum(abs(W)) > 1e+50){break} ## avoid the diverging case to caused error
		}	

	}else{

		is.restart = TRUE
		i.restart = 0
		TIME = NULL

		while(is.restart == TRUE && i.restart < 2){

			t1 = proc.time()
			i.restart = i.restart +1
			# Step 1: Initialization
			Beta = list()	
			for(j in 1: length(DD)){
				Beta[[j]] = matrix(0, K, DD[j])
				for(k in 1:K){
					set.seed(j+10*k+(i.restart-1)*100)
					Beta[[j]][k,] = mysd(rnorm(DD[j]))
				}
			}
			W = rep(1, K)	

			## Step 2: Alternative update	
			t = 0
			error = 1	
			TRUE.ERROR = NULL
			TERMINATE.ERROR = NULL
			OBJ = NULL
			W.mat = W
			
			while(error > EPS && t <= niter){	
				W.old = W	
				Beta.old = Beta
				t = t + 1

				## step 2_1: update beta_k^{(1)}, ..., beta_k^{(m+1)}; w_k for k =1,..., K.
				tmp = list()
				for(k in 1:K){
					tmp[[k]] = Beta[[1]][k,]
					for(j in 2:m){
						tmp[[k]] = outer(tmp[[k]], Beta[[j]][k,])
					}
					tmp[[k]] = W[k] * tmp[[k]]	
				}
						
				alpha = matrix(0, n, K)
				for(k in 1:K){
					for(i in 1:n){
						alpha[i,k] = as.numeric(t(Beta[[m+1]][k,]) %*% X[i,])	
					}
				}

				for(k in 1:K){	
					R = list()			
					for(i in 1:n){	
						Diff = array(0, dim(Y[[1]]))	
						for(k_prime in 1:K){
							if(k_prime != k & alpha[i,k_prime] != 0){
								Diff = Diff + tmp[[k_prime]] * alpha[i,k_prime]
							}
						}
						if(alpha[i,k]!= 0){
							R[[i]] = (Y[[i]] - Diff) / alpha[i,k]
						}else{
							R[[i]] = array(0, dim(Y[[i]]))
						}
					}
					
					R_bar = array(0, dim(Y[[1]]))
					index = which(alpha[,k] !=0)
					for(j in 1:length(index)){
						R_bar = R_bar + alpha[index[j], k]^2 * R[[index[j]]]
					}
					R_bar = R_bar/length(index)
							
					if(stdmethod == "lasso"){
						# if use lasso penalized sparse tensor decomposition
						OUT_std = stdlasso(R_bar, Para)
					}else{
						# if use truncated sparse tensor decomposition
						OUT_std = stdtruncate(R_bar, Para, is.symmetric,real_3d)
					}

					if(is.center == TRUE){
						W[k] = OUT_std$w / mean(alpha[,k]^2)
					}else{
						W[k] = OUT_std$w
					}

					for(j in 1:m){
						Beta[[j]][k,] = OUT_std$beta[[j]]
					}
					
					tmp[[k]] = Beta[[1]][k,]
					for(j in 2:m){
						tmp[[k]] = outer(tmp[[k]], Beta[[j]][k,])
					}
					tmp[[k]] = W[k] * tmp[[k]]			
				}

				## to avoid the case with zero weights, caused error in step 2_2.
				if(min(abs(W)) == 0){
					error = -1000
					is.restart = TRUE
					next
				}

				## step 2_2: update beta_k^{(m+1)}, for k = 1, ..., K		
				A = list()
				for(k in 1:K){
					A[[k]] = Beta[[1]][k,]
					for(j in 2:m){
						A[[k]] = outer(A[[k]], Beta[[j]][k,])
					}	
					A[[k]] = W[k] * A[[k]]		
				}

				for(k in 1:K){				
					Omega = solve(n^{-1} * t(X) %*% X) 
					T = list()
					numerator = rep(0, p)
					for(i in 1:n){
						Diff = array(0, dim(Y[[1]]))
						for(k_prime in 1:K){
							if(k_prime != k){
								Diff = Diff + A[[k_prime]] * as.numeric(t(Beta[[m+1]][k_prime,]) %*% X[i,])
							}
						}
						T[[i]] = Y[[i]] - Diff 	
						numerator = numerator + sumnorm(T[[i]], A[[k]]) * X[i,]
					}	
					numerator = numerator / n
					Beta[[m+1]][k,] = Omega %*% numerator / (fnorm(A[[k]]))^2	
					Beta[[m+1]][k,] = Beta[[m+1]][k,] / norm(Beta[[m+1]][k,], type="2")		
				}		

				# step 2_3: update error \|B.hat - B.hat.old\|_F	
				B.hat = array(0, DD)
				for(k in 1:K){  
					if(m == 2){
						B.hat = B.hat + W[k] * outer(outer(Beta[[1]][k,],Beta[[2]][k,]),Beta[[3]][k,])
					}else if(m ==3){
						B.hat = B.hat + W[k] * outer(outer(outer(Beta[[1]][k,],Beta[[2]][k,]),Beta[[3]][k,]), Beta[[4]][k,])
					}
				}
				B.hat.old = array(0, DD)
				for(k in 1:K){  
					if(m == 2){
						B.hat.old = B.hat.old + W.old[k] * outer(outer(Beta.old[[1]][k,],Beta.old[[2]][k,]),Beta.old[[3]][k,])
					}else if(m ==3){
				  		B.hat.old = B.hat.old + W.old[k] * outer(outer(outer(Beta.old[[1]][k,],Beta.old[[2]][k,]),Beta.old[[3]][k,]), Beta.old[[4]][k,])
					}
				}
				error = fnorm(B.hat - B.hat.old)/fnorm(B.hat)
				TERMINATE.ERROR = append(TERMINATE.ERROR, error)	
				OBJ = append(OBJ, myobj(Y,X,W,Beta))
				W.mat = cbind(W.mat,W)

			}	


			# Compute correlation among the components. if correlation is very high, has risky of local minimal, restart the initialization!
			cor.vec = NULL
			for(ii in 1:m){
				AA = Beta[[ii]]
				if(K>=2){
					for(k in 1:(K-1)){
						cor.vec = append(cor.vec, cor(AA[k,],AA[k+1,]))
					}
				}else{
					cor.vec = 0
				}
			}
			if(max(abs(cor.vec))> 0.95){
				is.restart = TRUE
			}else if(error == -1000){
				is.restart = TRUE
			}else{
				is.restart = FALSE
			}

			TIME = append(TIME, (proc.time() - t1)[3])
		}

	}

	# Compute BIC for model selection: 
	# bic1 = -2 * log(Likelihood) + log(n) * df 
	# bic2 = log(SSE) + log(n*d1*d2*d3)/(n*d1*d2*d3) * df
	A_final = list()
	mydf = 0
	for(k in 1:K){
		A_final[[k]] = Beta[[1]][k,]
		mydf = sum(Beta[[1]]!=0)
		for(j in 2:m){
			A_final[[k]] = outer(A_final[[k]], Beta[[j]][k,])
			mydf = mydf + sum(Beta[[j]]!=0)
		}	
		A_final[[k]] = W[k] * A_final[[k]]		
	}
	fitting_error = 0	
	residual = list()
	for(i in 1:n){	
		Diff = array(0, dim(Y[[1]]))	
		for(k in 1:K){
			Diff = Diff + A_final[[k]] * as.numeric(t(Beta[[m+1]][k,]) %*% X[i,])
		}
		residual[[i]] = Y[[i]] - Diff
		fitting_error = fitting_error + fnorm(residual[[i]])^2
	}
	fitting_error = 2 * fitting_error/n
	bic = fitting_error + log(n) * mydf
	bic2 = log(n * fitting_error / 2) + log(n * prod(DD[1:m]) )/(n * prod(DD[1:m])) * mydf

	OUT = list()
	OUT$B.hat = B.hat
	OUT$Beta.hat = Beta
	OUT$W.hat = W
	OUT$TERMINATE.ERROR = TERMINATE.ERROR
	OUT$OBJ = OBJ
	OUT$num.iter = t
	OUT$bic = bic
	OUT$bic2 = bic2
	OUT$fitting_error = fitting_error
	OUT$df = mydf
	OUT$i.restart = i.restart
	return(OUT)	
}
	  


###########################################################################
## Vectorized OLS: fit each element of Y on X one-at-a-time
###########################################################################

myols <- function(Y, X, is.symmetric = FALSE) {
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * d2 * d3
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	
	m = length(dim(Y[[1]]))
	X = as.matrix(X)
	n = nrow(X)
	p = ncol(X)
	DD = c(lapply(Y,dim)[[1]], p) 	# DD = (d_1, ..., d_m, p)
	Z = solve(t(X) %*% X) %*% t(X)
	B.hat = array(0, DD)

	if(m == 2){
		#matrix outcome
		for(i1 in 1:DD[1]){
			for(i2 in 1:DD[2]){
				for(j in 1:DD[3]){
					for(i in 1:n){
						B.hat[i1,i2,j] = B.hat[i1,i2,j] + Y[[i]][i1,i2] * Z[j,i] 
					}
				}
			}
		}
	}else if(m == 3){
		# 3rd order tensor outcome
		for(i1 in 1:DD[1]){
			for(i2 in 1:DD[2]){
				for(i3 in 1:DD[3]){
					for(j in 1:DD[4]){
						for(i in 1:n){
							B.hat[i1,i2,i3,j] = B.hat[i1,i2,i3,j] + Y[[i]][i1,i2,i3] * Z[j,i] 
						}
					}
				}
			}
		}
	}

	if(is.symmetric == TRUE && m==2){
		## in matrix symmetric output case, use B.hat.new = [B.hat + t(B.hta)]/2
		B.hat.new = B.hat
		for(jj in 1:p){
			B.hat.new[,,jj] = (B.hat[,,jj] + t(B.hat[,,jj]))/2
		}
		B.hat = B.hat.new
	}

	B.hat
}




##########################################################################################################################
## Deprecated: Vectorized OLS version2: fit each element of Y on X one-at-a-time. Same output as myols. 
###########################################################################################################################

myols2 <- function(Y, X) {
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * d2 * d3
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	
	m = length(dim(Y[[1]]))
	X = as.matrix(X)
	n = nrow(X)
	p = ncol(X)
	DD = c(lapply(Y,dim)[[1]], p) 	# DD = (d_1, ..., d_m, p)
	B.hat = array(0, DD)

	## stack Y into a big array
	Y.stack = array(0, c(lapply(Y,dim)[[1]], n))
	for(i in 1:n){
		if(m == 2){
			Y.stack[,,i] = Y[[i]]
		}else if(m == 3){
			Y.stack[,,,i] = Y[[i]]
		}
	}

	## fit one entry of response tensor at a time: lm()
	if(m == 2){
		#matrix outcome
		for(i1 in 1:DD[1]){
			for(i2 in 1:DD[2]){
				y.vec = Y.stack[i1,i2,]
				data.lm = data.frame(cbind(y.vec, X))
				out.lm = lm(y.vec ~ 0+. , data = data.lm)
				B.hat[i1,i2,] = as.numeric(out.lm$coefficients)
			}
		}
	}else if(m == 3){
		# 3rd order tensor outcome
		for(i1 in 1:DD[1]){
			for(i2 in 1:DD[2]){
				for(i3 in 1:DD[3]){					
					y.vec = Y.stack[i1,i2,i3,]
					data.lm = data.frame(cbind(y.vec, X))
					out.lm = lm(y.vec ~ 0+. , data = data.lm)
					B.hat[i1,i2,i3,] = as.numeric(out.lm$coefficients)
				}
			}
		}
	}

	B.hat
}



#############################################################################################
## Sparse OLS via hard thresholding: fit each element of Y on X one-at-a-time
#############################################################################################

mytruncateols <- function(Y, X, gamma) {
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * d2 * d3
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	# gamma: tuning parameter in the truncation step.
	
	m = length(dim(Y[[1]]))
	X = as.matrix(X)
	n = nrow(X)
	p = ncol(X)
	DD = c(lapply(Y,dim)[[1]], p) 	# DD = (d_1, ..., d_m, p)
	Z = solve(t(X) %*% X) %*% t(X)
	B.hat = array(0, DD)

	if(m == 2){
		#matrix outcome
		for(i1 in 1:DD[1]){
			for(i2 in 1:DD[2]){
				for(j in 1:DD[3]){
					for(i in 1:n){
						B.hat[i1,i2,j] = B.hat[i1,i2,j] + Y[[i]][i1,i2] * Z[j,i] 
					}
				}
			}
		}
	}else if(m == 3){
		# 3rd order tensor outcome
		for(i1 in 1:DD[1]){
			for(i2 in 1:DD[2]){
				for(i3 in 1:DD[3]){
					for(j in 1:DD[4]){
						for(i in 1:n){
							B.hat[i1,i2,i3,j] = B.hat[i1,i2,i3,j] + Y[[i]][i1,i2,i3] * Z[j,i] 
						}
					}
				}
			}
		}
	}

	B.hat[which(abs(B.hat) <= gamma)] = 0
	B.hat

}




#############################################################################################################################
## Sparse OLS via lasso: implement via the remMap R package with automatically BIC tuning by Peng et al. (2010)
## \min_{B} \| Y - B^{\top} X \|_F^2 + \lambda \|B\|_1, with Y \in R^{n * d1d2d3}, X \in R^{n * p}, B \in R^{p * d1d2d3}
#############################################################################################################################

mysparseols <- function(Y, X, lam.vec = exp(seq(-3,3, length=10)), tuning = "BIC", is.symmetric = FALSE) {
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * d2 * d3
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	# lam.vec: range of tuning parameters for lasso.
	# tuning: \in {"BIC", "CV"}: one of these two tuning methods
	
	m = length(dim(Y[[1]]))
	X = as.matrix(X)
	n = nrow(X)
	p = ncol(X)
	DD = c(lapply(Y,dim)[[1]], p) 	# DD = (d_1, ..., d_m, p)
	B.hat = array(0, DD)

	## stack Y into a big array
	Y.mat = matrix(0, n, prod(lapply(Y,dim)[[1]]))
	for(i in 1:n){
		Y.mat[i,] = as.numeric(Y[[i]])
	}

	## fit sparse multiple response regression via remMap. only tune lamL1 for lasso penalty. set the group penalty lamL2 = 0.
	lamL1.v = sort(lam.vec, decreasing = TRUE)
	lamL2.v = 0
	if(tuning == "BIC"){

		out.bic = remMap.BIC(X, Y.mat, lamL1.v, lamL2.v)
		if(min(out.bic$BIC) == -Inf && max(out.bic$BIC) == -Inf){
			lambda.opt = mean(lamL1.v)
		}else{
			lambda.opt = lamL1.v[which.min(out.bic$BIC)[1]]
		}
		out.bic.opt = remMap(X, Y.mat, lamL1 = lambda.opt, lamL2 = 0)
		B.mat.opt = out.bic.opt$phi

	}else if(tuning == "CV"){

		if(ncol(X) == 1){
			# error in remMap.cv. add a redundant column to X.
			Xnew = cbind(X,rep(1,nrow(X)))
			out.cv = remMap.CV(Xnew, Y.mat, lamL1.v, lamL2.v, fold = 5)
			lam.opt = as.numeric(out.cv$l.index[1,which.min(out.cv$ols.cv)[1]])
			out.cv.opt = remMap(Xnew, Y.mat, lamL1 = lam.opt, lamL2 = 0)
			B.mat.opt = matrix(out.cv.opt$phi[1,],p, ncol(Y.mat))			
		}else{
			out.cv = remMap.CV(X, Y.mat, lamL1.v, lamL2.v, fold = 5)
			lam.opt = as.numeric(out.cv$l.index[1,which.min(out.cv$ols.cv)[1]])
			out.cv.opt = remMap(X, Y.mat, lamL1 = lam.opt, lamL2 = 0)
			B.mat.opt = out.cv.opt$phi			
		}
	}

	for(i in 1:p){
		if(m==2){
			B.hat[,,i] = array(B.mat.opt[i,], lapply(Y,dim)[[1]])
		}else if(m==3){
			B.hat[,,,i] = array(B.mat.opt[i,], lapply(Y,dim)[[1]])
		}

	}

	if(is.symmetric == TRUE && m==2){
		## in matrix symmetric output case, use B.hat.new = [B.hat + t(B.hta)]/2
		B.hat.new = B.hat
		for(jj in 1:p){
			B.hat.new[,,jj] = (B.hat[,,jj] + t(B.hat[,,jj]))/2
		}
		B.hat = B.hat.new
	}

	B.hat

}



#######################################################################################################################################
## Higher-order low-rank regression (HOLLR) by Rabusseau and Kadri (2016, NIPS)
## \min_{\cW} { \| \cW \times_1 X - \cY \|_F^2 + \gamma \|\cW\|_F^2 } s.t. rank (\cW) <= (R0, R1, ..., Rm) 
## Here \cY \in R^{n \times d1 \times d2 \times ... \times dm\}, X \in R^{n * p}, \cW \in R^{p \times d1 \times d2 \times ... \times dm \}
#######################################################################################################################################

myholrr <- function(Y, X, R.vec = NULL, gamma = NULL, is.symmetric = FALSE) {
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * ... * dm
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	# R.vec = (R0, R1, ..., Rm) \in R^{m+1} is the input rank for each mode of \cW. for fair comparison with ours, use same rank in all modes.
	# gamma: tuning parameter in ridge regression constraint. 

	if(is.null(R.vec) | is.null(gamma)){
		# tuning via CV
		out.cv = myholrrcv(Y, X, is.symmetric = is.symmetric)
		R.vec = out.cv$opt.R.vec
		gamma = out.cv$opt.gamma
	}

	m = length(dim(Y[[1]]))
	X = as.matrix(X)
	n = nrow(X)
	p = ncol(X)
	DD = c(lapply(Y,dim)[[1]], p) 	# DD = (d_1, ..., d_m, p)
	B.hat = array(0, DD)

	## stack Y into a big array
	Y.tensor = array(0, c(n, lapply(Y,dim)[[1]]))
	for(i in 1:n){
		if(m == 2){
			Y.tensor[i,,] = Y[[i]]
		}else if(m == 3){
			Y.tensor[i,,,] = Y[[i]]
		}
	}
	Y.tnsr = as.tensor(Y.tensor)

	## compute U0, ... Um
	Y.mat = k_unfold(Y.tnsr, 1)@data
	vec.tmp = t(X) %*% Y.mat
	mat.tmp = solve(t(X) %*% X + gamma * diag(p)) %*% vec.tmp %*% t(vec.tmp)
	U0 = as.matrix(svd(mat.tmp)$u[,1:R.vec[1]])
	U.all = list()
	for(i in 1:m){
		Y.mat2 = k_unfold(Y.tnsr, i+1)@data
		mat.tmp2 = Y.mat2 %*% t(Y.mat2)
		U.all[[i]] = as.matrix(svd(mat.tmp2)$u[,1:R.vec[i+1]])
	}

	## compute \cG
	M.mat = solve( t(U0) %*% (t(X) %*% X + gamma * diag(p)) %*% U0 ) %*% t(U0) %*% t(X)
	G.tnsr = ttm(Y.tnsr, M.mat, 1)
	for(i in 1:m){
		G.tnsr = ttm(G.tnsr, t(U.all[[i]]), i+1)
	}

	## compute coefficient \cW
	W.tnsr = ttm(G.tnsr, U0, 1)
	for(i in 1:m){
		W.tnsr = ttm(W.tnsr, U.all[[i]], i+1)
	}
	W.tensor = W.tnsr@data

	## convert \cW to the final coefficient \cB \in R^{p \times d1 \times d2 \times ... \times dm \}
	for(i in 1:p){
		if(m == 2){
			B.hat[,,i]	= W.tensor[i,,]
		}else if(m == 3){
			B.hat[,,,i]	= W.tensor[i,,,]
		}
	}

	if(is.symmetric == TRUE && m==2){
	## in matrix symmetric output case, use B.hat.new = [B.hat + t(B.hta)]/2
		B.hat.new = B.hat
	for(jj in 1:p){
		B.hat.new[,,jj] = (B.hat[,,jj] + t(B.hat[,,jj]))/2
	}
		B.hat = B.hat.new
	}

	B.hat

}



#######################################################################################################################################
## 3-fold CV Tuning of Higher-order low-rank regression (HOLLR) by Rabusseau and Kadri (2016, NIPS)
## Set all ranks in tucker to be same for fair comparison of our CP decomposition.
#######################################################################################################################################

myholrrcv <- function(Y, X, R.list = NULL, gamma.list = exp(seq(-3,3, length=5)), seed = 1, is.symmetric = FALSE){
	# Y: a list of response tensor Y = {Y_1, ..., Y_n} wit Y[[i]] = Y_i a tensor of dimension d1 * ... * dm
	# X \in R^{n * p}: covariate X = (x_1, ..., x_n)^T with x_i a vector of dimension p
	# If R.list = NULL, use the range of (1, ..., \min{d1, ..., dm}) with length = 5.
	# gamma.list: list of tuning parameters in ridge regression constraint. 

	m = length(dim(Y[[1]]))
	X = as.matrix(X)
	n = nrow(X)
	p = ncol(X)
	DD = c(lapply(Y,dim)[[1]], p) 	# DD = (d_1, ..., d_m, p)

	if( is.null(R.list)){
		R.max = min(lapply(Y,dim)[[1]])
		R.list = as.integer(seq(1, R.max, length = 5))
	}

 	k1 = length(R.list)
    lambda1.v = sort(R.list)[k1:1]
    k2 = length(gamma.list)
    lambda2.v = sort(gamma.list)[k2:1]
    l1.index <- as.vector(matrix(lambda1.v, nrow = k1, ncol = k1, 
        byrow = FALSE))
    l2.index <- as.vector(matrix(lambda2.v, nrow = k2, ncol = k2, 
        byrow = TRUE))
    l.index <- rbind(l1.index, l2.index)


 	set.seed(seed)
    index.cv <- NULL
    fold = 3
    ran.order = sample(1:n, n, replace = F)
    f.n = floor(n/fold)
    for (f in 1:(fold - 1)) {
        index.cv[[f]] <- ran.order[(f - 1) * f.n + 1:f.n]
    }
    index.cv[[fold]] <- ran.order[((fold - 1) * f.n + 1):n]
	rss.cv.cv <- NULL
	for (f in 1:fold) {
	    index.cur.cv <- index.cv[[f]]
	    X.m <- as.matrix(X[-(index.cur.cv), ])
	    Y.m <- Y[-(index.cur.cv)]
	    X.t <- as.matrix(X[index.cur.cv, ])
	    Y.t <- Y[index.cur.cv]
	    rss.cv.c <- matrix(0, k1, k2)

	    for (i in 1:k1) {
	        for (j in 1:k2) {
	            cur.lam1 = lambda1.v[i]
	            cur.lam2 = lambda2.v[j]
	            Bhat.temp = myholrr(Y.m, X.m, R.vec = c(1, rep(cur.lam1, m)), gamma = cur.lam2, is.symmetric = is.symmetric)
	            Y.predict = (ttm(as.tensor(Bhat.temp), X.t, m+1))@data
	            Y.t.array = array(0, c(lapply(Y,dim)[[1]], length(Y.t)) )
	            for(k in 1:length(Y.t)){
	            	if(m == 2){
	            		Y.t.array[,,k] = Y.t[[k]]
	            	}else if(m == 3){
	            		Y.t.array[,,,k] = Y.t[[k]]
	            	}
	            }
	            rss.cv.c[i,j] = sum((Y.predict - Y.t.array)^2)
	        }
	    }
	  	rss.cv.cv[[f]] <- rss.cv.c
	}

	rss.cv.sum <- array(0, c(k1, k2, fold))
    for (f in (1:fold)) {
        rss.cv.sum[,,f]<- rss.cv.cv[[f]]
    }
    rss.cv <- apply(rss.cv.sum, c(1, 2), sum)
    rownames(rss.cv) = paste("lamL1=", round(lambda1.v, 3))
    colnames(rss.cv) = paste("lamL2=", round(lambda2.v, 3))
    opt.R.vec = c(1, rep(l.index[1,which.min(rss.cv)[1]], m))
    opt.gamma = l.index[2,which.min(rss.cv)[1]]

	result = list(opt.R.vec = opt.R.vec, opt.gamma = opt.gamma, rss.cv = rss.cv, l.index = l.index)
    return(result)

}
























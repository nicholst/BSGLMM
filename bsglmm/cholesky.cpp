/*
 *  cholesky.cpp
 *  NewHWCox04
 *
 *  Created by Jian Kang on 4/6/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "randgen.h"
#include "cholesky.h"

int cholesky_decomp(double *A, int num_col)
{
	int k,i,j;
	
	for (k=0;k<num_col;k++) {
		if (A[k + k*num_col] <= 0) {
			/*      printf("cholesky_decomp: error matrix A not SPD %lf %d\n",A[k][k],k);*/
			return 0;
		}
		
		A[k + k*num_col] = sqrt(A[k + k*num_col]);
		
		for (j=k+1;j<num_col;j++)
			A[k + j*num_col] /= A[k + k*num_col];
		
		for (j=k+1;j<num_col;j++)
			for (i=j;i<num_col;i++)
				A[j+i*num_col] = A[j + i*num_col] - A[k + i*num_col] * A[k + j*num_col];
	}
	return 1;
	
	/* this is the GAXPY update */
	/*  for (j=0;j<num_col;j++) {
	 if (j > 0) {
	 for (i=j;i<num_col;i++) {
	 tmp = 0;
	 for (k=0;k<j;k++)
	 tmp += A[i][k]*A[j][k];
	 A[i][j] -= tmp;
	 }
	 if (A[j][j] <=0) {
	 printf("cholesky_decomp: error matrix A not SPD %lf %d\n",A[k][k],k);
	 return 0;
	 }
	 }
	 tmp = sqrt(A[j][j]);
	 for (i=j;i<num_col;i++)
	 A[i][j] /= tmp;
	 }
	 return 1;*/
}

void cholesky_invert(int len,double *H)
{
	/* takes G from GG' = A and computes A inverse */
	int i,j,k;
	double temp,*INV;
	
	INV = (double *)calloc(len*len,sizeof(double));

	for (i=0;i<len;i++)
		INV[i + i*len] = 1;
	
	for (k=0;k<len;k++) {
		INV[0 + k*len] /= H[0];
		for (i=1;i<len;i++) {
			temp = 0.0;
			for (j=0;j<i;j++)
				temp += H[j + i*len]*INV[j + k*len];
			INV[i + k*len] = (INV[i + k*len] - temp)/H[i + i*len];
		}
		INV[len-1 + k*len] /= H[len-1 + (len-1)*len];
		for (i=len-2;i>=0;i--) {
			temp = 0.0;
			for (j=i+1;j<len;j++)
				temp += H[i + j*len]*INV[j + k*len];
			INV[i + k*len] = (INV[i + k*len] - temp)/H[i + i*len];
		}
	}
	for (i=0;i<len;i++)
		for (j=0;j<len;j++)
			H[j + i*len] = INV[j + i*len];
	free(INV);
}

int banded_cholesky_decomp(double *A,int num_col,int p)
{
	int i,j,k,lk,lambda;
	double tmp;
	for (j=0;j<num_col;j++) {
		lk = (0 > j-p+1) ? 0:j-p+1;
		for (k=lk;k<j;k++) {
			lambda = ((lambda=k+p) < num_col) ? lambda : num_col;  
			for (i=j;i<lambda;i++) {
				tmp = A[k +j*num_col]*A[k + i*num_col];
				A[j + i*num_col] -= tmp;
			}
		}
		lambda = ((lambda=j+p) < num_col) ? lambda: num_col;
		if (A[j + j*num_col] <= 0)
			return 0;
		tmp = sqrt(A[j + j*num_col]);
		for (i=j;i<lambda;i++)
			A[j + i*num_col] /= tmp;
	}
	return 1;
}

void banded_cholesky_invert(int len,double *H,int p)
{
	/* takes G from GG' = A and computes A inverse  where A is a banded matrix with bandwidth p*/
	int i,j,k,lj;
	double temp,*INV;
	
	INV = (double *)calloc(len*len,sizeof(double)); 
	for (i=0;i<len;i++)
		INV[i + i*len] = 1;
	
	for (k=0;k<len;k++) {
		INV[0 + k*len] /= H[0];
		for (i=1;i<len;i++) {
			temp = 0.0;
			lj = (0>(lj=i-p+1)) ? 0:lj;
			for (j=lj;j<i;j++)
				temp += H[j + i*len]*INV[j + k*len];
			INV[i + k*len] = (INV[i + k*len] - temp)/H[i + i*len];
		}
		INV[len-1 + k*len] /= H[len-1 * (len-1)*len];
		for (i=len-2;i>=0;i--) {
			temp = 0.0;
			lj = (len<(lj=i+p)) ? len:lj;
			for (j=i+1;j<lj;j++)
				temp += H[i + j*len]*INV[j + k*len];
			INV[i + k*len] = (INV[i + k*len] - temp)/H[i + i*len];
		}
	}
	for (i=0;i<len;i++)
		for (j=0;j<len;j++)
			H[j + i*len] = INV[j + i*len];
	free(INV);
}

int rmvnorm(double *result,double *A,int size_A,double *mean,unsigned long *seed,int flag)
{
	int i,j;
	double *runiv;
	double snorm(unsigned long *);
	int cholesky_decomp(double *,int);
	
	runiv = (double *)calloc(size_A,sizeof(double));
	j = 1;
	if (!flag)
		j = cholesky_decomp(A,size_A);
	if (j) {
		for (i=0;i<size_A;i++) 
			runiv[i] = snorm(seed);
		for (i=0;i<size_A;i++) {
			result[i] = 0;
			for (j=0;j<=i;j++)
				result[i] += A[j +i*size_A]*runiv[j];
			result[i] += mean[i];
		}
		free(runiv);
		return 1;
	}
	else {
		free(runiv);
		return 0;
	}
}


int rmvnorm_idx(double *result,int* idx, double *A,int size_A,double *mean,unsigned long *seed,int flag)
{
	int i,j;
	double *runiv;
	double snorm(unsigned long *);
	int cholesky_decomp(double *,int);
	
	runiv = (double *)calloc(size_A,sizeof(double));
	j = 1;
	if (!flag)
		j = cholesky_decomp(A,size_A);
	if (j) {
		for (i=0;i<size_A;i++) 
			runiv[i] = snorm(seed);
		for (i=0;i<size_A;i++) {
			result[idx[i]] = 0;
			for (j=0;j<=i;j++)
				result[idx[i]] += A[j + i*size_A]*runiv[j];
			result[idx[i]] += mean[i];
		}
		free(runiv);
		return 1;
	}
	else {
		free(runiv);
		return 0;
	}
}

int rwishart1(double *result,double *S,int size_S,int df,unsigned long *seed)
{
	int i,j,k;
	double *x,*zero;
	int rmvnorm(double *,double *,int,double *,unsigned long *,int);
	
	if ((double)df < (double)size_S)
		return 0;
	

	x = (double *)calloc(size_S,sizeof(double));
	zero = (double *)calloc(size_S,sizeof(double));
	
	for (i=0;i<size_S;i++) 
		for (j=0;j<size_S;j++) 
			result[j + i*size_S] = 0;
	for (k=0;k<df;k++) {
		if (rmvnorm(x,S,size_S,zero,seed,1)) {
			for (i=0;i<size_S;i++) 
				for (j=0;j<size_S;j++) 
					result[j + i*size_S] += x[i]*x[j];
		}
		else {
			free(zero);
			free(x);
			return 0;
		}
	}
	free(zero);
	free(x);
	return 1;
}


void riwishart1(double *result,double *S,int size_S,int df,unsigned long *seed)
{
	rwishart1(result, S, size_S, df, seed);
	cholesky_decomp(result, size_S);
	cholesky_invert(size_S, result);
}

int rwishart2(double *result,double *S,int size_S,int df,unsigned long *seed)
{
	int i,j,k,L,K;
	double *B;
	
	if ((double)df < (double)size_S)
		return 0;
	
	for (i=0;i<size_S;i++) 
		for (j=0;j<size_S;j++) 
			result[j + i*size_S] = 0;
	
	B = (double *)calloc(size_S*size_S,sizeof(double));
		
	for (i=0;i<size_S;i++) {
		result[i + i*size_S] = sqrt(rgamma(0.5*double(df - i),0.5,seed));
		for (j=0;j<i;j++)
			result[j + i*size_S] = snorm(seed);
	}

	for (i=0;i<size_S;i++) {
		for (j=0;j<=i;j++)  {
			L = (i <= j) ? i:j;
			K = (i >= j) ? i:j;
			for (k=L;k<=K;k++)
				B[j + i*size_S] += S[k + i*size_S]*result[j + k*size_S];
		}
	}
		
	for (i=0;i<size_S;i++) {
		for (j=0;j<size_S;j++) {
			result[j + i*size_S] = 0;
			K = (i <= j) ? i:j;
			for (k=0;k<=K;k++)
				result[j + i*size_S] += B[k + i*size_S]*B[k + j*size_S];
		}
	}

	free(B);

	return 1;
}


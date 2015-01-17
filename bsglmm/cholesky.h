#ifndef _CHOLESKY_H_
#define _CHOLESKY_H_

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/***************************************************************************/
/*                                                                         */
/* cholesky_decomp                                                         */
/*                                                                         */
/* This routine takes a symmetric positive definite matrix and performs    */
/* a cholesky decomposition.  It replaces the lower triangular part of     */
/* A with G.                                                               */
/*                                                                         */
/* The cholesky decomposition, decomposes A into A = GG'.  This version of */
/* the algorithm is an outer product (rank-1) update of the principal      */
/* submatrices of A.                                                       */
/*                                                                         */
/* See "Matrix Computations" by Golub and Van Loan.  2nd edition, Johns    */
/* Hopkins University Press, 1989. p. 423ff                                */
/*                                                                         */
/* Input  A - a SPD matrix                                                 */
/*        num_col - size of A                                              */
/*                                                                         */
/* Returns 1 upon success, 0 if A not SPD                                  */
/***************************************************************************/
int cholesky_decomp(double *A, int num_col);

void cholesky_invert(int len,double *H);

int banded_cholesky_decomp(double *A,int num_col,int p);

void banded_cholesky_invert(int len,double *H,int p);

int rmvnorm(double *result,double *A,int size_A,double *mean,unsigned long *seed,int flag);

int rmvnorm_idx(double *result, int* idx, double *A,int size_A,double *mean,unsigned long *seed,int flag);

int rwishart1(double *result,double *S,int size_S,int df,unsigned long *seed);

int rwishart2(double *result,double *S,int size_S,int df,unsigned long *seed);

void riwishart1(double *result,double *S,int size_S,int df,unsigned long *seed);

#endif //_CHOLESKY_H_

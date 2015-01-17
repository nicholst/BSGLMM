/*
 *  updateCovar.cpp
 *  BinCAR
 *
 *  Created by Timothy Johnson on 5/22/12.
 *  Copyright 2012 University of Michigan. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "binCAR.h"
#include "randgen.h"
#include "cholesky.h"

extern int NSUBS;
extern int NROW;
extern int NCOL;
extern int NDEPTH;
extern int TOTVOX;
extern int TOTVOXp;
extern int NCOVAR;
extern int NSIZE;
extern int NSUBTYPES;
extern int iter;
extern INDEX *INDX;
extern int *hostIdx;
extern int *hostIdxSC;
extern int MODEL;
extern float *XXprime;
extern float *XXprime_Fix;

#define idx(i,j,k) (i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k)
#define idx2(i,j,k) (i + (NROW)*j + (NROW)*(NCOL)*k)

int cholesky_decomp_float(float *A, int num_col)
{
	int k,i,j;
	
	for (k=0;k<num_col;k++) {
		if (A[k + k*num_col] <= 0) {
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
}

void cholesky_invert_float(int len,float *H)
{
	/* takes G from GG' = A and computes A inverse */
	int i,j,k;
	float temp,*INV;
	
	INV = (float *)calloc(len*len,sizeof(float));
		
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

int rmvnorm_float(float *result,float *A,int size_A,float *mean,unsigned long *seed,int flag)
{
	int i,j;
	float *runiv;
	
	runiv = (float *)calloc(size_A,sizeof(float));
	
	j = 1;
	if (!flag)
		j = cholesky_decomp_float(A,size_A);
	if (j) {
		for (i=0;i<size_A;i++) 
			runiv[i] = snorm(seed);//curand_normal(state);
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

void Spat_Coefff(float *covar,float *alpha,float *Z,float *WM,float *SpatCoef,float beta,float *SpatCoefPrec,unsigned char *nbrs,int *vox,const int N,unsigned long *seed)
{
	int idx,IDXSC; 
	int voxel,isub,ii,jj;
	
	float *sumV;
	float *mean;
	float *var;
	float *tmp2;
	float tmp=0;
	
	sumV = (float *)calloc(NCOVAR,sizeof(float));
	mean = (float *)calloc(NCOVAR,sizeof(float));
	tmp2 = (float *)calloc(NCOVAR,sizeof(float));
	var = (float *)calloc(NCOVAR*NCOVAR,sizeof(float));
	
	for (idx=0;idx<N;idx++) { 
		voxel =  vox[idx];
		IDXSC = hostIdxSC[voxel];
		float bwhtmat = beta*WM[hostIdx[voxel]];
		
		for (ii=0;ii<NCOVAR;ii++) {
			mean[ii] = 0;
			sumV[ii] = SpatCoef[(ii*TOTVOXp+hostIdxSC[voxel-1])]
		    			+ SpatCoef[(ii*TOTVOXp+hostIdxSC[voxel+1])]
		    	 		+ SpatCoef[(ii*TOTVOXp+hostIdxSC[voxel-(NROW+2)])]
		    			+ SpatCoef[(ii*TOTVOXp+hostIdxSC[voxel+(NROW+2)])]
		      			+ SpatCoef[(ii*TOTVOXp+hostIdxSC[voxel-(NROW+2)*(NCOL+2)])]
		      			+ SpatCoef[(ii*TOTVOXp+hostIdxSC[voxel+(NROW+2)*(NCOL+2)])];
		}
		
		for (isub=0;isub<NSUBS;isub++) {
			float ZZ = Z[isub*TOTVOX+hostIdx[voxel]];
			tmp = ZZ - bwhtmat;		
			
			for (ii=0;ii<NCOVAR;ii++) {
				mean[ii] += covar[ii*NSUBS+isub]*tmp;
			}
		}

		for (ii=0;ii<NCOVAR;ii++) { 
			for (jj=0;jj<NCOVAR;jj++) { 
				mean[ii] += SpatCoefPrec[jj + ii*NCOVAR]*sumV[jj];
				var[jj + ii*NCOVAR] = XXprime[jj + ii*NCOVAR] + nbrs[idx]*SpatCoefPrec[jj + ii*NCOVAR];
			}
		}

		cholesky_decomp_float(var,NCOVAR);
		cholesky_invert_float(NCOVAR, var);
		
		for (ii=0;ii<NCOVAR;ii++) {
			tmp2[ii] = 0;
			for (jj=0;jj<NCOVAR;jj++)
				tmp2[ii] += var[jj + ii*NCOVAR]*mean[jj];
		}			
		// draw MVN and with mean tmp2 and variance var --- result in mean;
		rmvnorm_float(mean,var,NCOVAR,tmp2,seed,0);
			
		for (int ii=0;ii<NCOVAR;ii++) 
			SpatCoef[(ii*TOTVOXp+IDXSC)] = mean[ii];

	}  
	free(sumV);
	free(mean);
	free(var); 
	free(tmp2);
}

void updateSpatCoef(float *covar,float *spatCoef,float *SpatCoefPrec,float *alpha,float *alphaMean,float *Z,unsigned char *msk,float beta,float *WM,unsigned long *seed)
{
	unsigned char nbrs;
	float mean=0,var=0,tmpvar = 0;
	float c = 0,sumV = 0;
	float y,t;


	float cmean = 0,ncnt=0;
	
	for (int i=0;i<2;i++) {//printf("%d ",i);fflush(NULL);
		Spat_Coefff(covar,alpha,Z,WM,spatCoef,beta,SpatCoefPrec,INDX[i].hostNBRS,INDX[i].hostVox,INDX[i].hostN,seed);// printf("%d\n",i);fflush(NULL);
	}
	
/*	for (int this_one=0;this_one<NCOVAR;this_one++) {
		c = 0; cmean = 0;
		for (int i=0;i<TOTVOXp;i++) {
			y = spatCoef[this_one*TOTVOXp+i] - c;	
			t = cmean + y;
			c = (t - cmean) - y;
			cmean = t;
		}

		cmean /= (float)TOTVOX;

		alphaMean[this_one] = cmean;
	}*/

}


double Spat_Coef_Precision(float *SpatCoef1,float *SpatCoef2,const int N,int *vox)
{	
	float c = 0;
	float t;
	float temp = 0;
	double y;
	float tmp,SC1,SC2,SCt1,SCt2;
	int voxel,tst,IDXSC;
	
	y = 0;
	for (int idx=0;idx<N;idx++) { 
		voxel =  vox[idx];
		IDXSC = hostIdxSC[voxel];
//		y = 0;
		
		SC1 = SpatCoef1[IDXSC];
		SC2 = SpatCoef2[IDXSC];
		SCt1 = SpatCoef1[hostIdxSC[voxel+1]];
		SCt2 = SpatCoef2[hostIdxSC[voxel+1]];
		tst = (SCt1 != 0);
		tmp = (SC1 - SCt1)*(SC2 - SCt2)*tst;
		y += tmp;

		SCt1 = SpatCoef1[hostIdxSC[voxel+(NROW+2)]];
		SCt2 = SpatCoef2[hostIdxSC[voxel+(NROW+2)]];
		tst = (SCt1 != 0);
		tmp = (SC1 - SCt1)*(SC2 - SCt2)*tst;
		y += tmp;

		SCt1 = SpatCoef1[hostIdxSC[voxel+(NROW+2)*(NCOL+2)]];
		SCt2 = SpatCoef2[hostIdxSC[voxel+(NROW+2)*(NCOL+2)]];
		tst = (SCt1 != 0);
		tmp = (SC1 - SCt1)*(SC2 - SCt2)*tst;
		y += tmp;

/*		y -= c;
		t = temp + y;
		c = (t - temp) -y;
		temp = t;*/		
	}  
//	return temp; 
	return y; 
}	

void updateSpatCoefPrec(float *SpatCoefPrec,float *SpatCoef,unsigned char *msk,unsigned long *seed)
{
	float alpha,beta;
	double *var;
	float prior_alpha=3.,prior_beta=2.;
	float c = 0;
	float y,t;
	float temp = 0;

	beta = 0;
	
	var = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
	for (int ii=0;ii<NCOVAR;ii++) {
		for (int jj=0;jj<=ii;jj++) {
			for (int i=0;i<2;i++) {
				var[jj + ii*NCOVAR] += (double)Spat_Coef_Precision(&(SpatCoef[ii*TOTVOXp]),&(SpatCoef[jj*TOTVOXp]),INDX[i].hostN,INDX[i].hostVox);			
			}
			var[ii + jj*NCOVAR] = var[jj + ii*NCOVAR];
			if (ii==jj) var[ii + ii*NCOVAR] += 1;	
		}
	}
	
	cholesky_decomp(var, NCOVAR);
	cholesky_invert(NCOVAR, var);
	cholesky_decomp(var,NCOVAR);
	
	double *tmp = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
	rwishart2(tmp,var,NCOVAR,TOTVOX+NCOVAR+1-1,seed);

	for (int ii=0;ii<NCOVAR;ii++) 
		for (int jj=0;jj<=ii;jj++)   // filling in the lower triangular part of the matrix
			SpatCoefPrec[jj + ii*NCOVAR] = SpatCoefPrec[ii + jj*NCOVAR] = (float)tmp[jj + ii*NCOVAR];

	free(tmp);
	free(var);
}


void updateZ(float *Z,float *alpha,unsigned char *data,unsigned char *msk,float beta,float *WM,float *spatCoef,float *covar,unsigned long *seed)
{
	float mean=0;
	
	for (int i=0;i<2;i++) {
		for (int j=0;j<INDX[i].hostN;j++) {
			int vox = INDX[i].hostVox[j];
			int IDX = hostIdx[vox];
			int IDXSC = hostIdxSC[vox];
			mean = beta*WM[IDX];
			for (int isub=0;isub<NSUBS;isub++) {
				float tmp = 0;
				for (int ii=0;ii<NCOVAR;ii++)
					tmp += spatCoef[ii*TOTVOXp+IDXSC]*covar[ii*NSUBS+isub];
				if (data[isub*TOTVOX+IDX]==1) {
					Z[isub*TOTVOX+IDX] = (float)truncNorm2((double)(mean+tmp),1.,0.0,500,seed);
				}
				else {
					Z[isub*TOTVOX+IDX] = (float)truncNorm2((double)(mean+tmp),1.,-500,-0.0,seed);
				}
			}
		}
	}
}

double ProbSDNorm(const double x)
//Calculate the cumulative probability of standard normal distribution;
{
	const double b1 =  0.319381530;
	const double b2 = -0.356563782;
	const double b3 =  1.781477937;
	const double b4 = -1.821255978;
	const double b5 =  1.330274429;
	const double p  =  0.2316419;
	const double c  =  0.39894228;
	
	if(x >= 0.0) {
		double t = 1.0 / ( 1.0 + p * x );
		return (1.0 - c * exp( -x * x / 2.0 ) * t *
				( t *( t * ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
	}
	else {
		double t = 1.0 / ( 1.0 - p * x );
		return ( c * exp( -x * x / 2.0 ) * t *
				( t *( t * ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
	}
}

void compute_prb(float *prb,float *alpha,float *spatCoef,float beta,float *WM,unsigned char *msk)
{

	for (int ii=0;ii<NSUBTYPES;ii++) {
		for (int k=1;k<NDEPTH+1;k++)  {
			for (int j=1;j<NCOL+1;j++) { 
				for (int i=1;i<NROW+1;i++) {
					int IDX = idx(i,j,k); 
					if (msk[IDX])
						prb[ii*TOTVOX+hostIdx[IDX]] = ProbSDNorm((spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]+beta*WM[hostIdx[IDX]]));
				}
			}
		}
	}
}

void ProbSDNorm_prediction(float *covar,float *covarFix,float *pred,float *SpatCoef,float beta,float *WM,int *dIdx,int *dIdxSC,int *vox,const int TOTVOX,const int TOTVOXp,const int NSUBTYPES,const int NCOVAR,const int NCOV_FIX,const int isub,const int N)
{	
	float x,y;
	float bwhtmat=0;
	int voxel,ii,IDX,IDXSC;

	for (int idx=0;idx<N;idx++) {
		voxel =  vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];

		bwhtmat = beta*WM[IDX];
		
		y = 0;
		for (ii=NSUBTYPES;ii<NCOVAR;ii++) {
			y += (SpatCoef[ii*TOTVOXp+IDXSC])*covar[ii*NSUBS+isub];
		}
		for (ii=0;ii<NSUBTYPES;ii++) {
			x = SpatCoef[ii*TOTVOXp+IDXSC] + bwhtmat + y;	
			pred[ii*TOTVOX+IDX] = ProbSDNorm(x);									
		}
	}
}

double compute_predict(float beta,unsigned char *data,unsigned char *msk,float *covar,float *predict,double **Qhat)
{
	double *Mpredict,DIC;
	
	Mpredict = (double *)calloc(NSUBTYPES,sizeof(double));
	

	DIC = 0;
/*	for (int isub=0;isub<NSUBS;isub++) {
		if (MODEL != 1) {
			for (int i=0;i<2;i++) {
//				ProbLogitGPU_prediction<<<(INDX[i].hostN+511)/512, 512 >>>(deviceCovar,deviceCov_Fix,devicePredict,deviceFixMean,
//				deviceAlphaMean,deviceSpatCoef,beta,deviceWM,deviceIdx,deviceIdxSC,INDX[i].deviceVox,TOTVOX,TOTVOXp,
//				NSUBTYPES,NCOVAR,NCOV_FIX,isub,INDX[i].hostN,logit_factor);
			}
		}
		else {
			for (int i=0;i<2;i++) {
				ProbSDNorm_prediction(covar,covarFIX,predict,SpatCoef,beta,WM,hostIdx,hostIdxSC,INDX[i].hostVox,TOTVOX,TOTVOXp,
				NSUBTYPES,NCOVAR,NCOV_FIX,isub,INDX[i].hostN);
			}
		}
		
		//  Compute Prediction for each subtype	
		for (int ii=0;ii<NSUBTYPES;ii++) {
			Mpredict[ii] = 0;
			
			for  (int ii=0;ii<TOTVOX;i++) {
				double tmp = (double)data[ivox];
				double inv = (double)predict[ivox];
				Mpredict[ii] += tmp*logf(inv + 1E-35) + (1.0f - tmp)*logf(1.0 - inv + 1E-35);
			}
		}

		int subtype=0;
		for (int i=0;i<NSUBTYPES;i++) {
			if (covar[i*NSUBS+isub]) {
				subtype = i;
				break;
			}
		}
		DIC += Mpredict[subtype];
		for (int i=0;i<NSUBTYPES;i++) {
			Qhat[isub][i] += exp(Mpredict[i] - Mpredict[subtype]);
		}
	}	
	
	free(Mpredict);	
  */
	return(DIC);
}

double compute_prb_DIC(double *Mcov,double *MWM,float *covar,unsigned char *data,unsigned char *msk,int RSIZE)
{
	double x,y,DE=0;
	int tmp;
	
	for (int isub=0;isub<NSUBS;isub++) {
	 	for (int k=1;k<NDEPTH+1;k++)  {
			for (int j=1;j<NCOL+1;j++) { 
				for (int i=1;i<NROW+1;i++) {
					int IDX = idx(i,j,k); 
					int i1=i-1; 
					int j1=j-1;
					int k1=k-1;
					int IDX2 = idx2(i1,j1,k1);
					if (msk[IDX]) {
						if (MODEL != 1) {
							y = 0;
							for (int ii=0;ii<NCOVAR;ii++) {
								y += Mcov[ii*RSIZE+IDX2]*(double)covar[ii*NSUBS+isub];
							}

							x = exp(y + MWM[IDX2]);	
							x /= (1.0+x);
							tmp = (int)data[isub*TOTVOX+hostIdx[IDX]];
							DE += tmp*log(x+DBL_MIN) + (1-tmp)*log(1.0-x+DBL_MIN);									
						}
						else {
							y = 0;
							for (int ii=0;ii<NCOVAR;ii++) {
								y += Mcov[ii*RSIZE+IDX2]*(double)covar[ii*NSUBS+isub];
							}
							x = ProbSDNorm((const float)(y + MWM[IDX2]));
//							x = ProbSDNorm(y);
							tmp = (int)data[isub*TOTVOX+hostIdx[IDX]];
							DE += tmp*log(x+DBL_MIN) + (1-tmp)*log(1.0-x+DBL_MIN);									
						}
					}
				}
			}
		}
	}

	DE *= -2.0;
	return DE;
}



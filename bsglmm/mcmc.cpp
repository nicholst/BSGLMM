/*
 *  mcmc.cpp
 *
 *  Created by Timothy Johnson on 4/14/12.
 *  Copyright 2012 University of Michigan. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include <time.h>
#include "binCAR.h"
#include "randgen.h"
#include "cholesky.h"
#include <float.h>

typedef double MY_DATATYPE;
extern float logit_factor;
extern float t_df;
extern double *grp_prior;

int iter;
extern int MODEL;
extern int GPU;
extern int MAXITER;
extern int BURNIN;
extern int NSUBS;
extern int NROW;
extern int NCOL;
extern int TOTVOX;
extern int TOTVOXp;
extern int NDEPTH;
extern int NSUBTYPES;
extern int NCOVAR;
extern int NCOV_FIX;
extern int RESTART;
extern int *hostIdx;
extern int *deviceIdx;
extern int *hostIdxSC;
extern int *deviceIdxSC;
float *deviceSpatCoef;
extern float *deviceCovar;
extern float *deviceCov_Fix;
extern float *XXprime;
unsigned char *deviceData;
unsigned int *deviceResid;
extern char **SS;

//short *deviceData;
float *deviceWM;
float *deviceAlphaMean;
float *deviceFixMean;
float *deviceZ;
float *devicePhi;
float *devicePrb;
float *devicePredict;
float sqrt2pi =  2.506628274631;
float NCNT;
int MIX = 0;

int NSIZE;

#define idx(i,j,k) (i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k)
#define idx2(i,j,k) (i + (NROW)*j + (NROW)*(NCOL)*k)

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) {printf("\nCUDA Error: %s (err_num=%d) \n",cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}

void mcmc(float *covar,float *covar_fix,unsigned char *data,float *WM,unsigned char *msk,unsigned long *seed)
{
	int TOTSAV=0,IDX,PRNtITER,SAVE_ITER,FIRST;
	float *Z,*Phi,*alpha,*prb,*predict;
	double *Mprb,*MCov,*SCov,*MWM,**Qhat,ED=0;
	float beta=0,prior_mean_beta=0,prior_prec_beta=0.000001f;
	float *SpatCoefPrec,*SpatCoefMean,*spatCoef;

	float *alphaMean,*alphaPrec;//scarPrec=1;
	float *fixMean,*fixPrec;//scarPrec=1;
	float prior_alphaPrec_A=3.0f;
	float prior_alphaPrec_B=2.0f;
//		float prior_carPrec_A=3.0,prior_carPrec_B=2.0;
	float Prec = 0.000001f;
	SUB  *subj;
	char *S;
	
//	float *hostWM;
	FILE *out,*out2,*fWM,*fcoef,*fcoefsd,*fBF,*fresid;
	
	void itoa(int,char *);
//	unsigned char *get_neighbors();
	void initializeAlpha(unsigned char *,float *,float *,float *,unsigned long *);
	void initializeZ(float *,unsigned char *,unsigned char *,unsigned long *);
	void initializePhi(float *,unsigned char *,unsigned char *,float,unsigned long *);
	void updateZGPU(unsigned char *,float,unsigned long *);
	void updateZ(float *,float *,unsigned char *,unsigned char *,float, float *,float *,float *,unsigned long *);
	void updatePhiGPU(float,float *);
	void updatePhi(unsigned char *data,float *Z,float *Phi,unsigned char *msk,float beta,float *WM,float *spatCoef,float *covar,unsigned long *seed); 
	void updateAlpha(float *,float ,float ,float *,unsigned char *,float,float *,float *,float *,int,unsigned long *);
	void updateAlphaGPU(float *,float *,float *,float *,unsigned char *,float,float *,float *,float *,unsigned long *);
	void updateAlphaPrecGPU(float *,float *,float,float,float,unsigned long *);
	void updateAlphaPrec(float *,float *,float,float,float,unsigned char *,unsigned long *);
	void updateAlphaMeanGPU(float *,float,float *,float *,float *,float,unsigned long *);
	void updatefixMeanGPU(float *,float,float *,float *,float *,float,unsigned long *);
	void updateAlphaMean(float *,float *,float *,float,float *,float *,float,unsigned char *,unsigned long *);
	void compute_prbGPU(float);
	void compute_prb(float *,float *,float *,float,float *,unsigned char *);
	void updateBeta(float *,float *,float *,float *,unsigned char *,float,float,float *,float *,unsigned long *);
	void updateBetaGPU(float *,float *,float *,float *,float *,float,float,float *,float *,unsigned long *);
	void updateSpatCoefGPU(float *,float *,float *,float *,float *,float *,float *,unsigned char *,float,float *,unsigned long *);
	void updateSpatCoef(float *,float *,float *,float *,float *,float *,unsigned char *,float,float *,unsigned long *);
	void updateSpatCoefMean(float *,float *,float,float,unsigned char *,float **,float *,float *,float *,float,int,unsigned long *);
	void updateSpatCoefPrec(float *,float *,unsigned char *,unsigned long *);
	void updateSpatCoefPrecGPU(float *,unsigned long *);
	void updateSpatCoefPrecGPU_Laplace(float *,unsigned long *);
	void save_iter(float,float *,float *,float *,float *,float *,float *,float *,float *,float *);
	void restart_iter(float *,float *,float *,float *,float *,float *,float *,float *,float *,float*);
	int write_nifti_file(int NROW,int NCOL,int NDEPTH,int NTIME,char *hdr_file,char *data_file,MY_DATATYPE *data);
	double compute_predictGPU(float,unsigned char *,unsigned char *msk,float *,float *,double **Qhat,int);
	double compute_prb_DIC(double *,double *,float *,unsigned char *,unsigned char *,int);

//	void tmp(float *,float *);
	fresid = fopen("resid.dat","w");
	
 	NSIZE = (NROW+2)*(NCOL+2)*(NDEPTH+2);
	int RSIZE = NROW*NCOL*NDEPTH;
//	printf("%d %d \n",TOTVOX,NSIZE);

	if (GPU) {
		CUDA_CALL( cudaMalloc((void **)&deviceData,NSUBS*TOTVOX*sizeof(unsigned char)) );
	//	unsigned char *data2;
	//	data2 = (unsigned char *)calloc(NSUBS*TOTVOX,sizeof(unsigned char));
	//	for (int i=0;i<NSUBS*TOTVOX;i++) {
	//		if (data[i] == 1)
	//			data2[i] = 2;
	//	}
	//	CUDA_CALL( cudaMemcpy(deviceData,data2,NSUBS*TOTVOX*sizeof(unsigned char),cudaMemcpyHostToDevice) );				
		CUDA_CALL( cudaMemcpy(deviceData,data,NSUBS*TOTVOX*sizeof(unsigned char),cudaMemcpyHostToDevice) );				
	//	free(data2);
	}
	
	SpatCoefMean = (float *)calloc(NCOVAR,sizeof(float));
	SpatCoefPrec = (float *)calloc(NCOVAR*NCOVAR,sizeof(float));
	for (int i=0;i<NCOVAR;i++) {
		SpatCoefMean[i] = 0;
		SpatCoefPrec[i + i*NCOVAR] = 1;
	}

	alphaMean = (float *)calloc(NCOVAR,sizeof(float));
	if (NCOV_FIX > 0)
		fixMean = (float *)calloc(NCOV_FIX,sizeof(float));

	alphaPrec = (float *)calloc(NSUBTYPES,sizeof(float));
	for (int i = 0;i<NSUBTYPES;i++)
		alphaPrec[i] = 1.0;
	
	if (GPU) {
	 	CUDA_CALL( cudaMalloc((void **)&deviceWM,TOTVOX*sizeof(float)) );
	 	CUDA_CALL( cudaMemcpy(deviceWM,WM,TOTVOX*sizeof(float),cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaDeviceSynchronize() );
	}	

		
	subj = (SUB *)calloc(NSUBS,sizeof(SUB));
	
	Z = (float *)calloc(TOTVOX*NSUBS,sizeof(float));
	Phi = (float *)calloc(TOTVOX*NSUBS,sizeof(float));
	if (GPU) {
	//	CUDA_CALL( cudaHostAlloc((void **)&Z, TOTVOX*NSUBS*sizeof(float),cudaHostAllocDefault) );
		CUDA_CALL( cudaMalloc((void **)&deviceZ,TOTVOX*NSUBS*sizeof(float)) );
		CUDA_CALL( cudaMemset(deviceZ,0,TOTVOX*NSUBS*sizeof(float)) );
	//	CUDA_CALL( cudaHostAlloc((void **)&Phi, TOTVOX*NSUBS*sizeof(float),cudaHostAllocDefault) );
//		CUDA_CALL( cudaMalloc((void **)&devicePhi,TOTVOX*NSUBS*sizeof(float)) );
//		CUDA_CALL( cudaMemset(devicePhi,1,TOTVOX*NSUBS*sizeof(float)) );
//		CUDA_CALL( cudaMalloc((void **)&deviceResid,TOTVOX*NSUBS*sizeof(unsigned int)) );
//		CUDA_CALL( cudaMemset(deviceResid,0,TOTVOX*NSUBS*sizeof(unsigned int)) );
	}
//	else
//		Z = (float *)calloc(TOTVOX*NSUBS,sizeof(float));

/*	if (!RESTART) {
		initializeZ(Z,data,msk,seed);
//		initializePhi(Phi,data,msk,t_df,seed);
		if (GPU) {
			CUDA_CALL( cudaMemcpy(deviceZ,Z,NSUBS*TOTVOX*sizeof(float),cudaMemcpyHostToDevice) );	
//			CUDA_CALL( cudaMemcpy(devicePhi,Phi,NSUBS*TOTVOX*sizeof(float),cudaMemcpyHostToDevice) );	
		}
	}	*/
	alpha = (float *)calloc(TOTVOX*NSUBTYPES,sizeof(float));
	if (GPU) {
//		cutilSafeCall(cudaMemcpy(deviceZ,Z,NSUBS*TOTVOX*sizeof(float),cudaMemcpyHostToDevice));

//		CUDA_CALL( cudaHostAlloc((void **)&alpha, TOTVOX*NSUBTYPES*sizeof(float),cudaHostAllocDefault) );
		CUDA_CALL( cudaMalloc((void **)&deviceAlphaMean,NCOVAR*sizeof(float)) );
		CUDA_CALL( cudaMemset(deviceAlphaMean,0,NCOVAR*sizeof(float)) );
		if (NCOV_FIX > 0) {
			CUDA_CALL( cudaMalloc((void **)&deviceFixMean,NCOV_FIX*sizeof(float)) );
			CUDA_CALL( cudaMemset(deviceFixMean,0,NCOV_FIX*sizeof(float)) );
		}
	}
//	else
//		alpha = (float *)calloc(TOTVOX*NSUBTYPES,sizeof(float));

/*	if (!RESTART) {
		initializeAlpha(msk,alphaMean,alphaPrec,alpha,seed);
		if (GPU)
		       cudaMemcpy(deviceAlpha,alpha,NSUBTYPES*TOTVOX*sizeof(float),cudaMemcpyHostToDevice);
	}*/
	
//	Malpha = (double *)calloc(NSUBTYPES*RSIZE,sizeof(double));
	MWM = (double *)calloc(RSIZE,sizeof(double));
	Mprb = (double *)calloc(NSUBTYPES*RSIZE,sizeof(double));
	MCov = (double *)calloc(NCOVAR*RSIZE,sizeof(double));
	SCov = (double *)calloc(NCOVAR*RSIZE,sizeof(double));
	
	Qhat = (double **)calloc(NSUBS,sizeof(double *));
	for (int i=0;i<NSUBS;i++)
		Qhat[i] = (double *)calloc(NSUBTYPES,sizeof(double));

	spatCoef = (float *)calloc(TOTVOXp*NCOVAR,sizeof(float));
/*	for (int k=1;k<NDEPTH+1;k++) {
		for (int j=1;j<NCOL+1;j++) {
			for (int i=1;i<NROW+1;i++) {
				IDX = idx(i,j,k);
				if (msk[IDX]) 
					spatCoef[hostIdxSC[IDX]] = -25.0f;
			}
		}
	}*/
	prb = (float *)calloc(NSUBTYPES*TOTVOX,sizeof(float));
	predict = (float *)calloc(NSUBTYPES*TOTVOX,sizeof(float));
	if (GPU) {
//		cudaHostAlloc((void **)&spatCoef, TOTVOXp*NCOVAR*sizeof(float),cudaHostAllocDefault);
		CUDA_CALL( cudaMalloc((void **)&deviceSpatCoef,TOTVOXp*NCOVAR*sizeof(float)) );

//		cudaMemset(deviceSpatCoef,0,TOTVOXp*NCOVAR*sizeof(float));
		CUDA_CALL( cudaMemcpy(deviceSpatCoef,spatCoef,NCOVAR*TOTVOXp*sizeof(float),cudaMemcpyHostToDevice) );

//		CUDA_CALL( cudaHostAlloc((void **)&prb, NSUBTYPES*TOTVOX*sizeof(float),cudaHostAllocDefault) );
		CUDA_CALL( cudaMalloc((void **)&devicePrb,NSUBTYPES*TOTVOX*sizeof(float)) );
		CUDA_CALL( cudaMemset(devicePrb,0,NSUBTYPES*TOTVOX*sizeof(float)) );

//		CUDA_CALL( cudaHostAlloc((void **)&predict, NSUBTYPES*TOTVOX*sizeof(float),cudaHostAllocDefault) );
		CUDA_CALL( cudaMalloc((void **)&devicePredict,NSUBTYPES*TOTVOX*sizeof(float)) );
		CUDA_CALL( cudaMemset(devicePredict,0,NSUBTYPES*TOTVOX*sizeof(float)) );
	}
//	else {
//		prb = (float *)calloc(NSUBTYPES*TOTVOX,sizeof(float));
//	}
	
/*	if (RESTART) {
		restart_iter(&beta,fixMean,alphaMean,alphaPrec,SpatCoefMean,SpatCoefPrec,spatCoef,alpha,Z,Phi);
		printf("***RESTART VALUES***\n\n");
		double *V = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
		for (int ii=0;ii<NCOVAR;ii++)
			for (int jj=0;jj<NCOVAR;jj++)
				V[jj + ii*NCOVAR] = (double)SpatCoefPrec[jj + ii*NCOVAR];
		cholesky_decomp(V, NCOVAR);
		cholesky_invert(NCOVAR, V);
		printf("iter = %d\t",iter);
		printf("%f \n",beta);
		for (int ii=0;ii<NCOV_FIX;ii++)
			printf("%f \n",fixMean[ii]);
		for (int ii=0;ii<NSUBTYPES;ii++)
			printf("\t %10.6lf %10.6f\n",alphaMean[ii],V[ii+ii*NCOVAR]);
		printf("\n");
		for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
			printf("\t %10.6lf %10.6f\n",alphaMean[ii],V[ii+ii*NCOVAR]);
		printf("\n");
		for (int ii=0;ii<NCOVAR;ii++) {
			for (int jj=0;jj<=ii;jj++)
				printf("%6.3lf ",V[jj + ii*NCOVAR]/sqrt(V[ii + ii*NCOVAR]*V[jj + jj*NCOVAR]));
			printf("\n");
		}
		printf("\n");
		fflush(NULL);
		free(V);


		out  = fopen("parms.dat","a");
		out2  = fopen("select_parms.dat","w");
	}*/
	//else {
		out = fopen("parms.dat","w");	
		out2 = fopen("select_parms.dat","w");	
	//}
	fBF = fopen("fBF.dat","w");

	cudaEvent_t start, stop;
	float time;
	CUDA_CALL( cudaEventCreate(&start) );
	CUDA_CALL( cudaEventCreate(&stop) );
//cudaEventRecord(start,0);
	SAVE_ITER = (MAXITER - BURNIN)/10000;
	PRNtITER = 100;

	for (iter=0;iter<=MAXITER;iter++) {//printf("iter = %d\n",iter);fflush(NULL);

		
		if (GPU) {
			if (!(iter%PRNtITER))
				CUDA_CALL( cudaEventRecord(start, 0) ); 
			//for (int tj=0;tj<10;tj++)
			updateZGPU(data,beta,seed);

			if (!(iter%PRNtITER)) {
				CUDA_CALL( cudaEventRecord(stop, 0) );
				CUDA_CALL( cudaEventSynchronize(stop) );
				CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
				printf ("Time to updateZGPU kernel: %f ms\n", time);
			}
			
			if (MODEL != 1) {
				if (!(iter%PRNtITER))
					CUDA_CALL( cudaEventRecord(start, 0) ); 
				//for (int tj=0;tj<10;tj++)
				updatePhiGPU(beta,Phi);
				//updatePhi(data,Z,Phi,msk,beta,WM,spatCoef,covar,seed);			
				if (!(iter%PRNtITER)) {
					CUDA_CALL( cudaEventRecord(stop, 0) );
					CUDA_CALL( cudaEventSynchronize(stop) );
					CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
					printf ("Time to updatePhiGPU kernel: %f ms\n", time);
				}
			}
			
			if (!(iter%PRNtITER))
				CUDA_CALL( cudaEventRecord(start, 0) );
			//for (int tj=0;tj<10;tj++)
			updateBetaGPU(&beta,deviceZ,devicePhi,deviceAlphaMean,deviceWM,prior_mean_beta,prior_prec_beta,deviceSpatCoef,deviceCovar,seed);
			if (!(iter%PRNtITER)) {
				CUDA_CALL( cudaEventRecord(stop, 0) );
				CUDA_CALL( cudaEventSynchronize(stop) );
				CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
				printf ("Time for the updateBetaGPU kernel: %f ms\n", time);
			}

			if (!(iter%PRNtITER))
				CUDA_CALL( cudaEventRecord(start, 0) );
			//for (int tj=0;tj<10;tj++)
			updateSpatCoefGPU(covar,spatCoef,SpatCoefMean,SpatCoefPrec,alpha,alphaMean,Z,msk,beta,WM,seed);
			if (!(iter%PRNtITER)) {
				CUDA_CALL( cudaEventRecord(stop, 0) );
				CUDA_CALL( cudaEventSynchronize(stop) );
				CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
				printf ("Time to updateSpatCoefGPU kernel: %f ms\n", time);
			}

			if (!(iter%PRNtITER))
				CUDA_CALL( cudaEventRecord(start, 0) );
			//for (int tj=0;tj<10;tj++)
			updateSpatCoefPrecGPU(SpatCoefPrec,seed);
//			updateSpatCoefPrecGPU_Laplace(SpatCoefPrec,seed);
			if (!(iter%PRNtITER)) {
				CUDA_CALL( cudaEventRecord(stop, 0) );
				CUDA_CALL( cudaEventSynchronize(stop) );
				CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
				printf ("Time to updateSpatCoefPrecGPU kernel: %f ms\n\n", time);
			}
			//CUDA_CALL( cudaEventRecord(start, 0) );
			//for (int tj=0;tj<10;tj++)
			//	ED += compute_predictGPU(beta,data,msk,covar,predict,Qhat);
			//CUDA_CALL( cudaEventRecord(stop, 0) );
			//CUDA_CALL( cudaEventSynchronize(stop) );
			//CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
			//printf ("Time for compute_predictGPU kernel: %f ms\n\n", time/10);

			//CUDA_CALL( cudaEventRecord(start, 0) );
			//for (int tj=0;tj<10;tj++)
			//compute_prbGPU(beta);
			//CUDA_CALL( cudaEventRecord(stop, 0) );
			//CUDA_CALL( cudaEventSynchronize(stop) );
			//CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
			//printf ("Time for the compute_prbGPU kernel: %f ms\n\n", time/10);exit(0);

			if (NCOV_FIX > 0)
				updatefixMeanGPU(fixMean,Prec,deviceCovar,deviceZ,deviceSpatCoef,beta,seed);
		//	updateAlphaMeanGPU(alphaMean,Prec,deviceCovar,deviceZ,deviceSpatCoef,beta,seed);
		}
		else {	
			if (!(iter%PRNtITER))
				CUDA_CALL( cudaEventRecord(start, 0) );
	
		//	for (int tj=0;tj<10;tj++)
			updateZ(Z,alpha,data,msk,beta,WM,spatCoef,covar,seed);
			if (!(iter%PRNtITER)) {			
				CUDA_CALL( cudaEventRecord(stop, 0) );
				CUDA_CALL( cudaEventSynchronize(stop) );
				CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
				printf ("Time for the updateZCPU kernel: %f ms\n", time);
			}
		//	CUDA_CALL( cudaEventRecord(start, 0) );	
		//	for (int tj=0;tj<10;tj++)
			updateBeta(&beta,Z,alpha,WM,msk,prior_mean_beta,prior_prec_beta,spatCoef,covar,seed);
		//	CUDA_CALL( cudaEventRecord(stop, 0) );
		//	CUDA_CALL( cudaEventSynchronize(stop) );
		//	CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
		//	printf ("Time for the updateBetaCPU kernel: %f ms\n", time);

			if (!(iter%PRNtITER))		
				CUDA_CALL( cudaEventRecord(start, 0) );	
		//	for (int tj=0;tj<10;tj++)
			updateSpatCoef(covar,spatCoef,SpatCoefPrec,alpha,alphaMean,Z,msk,beta,WM,seed);
			if (!(iter%PRNtITER)) {				
				CUDA_CALL( cudaEventRecord(stop, 0) );
				CUDA_CALL( cudaEventSynchronize(stop) );
				CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
				printf ("Time for the updateSpatCoefCPU kernel: %f ms\n", time);
			}
			if (!(iter%PRNtITER))
				CUDA_CALL( cudaEventRecord(start, 0) );
		//	for (int tj=0;tj<10;tj++)
			updateSpatCoefPrec(SpatCoefPrec,spatCoef,msk,seed);
			if (!(iter%PRNtITER)) {		
				CUDA_CALL( cudaEventRecord(stop, 0) );
				CUDA_CALL( cudaEventSynchronize(stop) );
				CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
				printf ("Time for the updateSpatCoefPrecCPU kernel: %f ms\n\n", time);
			}
	
		//	CUDA_CALL( cudaEventRecord(start, 0) );
		//	for (int tj=0;tj<10;tj++)
		//		compute_prb(prb,alpha,spatCoef,beta,WM,msk);
		//	CUDA_CALL( cudaEventRecord(stop, 0) );
		//	CUDA_CALL( cudaEventSynchronize(stop) );
		///	CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
		//	printf ("Time for the compute_prbCPU kernel: %f ms\n\n", time);
		}

/*		if (!(iter%PRNtITER) && iter > 0) {
			CUDA_CALL( cudaMemcpy(spatCoef,deviceSpatCoef,NCOVAR*TOTVOXp*sizeof(float),cudaMemcpyDeviceToHost) );
			int i = 56;//57;
			int j = 77;//46;
			int k = 19;//72;
			IDX = idx(i,j,k);
	
			for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
				fprintf(out2,"%f ",spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);

			i = 62;//57;
			j = 91;//46;
			k = 49;//72;
			IDX = idx(i,j,k);
	
			for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
				fprintf(out2,"%f ",spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);

			i = 31;//57;
			j = 39;//46;
			k =  5;//72;
			IDX = idx(i,j,k);
	
			for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
				fprintf(out2,"%f ",spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);
	
		 	i = 17;//57;
			j = 42;//46;
			k = 12;//72;
			IDX = idx(i,j,k);
	
			for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
				fprintf(out2,"%f ",spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);

			i = 25;//57;
			j = 93;//46;
			k = 30;//72;
			IDX = idx(i,j,k);
	
			for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
				fprintf(out2,"%f ",spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);
			fprintf(out2,"\n");fflush(NULL);
//			int *ss;
//			ss = (int *)malloc(4*sizeof(int));
//			ss[0] = 56;
//			ss[1] = 134;
//			ss[2] = 141;
//			ss[3] = 39;
			
//			float *zs;
//			zs = (float *)malloc(sizeof(float));
//			for (int i=0;i<4;i++) {
//				float xb = 0;
//				for (int ii=0;ii<NCOVAR;ii++) {
//					xb += spatCoef[ii*TOTVOXp + hostIdxSC[IDX]]*covar[ss[i]*NCOVAR+ii];
//					fprintf(out2,"%f ",spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);
//				}
//				cudaMemcpy(zs,&(deviceZ[ss[i]*TOTVOX+hostIdx[IDX]]),sizeof(float),cudaMemcpyDeviceToHost);
//			
			//	printf("%7.4f ",zs[0] - xb);
//				fprintf(fresid,"%f ",zs[0] - xb);
//			}
//			fprintf(fresid,"\n");
//			free(zs);
//			free(ss);
			
	//		i = 59;
	//		j = 25;
	//		k = 48;
///			i = 40;
//			j = 34;
//			k = 7;
//			IDX = idx(i,j,k);
			
//			for (int ii=0;ii<NCOVAR;ii++)
//				fprintf(out2,"%f ",spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);
//			fprintf(out2,"\n");
		
			fprintf(out,"%f ",beta);
			for (int ii=0;ii<NCOV_FIX;ii++)
				fprintf(out,"%f ",fixMean[ii]);
			for (int ii=0;ii<NCOVAR;ii++)
				fprintf(out,"%f ",alphaMean[ii]/logit_factor);
			for (int ii=0;ii<NCOVAR;ii++)
				for (int jj=0;jj<=ii;jj++)
					fprintf(out,"%f ",SpatCoefPrec[jj + ii*NCOVAR]);				
			fprintf(out,"\n");
		fflush(NULL);
		}*/
		
		if (!(iter%PRNtITER)) {
/*			CUDA_CALL( cudaMemcpy(spatCoef,deviceSpatCoef,NCOVAR*TOTVOXp*sizeof(float),cudaMemcpyDeviceToHost) );
			float max = -500;
			for (int k=1;k<NDEPTH+1;k++) {
				for (int j=1;j<NCOL+1;j++) {
					for (int i=1;i<NROW+1;i++) {
						IDX = idx(i,j,k);
						if (msk[IDX]) 
							max = (max >	(double)(spatCoef[4*TOTVOXp+hostIdxSC[IDX]]/logit_factor)) ?
									max:(double)(spatCoef[4*TOTVOXp+hostIdxSC[IDX]]/logit_factor);
					}
				}
			}
			printf("max = %lf\n",max);	*/			
			
			double *V = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
			for (int ii=0;ii<NCOVAR;ii++)
				for (int jj=0;jj<NCOVAR;jj++)
					V[jj + ii*NCOVAR] = (double)SpatCoefPrec[jj + ii*NCOVAR];
			cholesky_decomp(V, NCOVAR);
			cholesky_invert(NCOVAR, V);
			printf("iter = %d\t",iter);
			printf("%f \n",beta);
			for (int ii=0;ii<NCOV_FIX;ii++)
				printf("%f \n",fixMean[ii]);
			for (int ii=0;ii<NSUBTYPES;ii++)
				printf("\t %10.6lf %10.6f\n",alphaMean[ii]/logit_factor,sqrt(V[ii+ii*NCOVAR])/logit_factor);
			printf("\n");
			for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
				printf("\t %10.6lf %10.6f\n",alphaMean[ii]/logit_factor,sqrt(V[ii+ii*NCOVAR])/logit_factor);
			printf("\n");
			for (int ii=0;ii<NCOVAR;ii++) {
				for (int jj=0;jj<=ii;jj++)
//					printf("%6.3lf ",sqrt(V[jj + ii*NCOVAR]));
					printf("%6.3lf ",V[jj + ii*NCOVAR]/sqrt(V[ii + ii*NCOVAR]*V[jj + jj*NCOVAR]));
				printf("\n");
			}
			printf("\n");
			fflush(NULL);
			free(V);
		}
		if ((iter > BURNIN) && (!(iter%SAVE_ITER))) {
//		if ((iter > 0)) {
				//  compute probability of lesion

			if (GPU) {
				if (TOTSAV == 0)
					FIRST = 1;
				else
					FIRST = 0;
				if (!(iter%PRNtITER))
					cudaEventRecord(start, 0);
				ED += compute_predictGPU(beta,data,msk,covar,predict,Qhat,FIRST);
				if (!(iter%PRNtITER)) {
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&time, start, stop);
					printf ("Time to compute_predictGPU: %f ms\n", time);
				}

				if (!(iter%PRNtITER))
					cudaEventRecord(start, 0);
				compute_prbGPU(beta);
				CUDA_CALL( cudaMemcpy(spatCoef,deviceSpatCoef,NCOVAR*TOTVOXp*sizeof(float),cudaMemcpyDeviceToHost) );
				CUDA_CALL( cudaMemcpy(prb,devicePrb,NSUBTYPES*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost) );
				if (!(iter%PRNtITER)) {
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&time, start, stop);
					printf ("Time to compute_prbGPU: %f ms\n\n", time);
				}
			}
			else {
				if (!(iter%PRNtITER))
					cudaEventRecord(start, 0);
				compute_prb(prb,alpha,spatCoef,beta,WM,msk);
				if (!(iter%PRNtITER)) {
					cudaEventRecord(stop, 0);
					cudaEventSynchronize(stop);
					cudaEventElapsedTime(&time, start, stop);
					printf ("Time to compute_prbGPU: %f ms\n\n", time);
				}
			}
			
			for (int k=1;k<NDEPTH+1;k++) {
				for (int j=1;j<NCOL+1;j++) {
					for (int i=1;i<NROW+1;i++) {
						IDX = idx(i,j,k);
						int i1=i-1; 
						int j1=j-1;
						int k1=k-1;
						int IDX2 = idx2(i1,j1,k1);
						if (msk[IDX]) {
							MWM[IDX2] += (double)beta*WM[hostIdx[IDX]];
							for (int ii=0;ii<NCOVAR;ii++) {
								double tmp = (double)(spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]/logit_factor);
								MCov[ii*RSIZE+IDX2] += tmp;
								SCov[ii*RSIZE+IDX2] += tmp*tmp;
							}
							for (int ii=0;ii<NSUBTYPES;ii++) {
						//		Malpha[ii*RSIZE+IDX2] += (double)alpha[ii*TOTVOX+hostIdx[IDX]];
								Mprb[ii*RSIZE+IDX2] += (double)prb[ii*TOTVOX+hostIdx[IDX]];
							}
						}						
					}
				}
			}
			TOTSAV++;
		}	
/*		if (!(iter%10000) && iter > 1) {
			if (GPU) {
				CUDA_CALL( cudaMemcpy(Z,deviceZ,NSUBS*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost) );
				CUDA_CALL( cudaMemcpy(Phi,devicePhi,NSUBS*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost) );
//				cudaMemcpy(alpha,deviceAlpha,NSUBTYPES*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost);
				CUDA_CALL( cudaMemcpy(spatCoef,deviceSpatCoef,NCOVAR*TOTVOXp*sizeof(float),cudaMemcpyDeviceToHost) );
			}
			save_iter(beta,fixMean,alphaMean,alphaPrec,SpatCoefMean,SpatCoefPrec,spatCoef,alpha,Z,Phi);
		}*/
	}
//			CUDA_CALL( cudaEventRecord(stop, 0) );
//			CUDA_CALL( cudaEventSynchronize(stop) );
//			CUDA_CALL( cudaEventElapsedTime(&time, start, stop) );
//			printf ("Time for the updateZGPU kernel: %f sec.\n", time/1000.0);

	fclose(fBF);
	
//	unsigned int *Resid;
//	double *ResidMap;
//	Resid = (unsigned int *)calloc(NSUBS*TOTVOX,sizeof(unsigned int)); 	
//	ResidMap = (double *)calloc(RSIZE,sizeof(double)); 	
	if (GPU) {
		CUDA_CALL( cudaMemcpy(Z,deviceZ,NSUBS*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost) );
//		CUDA_CALL( cudaMemcpy(Phi,devicePhi,NSUBS*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost) );
//		cudaMemcpy(alpha,deviceAlpha,NSUBTYPES*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost);
		CUDA_CALL( cudaMemcpy(spatCoef,deviceSpatCoef,NCOVAR*TOTVOXp*sizeof(float),cudaMemcpyDeviceToHost) );
		
//		CUDA_CALL( cudaMemcpy(Resid,deviceResid,NSUBS*TOTVOX*sizeof(unsigned int),cudaMemcpyDeviceToHost) );
	}
//	save_iter(beta,fixMean,alphaMean,alphaPrec,SpatCoefMean,SpatCoefPrec,spatCoef,alpha,Z,Phi);
/*	for (int isub=0;isub<NSUBS;isub++) {
		for (int k=1;k<NDEPTH+1;k++) {
			for (int j=1;j<NCOL+1;j++) {
				for (int i=1;i<NROW+1;i++) {
					IDX = idx(i,j,k);
					int i1 = i-1;
					int j1 = j-1;
					int k1 = k-1;
					int IDX2 = idx2(i1,j1,k1);
					if (msk[IDX]) {
						double tmp = (double)Resid[isub*TOTVOX+hostIdx[IDX]]/(double)(MAXITER-BURNIN);
						if (tmp > 0.05) {
							fprintf(fresid,"%d %d %d %d %lf\n",isub,i,j,k,tmp);
							ResidMap[IDX2]++;
						}
					}
				}
			}
		}
	}
	free(Resid);
	fclose(out);
	fclose(out2);
	fclose(fresid);*/

	char *RR = (char *)calloc(500,sizeof(char));
	S = (char *)calloc(100,sizeof(char));
//	S = strcpy(S,"ResidMap.nii");
//	int rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,ResidMap);
//	free(ResidMap);
	int rtn;
//	RR = strcpy(RR,"gzip -f ");
//	RR = strcat(RR,(const char *)S);
//	rtn = system(RR);

	printf("TOTSAV = %d\n",TOTSAV);

	out = fopen("Qhat.dat","w");	
	for (int isub=0;isub<NSUBS;isub++) {
		fprintf(out,"%d ",isub);
		double max = -DBL_MAX;
		for (int i=0;i<NSUBTYPES;i++) {
			max = (max < Qhat[isub][i]) ? Qhat[isub][i]:max;
		}
		double qh = exp(Qhat[isub][0] - max);
		for (int i=1;i<NSUBTYPES;i++)
			qh += exp(Qhat[isub][i] - max);
		qh = log(qh) + max;
		for (int i=0;i<NSUBTYPES;i++) {
			fprintf(out,"%.6lf ",exp(Qhat[isub][i]-qh));
		}
		max =-DBL_MAX;
		int predtype = -1;
		for (int i=0;i<NSUBTYPES;i++) {
			if (max <= Qhat[isub][i])	{
				max = Qhat[isub][i];
				predtype = i;
			}
		}
		fprintf(out,"%d ",predtype);
		int truetype = 0;
		for (int i=0;i<NSUBTYPES;i++) {
			if (covar[i*NSUBS + isub]) {
				truetype = i;
				break;
			}
		}
		fprintf(out,"%d\n",truetype);fflush(NULL);
	}			
	fclose(out);

	out = fopen("Qhat2.dat","w");	
	for (int isub=0;isub<NSUBS;isub++) {
		fprintf(out,"%d ",isub);
		double max = -DBL_MAX;
		for (int i=0;i<NSUBTYPES;i++) {
			Qhat[isub][i] = grp_prior[i]*Qhat[isub][i];
			max = (max < Qhat[isub][i]) ? Qhat[isub][i]:max;
		}
		double qh = exp(Qhat[isub][0] - max);
		for (int i=1;i<NSUBTYPES;i++)
			qh += exp(Qhat[isub][i] - max);
		qh = log(qh) + max;
		for (int i=0;i<NSUBTYPES;i++) {
			fprintf(out,"%.6lf ",exp(Qhat[isub][i]-qh));
		}
		max =-DBL_MAX;
		int predtype = -1;
		for (int i=0;i<NSUBTYPES;i++) {
			if (max <= Qhat[isub][i])	{
				max = Qhat[isub][i];
				predtype = i;
			}
		}
		fprintf(out,"%d ",predtype);
		int truetype = 0;
		for (int i=0;i<NSUBTYPES;i++) {
			if (covar[i*NSUBS + isub]) {
				truetype = i;
				break;
			}
		}
		fprintf(out,"%d\n",truetype);fflush(NULL);
	}
	fclose(out);

//	fWM = fopen("bWM.dat","w");

	char *char_ii;
	char_ii = (char *)calloc(3,sizeof(char));
//	S = (char *)calloc(100,sizeof(char));
	

	for (int k=0;k<NDEPTH;k++) {
		for (int j=0;j<NCOL;j++) {
			for (int i=0;i<NROW;i++) {
				IDX = idx2(i,j,k);
				MWM[IDX] /= (double)TOTSAV;
//				fprintf(fWM,"%f ",MWM[IDX]);
			}
		}	
	}
	
/*	for (int k=1;k<NDEPTH+1;k++) {
		for (int j=1;j<NCOL+1;j++) {
			for (int i=1;i<NROW+1;i++) {
				IDX = idx(i,j,k);
				int i1=i-1; 
				int j1=j-1;
				int k1=k-1;
				int IDX2 = idx2(i1,j1,k1);
				if (!msk[IDX]) {
					MWM[IDX2] = NAN;
					for (int ii=0;ii<NCOVAR;ii++) {
						MCov[ii*RSIZE+IDX2] = NAN;
						SCov[ii*RSIZE+IDX2] = NAN;
					}
					for (int ii=0;ii<NSUBTYPES;ii++) {
						Malpha[ii*RSIZE+IDX2] = NAN;
						Mprb[ii*RSIZE+IDX2] = NAN;
					}
				}						
			}
		}
	}*/

//	fclose(fWM);
	S = strcpy(S,"bWM.nii");
	rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,MWM);

	RR = strcpy(RR,"gzip -f ");
	RR = strcat(RR,(const char *)S);
	rtn = system(RR);

/*	for (int ii=0;ii<NSUBTYPES;ii++) {	
		for (int k=0;k<NDEPTH;k++) {
			for (int j=0;j<NCOL;j++) {
				for (int i=0;i<NROW;i++) {
				    IDX = idx2(i,j,k);
					double tmp = MCov[ii*RSIZE+IDX]/(double)TOTSAV;
					MCov[ii*RSIZE+IDX] = tmp;
					tmp = SCov[ii*RSIZE+IDX] - (double)TOTSAV*tmp*tmp;
					tmp /= (double)(TOTSAV - 1);
					SCov[ii*RSIZE+IDX] = tmp;
				}
			}
		}

		
		S = strcpy(S,"spatRE");
		itoa(ii,char_ii);
		S = strcat(S,char_ii);
		S = strcat(S,".nii");
		int rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,&MCov[ii*RSIZE]);

		S = strcpy(S,"spatRE_Var");
		itoa(ii,char_ii);
		S = strcat(S,char_ii);
		S = strcat(S,".nii");
		rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,&SCov[ii*RSIZE]);

	}*/


	for (int ii=0;ii<NCOVAR;ii++) {	
		double *standCoef = (double *)calloc(RSIZE,sizeof(double));
		for (int k=0;k<NDEPTH;k++) {
			for (int j=0;j<NCOL;j++) {
				for (int i=0;i<NROW;i++) {
					IDX = idx2(i,j,k);
					double tmp = MCov[ii*RSIZE+IDX]/(double)TOTSAV;
					MCov[ii*RSIZE+IDX] = tmp;
					tmp = SCov[ii*RSIZE+IDX] - (double)TOTSAV*tmp*tmp;
					tmp /= (double)(TOTSAV - 1);
					SCov[ii*RSIZE+IDX] = tmp;
//					int ii = i+1;
//					int jj = j+1;
//					int kk = j+1;
//					int IDX2 = idx(ii,jj,kk);
					if (tmp>0)
						standCoef[IDX] = MCov[ii*RSIZE+IDX]/sqrt(tmp);
				}
			}
		}

		S = strcpy(S,"spatCoef_");
		S = strcat(S,SS[ii]);
		S = strcat(S,".nii");
		int rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,&MCov[ii*RSIZE]);

		RR = strcpy(RR,"gzip -f ");
		RR = strcat(RR,(const char *)S);
		rtn = system(RR);

		S = strcpy(S,"spatCoef_");
		S = strcat(S,SS[ii]);
		S = strcat(S,".Var.nii");
		rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,&SCov[ii*RSIZE]);

		RR = strcpy(RR,"gzip -f ");
		RR = strcat(RR,(const char *)S);
		rtn = system(RR);

/*		for (int k=0;k<NDEPTH;k++) {
			for (int j=0;j<NCOL;j++) {
				for (int i=0;i<NROW;i++) {
					int ii = i + 1;
					int jj = j + 1;
					int kk = k + 1;
					IDX = idx(ii,jj,kk);
					int IDX2 = idx2(i,j,k);
					if (msk[IDX])
						SCov[ii*RSIZE + IDX2] = MCov[ii*RSIZE + IDX2]/sqrt(SCov[ii*RSIZE + IDX2]);					
				}
			}
		}*/

		S = strcpy(S,"standCoef_");
		S = strcat(S,SS[ii]);
		S = strcat(S,".nii");
		rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,standCoef);

		RR = strcpy(RR,"gzip -f ");
		RR = strcat(RR,(const char *)S);
		rtn = system(RR);

		free(standCoef);
	}
	
	for (int ii=0;ii<NSUBTYPES;ii++) {	
		for (int k=0;k<NDEPTH;k++) {
			for (int j=0;j<NCOL;j++) {
				for (int i=0;i<NROW;i++) {
					IDX = idx2(i,j,k);
					double tmp = Mprb[ii*RSIZE+IDX]/(double)TOTSAV;
					Mprb[ii*RSIZE+IDX] = tmp;
					
//					tmp = Malpha[ii*RSIZE+IDX]/(double)TOTSAV;
//					Malpha[ii*RSIZE+IDX] = tmp;
				}
			}
		}

		S = strcpy(S,"prb_");
		S = strcat(S,SS[ii]);
		S = strcat(S,".nii");
		int rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,&Mprb[ii*RSIZE]);

		RR = strcpy(RR,"gzip -f ");
		RR = strcat(RR,(const char *)S);
		rtn = system(RR);

/*		S = strcpy(S,"RE");
		itoa(ii,char_ii);
		S = strcat(S,char_ii);
		S = strcat(S,".nii");
		rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,&Malpha[ii*RSIZE]);*/
	}

	FILE *fdic;
	fdic = fopen("DIC.dat","w");
	ED /= (double)TOTSAV;
	ED *= -2.0;
	double DE = compute_prb_DIC(MCov,MWM,covar,data,msk,RSIZE);
	double PD = ED - DE;
	printf("DE = %lf ED = %lf PD = %lf DIC = %lf\n",DE,ED,PD,DE + 2*PD);
	fprintf(fdic,"DE = %lf ED = %lf PD = %lf DIC = %lf\n",DE,ED,PD,DE + 2*PD);
	
	fclose(fdic);
	free(S);
	free(RR);
	free(char_ii);
	
	free(alphaMean);
	free(alphaPrec);

	free(MCov);
	free(SCov);
	
	free(MWM);


	if (GPU) {
		CUDA_CALL( cudaFree(deviceData) );
		CUDA_CALL( cudaFree(deviceZ) );
		CUDA_CALL( cudaFree(devicePrb) );
		CUDA_CALL( cudaFree(devicePredict) );
//		CUDA_CALL( cudaFree(devicePhi) );
	}
	free(prb);
	free(predict);		
	free(spatCoef);	
	free(SpatCoefMean);
	free(SpatCoefPrec);
	free(Mprb);
	free(Z);
	free(Phi);
	free(alpha);

}


void updateAlphaMean(float *alphaMean,float *Z,float *spatCoef,float beta,float *WM,float *covar,float Prec,unsigned char *msk,unsigned long *seed)
{
	double *mean,*V,*tmpmean;
	double sum,tmp;
	FILE *varout;

	mean = (double *)calloc(NCOVAR,sizeof(double));
	tmpmean = (double *)calloc(NCOVAR,sizeof(double));
	V = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));

	for (int i=0;i<NCOVAR;i++) {
		for (int j=0;j<NCOVAR;j++) {
			V[j + i*NCOVAR] = (double)TOTVOX*(double)XXprime[j + i*NCOVAR];
//			if (i==j)
//				V[j + j*NCOVAR] += (double)Prec;
		}
	}


	cholesky_decomp(V, NCOVAR);
	cholesky_invert(NCOVAR, V);

	for (int isub=0;isub<NSUBS;isub++) {
		sum = 0;
		for (int k=1;k<NDEPTH+1;k++) {
			for (int j=1;j<NCOL+1;j++) {
				for (int i=1;i<NROW+1;i++) {
					int IDX = idx(i,j,k);
					if (msk[IDX]) {
						tmp = (double)Z[isub*TOTVOX+hostIdx[IDX]] - (double)beta*(double)WM[hostIdx[IDX]];
						for (int ii=0;ii<NCOVAR;ii++)
							tmp -= (double)spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]*(double)covar[ii*NSUBS+isub]; 
						sum += tmp;
					}
				}
			}
		}
			
		for (int ii=0;ii<NCOVAR;ii++)
			tmpmean[ii] += sum*(double)covar[ii*NSUBS+isub];		
	}

	for (int i=0;i<NCOVAR;i++) {
		for (int j=0;j<NCOVAR;j++)
			mean[i] += V[j + i*NCOVAR]*tmpmean[j];
	}

	rmvnorm(tmpmean,V,NCOVAR,mean,seed,0);
		
	for (int i=0;i<NCOVAR;i++)
		alphaMean[i] = (float)tmpmean[i];
	
	CUDA_CALL( cudaMemcpy(deviceAlphaMean,alphaMean,NCOVAR*sizeof(float),cudaMemcpyHostToDevice) );

	free(tmpmean);
	free(mean);
	free(V);
}

void updateAlphaPrec(float *alphaPrec,float *alpha,float alphaMean,float prior_alphaPrec_A,float prior_alphaPrec_B,unsigned char *msk,unsigned long *seed)
{
	float Alpha,beta,tmp;
	float c = 0;
	float y,t;
//	float temp = 0;
	
	Alpha = 0.5*TOTVOX + prior_alphaPrec_A;
	beta = 0;

	for (int k=1;k<NDEPTH+1;k++) {
		for (int j=1;j<NCOL+1;j++) {
			for (int i=1;i<NROW+1;i++) {
				int IDX = idx(i,j,k);
				if (msk[IDX]) {
					tmp = (alpha[hostIdx[IDX]] - alphaMean);
					y = tmp*tmp - c;	
					t = beta + y;
					c = (t - beta) - y;
					beta = t;
//					beta += tmp*tmp;	
				}
			}
		}
	}
	beta = 0.5*beta + prior_alphaPrec_B;
	*alphaPrec = (float)rgamma((double)Alpha,(double)beta,seed);
	
}


void updateBeta(float *beta,float *Z,float *alpha,float *WM,unsigned char *msk,float prior_mean_beta,float prior_prec_beta,float *spatCoef,float *covar,unsigned long *seed)
{
	float mean=0,var=0,tmp;
	float c = 0,cm=0;
	float y,t,ym,tm;
//	float temp = 0,tempm=0;
		
	for (int k=1;k<NDEPTH+1;k++) {
		for (int j=1;j<NCOL+1;j++) {
			for (int i=1;i<NROW+1;i++) {
				int IDX = idx(i,j,k);
				if (msk[IDX]) {
					y = WM[hostIdx[IDX]]*WM[hostIdx[IDX]] - c;	
					t = var + y;
					c = (t - var) - y;
					var = t;
//					var += (double)WM[IDX]*(double)WM[IDX];
					tmp = 0;
					for (int isub=0;isub<NSUBS;isub++) {
						tmp += (Z[isub*TOTVOX+hostIdx[IDX]]);
						for (int ii=0;ii<NSUBTYPES;ii++)
							tmp -= (spatCoef[ii*TOTVOXp+hostIdxSC[IDX]])*covar[ii*NSUBS+isub];
						for (int ii=NSUBTYPES;ii<NCOVAR;ii++)
							tmp -= spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]*covar[ii*NSUBS+isub]; 
					}
					tmp *= WM[hostIdx[IDX]];
					ym = tmp - cm;
					tm = mean + ym;
					cm = (tm - mean) - ym;
					mean = tm;
	//				mean += tmp;
				}
			}
		}
	}
	var *= NSUBS;
	var += prior_prec_beta;
	var = 1./var;
	mean = var*(mean + prior_mean_beta*prior_prec_beta);
	tmp = rnorm((double)mean,sqrt((double)var),seed);
	*beta = (float)tmp;
}

void updateAlpha(float *alpha,float alphaMean,float alphaPrec,float *Z,unsigned char *msk,float beta,float *WM,float *spatCoef,float *covar,int grp,unsigned long *seed)
{
	float mean=0,var=0,N=0;
	
	for (int k=1;k<NDEPTH+1;k++) {
		for (int j=1;j<NCOL+1;j++) {
			for (int i=1;i<NROW+1;i++) {
				int IDX = idx(i,j,k);
				if (msk[IDX]) {
					mean = alphaMean*alphaPrec;
					N=0;
					for (int isub=0;isub<NSUBS;isub++) {
						if (covar[grp*NSUBS+isub]) {
							N++;
							mean += (Z[isub*TOTVOX+hostIdx[IDX]] - beta*WM[hostIdx[IDX]]);
//							mean -= spatCoef[grp][i][j][k]*covar[grp*NSUBS+isub]; 
							for (int ii=0;ii<NCOVAR;ii++)
								mean -= spatCoef[ii*TOTVOXp+hostIdxSC[IDX]]*covar[ii*NSUBS+isub]; 
						}
					}
					var = 1./(N + alphaPrec);
					mean *= var;
					double tmp = rnorm((double)mean,sqrt((double)var),seed);			
					alpha[hostIdx[IDX]] = (float)tmp;
				}
			}
		}
	}
}



void initializeAlpha(unsigned char *msk,float *alphaMean,float *alphaPrec,float *alpha,unsigned long *seed){
	
	for (int ii=0;ii<NSUBTYPES;ii++) {

/*		if (RESTART) {
			FILE *in;
			float dta;
			in = fopen("last_re.dat","r");
			for (int k=1;k<NDEPTH+1;k++) {
				for (int j=1;j<NCOL+1;j++) {
					for (int i=1;i<NROW+1;i++) {
						int IDX = idx(i,j,k);
						rtn = fscanf(in,"%f ",&dta);
						if (msk[IDX])
							alpha[ii*TOTVOX+hostIdx[IDX]] = dta;
					}
				}
			}
			fclose(in);
		}
		else {*/
//			FILE *fout;
//			fout = fopen("Af.dat","w");
			for (int k=1;k<NDEPTH+1;k++) {
				for (int j=1;j<NCOL+1;j++) {
					for (int i=1;i<NROW+1;i++) {
						int IDX = idx(i,j,k);
						if (msk[IDX]) {
							alpha[ii*TOTVOX+hostIdx[IDX]] = rnorm(alphaMean[ii],sqrt(1./alphaPrec[ii]),seed);
//							fprintf(fout,"%.6f\n",alpha[IDX]);
						}
					}
				}
			}
//			fclose(fout);
		}
//	}
}

void initializeZ(float *Z,unsigned char *data,unsigned char *msk,unsigned long *seed){
	
		
		for (int k=1;k<NDEPTH+1;k++) {
			for (int j=1;j<NCOL+1;j++) {
				for (int i=1;i<NROW+1;i++) {
					int IDX = idx(i,j,k);
					if (msk[IDX]) {
						for (int isub=0;isub<NSUBS;isub++) {
							if (data[isub*TOTVOX+hostIdx[IDX]]==1) {
								Z[isub*TOTVOX+hostIdx[IDX]] = (float)truncNorm2(5.,.01,0.,50.,seed);
							}
							else {
								Z[isub*TOTVOX+hostIdx[IDX]] = (float)truncNorm2(-5.,.01,-50.,0.,seed);
							}
						}
					}
				}
			}
		}
}

void initializePhi(float *Phi,unsigned char *data,unsigned char *msk,float df,unsigned long *seed){
	
		
		for (int k=1;k<NDEPTH+1;k++) {
			for (int j=1;j<NCOL+1;j++) {
				for (int i=1;i<NROW+1;i++) {
					int IDX = idx(i,j,k);
					if (msk[IDX]) {
						for (int isub=0;isub<NSUBS;isub++) {
							Phi[isub*TOTVOX+hostIdx[IDX]] = (float)rgamma(0.5*double(df),0.5*double(df),seed);
							Phi[isub*TOTVOX+hostIdx[IDX]] = (float)1.0;
						}
					}
				}
			}
		}
}


unsigned char *get_neighbors()
{
	unsigned char *nbrs;
	
	nbrs = (unsigned char *)calloc(NSIZE,sizeof(unsigned char));
		
	return nbrs;
}



void save_iter(float beta,float *fixMean,float *alphaMean,float *alphaPrec,float *SpatCoefMean,float *SpatCoefPrec,float *spatCoef,float *alpha,float *Z,float *Phi)
{
	FILE *out;
	size_t rtn;
	
	out  = fopen("last_parms.dat","w");

	fprintf(out,"%f ",beta);
	for (int ii=0;ii<NCOV_FIX;ii++)
		fprintf(out,"%f ",fixMean[ii]);
	for (int ii=0;ii<NCOVAR;ii++)
		fprintf(out,"%f ",alphaMean[ii]);
	for (int ii=0;ii<NCOVAR;ii++)
		for (int jj=0;jj<NCOVAR;jj++)
			fprintf(out,"%f ",SpatCoefPrec[jj + ii*NCOVAR]);
	fprintf(out,"\n");

	fclose(out);

	out = fopen("last_spatCoef.dat","w");
	rtn = fwrite(spatCoef,sizeof(float),TOTVOXp*NCOVAR,out);
	fclose(out);


	out = fopen("last_Z.dat","w");
	rtn = fwrite(Z,sizeof(float),NSUBS*TOTVOX,out);
	fclose(out);

	out = fopen("last_Phi.dat","w");
	rtn = fwrite(Z,sizeof(float),NSUBS*TOTVOX,out);
	fclose(out);
}

void restart_iter(float *beta,float *fixMean,float *alphaMean,float *alphaPrec,float *SpatCoefMean,float *SpatCoefPrec,float *spatCoef,float *alpha,float *Z,float *Phi)
{
	size_t rtn;
	FILE *in;
	float dta;
	in = fopen("last_spatCoef.dat","r");

	rtn = fread(spatCoef,sizeof(float),NCOVAR*TOTVOXp,in);
	fclose(in);
	
	in = fopen("last_Z.dat","r");
	rtn = fread(Z,sizeof(float),NSUBS*TOTVOX,in);
	
	in = fopen("last_Phi.dat","r");
	rtn = fread(Phi,sizeof(float),NSUBS*TOTVOX,in);
	
	fclose(in);

	if (GPU) {
		CUDA_CALL( cudaMemcpy(deviceSpatCoef,spatCoef,NCOVAR*TOTVOXp*sizeof(float),cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(deviceZ,Z,NSUBTYPES*TOTVOX*sizeof(float),cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(devicePhi,Phi,NSUBTYPES*TOTVOX*sizeof(float),cudaMemcpyHostToDevice) );
	}
		
	printf("RESTART = %d BURNIN = %d\n",RESTART,BURNIN);

	
	in = fopen("last_parms.dat","r");
	rtn = fscanf(in,"%f ",&dta);
	*beta = dta;
	for (int ii=0;ii<NCOV_FIX;ii++)
		rtn = fscanf(in,"%f ",&fixMean[ii]);
	for (int ii=0;ii<NCOVAR;ii++)
		rtn = fscanf(in,"%f ",&alphaMean[ii]);
	for (int ii=0;ii<NCOVAR;ii++)
		for (int jj=0;jj<NCOVAR;jj++)
			rtn = fscanf(in,"%f ",&SpatCoefPrec[jj + ii*NCOVAR]);
	fclose(in);
}

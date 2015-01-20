/*
 *  covarGPU.cu
 *  BinCAR
 *
 *  Created by Timothy Johnson on 5/22/12.
 *  Copyright 2012 University of Michigan. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "binCAR.h"
#include "randgen.h"
#include "cholesky.h"
#include <float.h>

cudaError_t err;

extern float logit_factor;
extern float t_df;
extern int MODEL;
extern int NSUBS;
extern int NROW;
extern int NCOL;
extern int NDEPTH;
extern int TOTVOX;
extern int TOTVOXp;
extern int NCOVAR;
extern int NCOV_FIX;
extern int NSUBTYPES;
//extern int NSIZE;
extern int iter;
extern INDEX *INDX;
extern float *deviceCovar;
extern float *hostCovar;
extern float *deviceCov_Fix;
extern float *hostCov_Fix;
extern float *deviceAlphaMean;
extern float *deviceFixMean;
extern float *deviceZ;
extern float *devicePhi;
extern float *deviceWM;
extern unsigned int *deviceResid;
extern int *deviceIdx;
extern int *deviceIdxSC;
extern float *devicePrb;
extern float *devicePredict;
extern float *XXprime;
extern float *XXprime_Fix;
extern int *hostIdx;
extern int *hostIdxSC;
//extern float *deviceChiSqHist;
//extern int   ChiSqHist_N;

const int NN=256;
const int BPG=32;
extern curandState *devStates;
extern unsigned char *deviceData;
//extern short *deviceData;
extern float *deviceSpatCoef;
__constant__ float dPrec[20*20];
__constant__ float dtmpvar[20*20];
__constant__ float dalphaMean[20];
__constant__ float dalphaPrec[20];

const int TPB=192;
const int TPB2=96;

#define idx(i,j,k) (i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k)

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) {printf("\nCUDA Error: %s (err_num=%d) \n",cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}

__global__ void setup_kernel ( curandState * state, unsigned long long devseed, const int N )
{
/* Each thread gets same seed , a different sequence number, no offset */

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (idx < N) {
		curand_init (devseed , idx , 0 , & state [ idx ]) ;
   		idx += blockDim.x *gridDim.x;
	}
}


__global__ void reduce(float *in,float *out,const int N)
{
	__shared__ float cache[NN];
	int cacheIndex = threadIdx.x;
	int ivox = threadIdx.x + blockIdx.x * blockDim.x;
	
	float c = 0;
	float y,t;
	float temp = 0;
	while (ivox < N) {
		y = in[ivox] - c;
		t = temp + y;
		c = (t - temp) - y;
		temp = t;
		ivox += blockDim.x * gridDim.x;
	}
	
	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i = i >> 1;	
	}
	
	if (cacheIndex == 0)
		out[blockIdx.x] = cache[0];
}


__device__ int cholesky_decompGPU(float *A, int num_col)
{
	int k,i,j;
	
	for (k=0;k<num_col;k++) {
		if (A[k + k*num_col] <= 0) {
			return 0;
		}
		
		A[k + k*num_col] = sqrtf(A[k + k*num_col]);
		
		for (j=k+1;j<num_col;j++)
			A[k + j*num_col] /= A[k + k*num_col];
		
		for (j=k+1;j<num_col;j++)
			for (i=j;i<num_col;i++)
				A[j+i*num_col] = A[j + i*num_col] - A[k + i*num_col] * A[k + j*num_col];
	}
	return 1;
}

__device__ void cholesky_invertGPU(int len,float *H)
{
	/* takes G from GG' = A and computes A inverse */
	int i,j,k;
	float temp,INV[20*20];
	
	for (i=0;i<20*20;i++)
		INV[i] = 0;
		
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
}

__device__ int rmvnormGPU(float *result,float *A,int size_A,float *mean,curandState *state,int flag)
{
	int i,j;
	float runiv[20],tmp=0;
	

	j = 1;
	if (!flag)
		j = cholesky_decompGPU(A,size_A);
	if (j) {
		for (i=0;i<size_A;i++) 
			runiv[i] = curand_normal(state);
		for (i=0;i<size_A;i++) {
			tmp = 0;
			for (j=0;j<=i;j++)
				tmp += A[j +i*size_A]*runiv[j];
			tmp += mean[i];
			result[i] = tmp;
		}
		return 1;
	}
	else {
		return 0;
	}
}

__global__ void Spat_Coef_for_Spat_Covar(float *covar,float *alpha,float *Z,float *WM,float *SpatCoef,float *beta,unsigned char *nbrs,int *vox,const int NSUBS,const int NROW,const int NCOL,const int NSUBTYPES,const int NCOVAR,int NCOVAR2,const int N,const int TOTVOXp,const int TOTVOX,curandState *state,int *dIdx,int *dIdxSC)
{
	extern __shared__ float shared[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int localsize = blockDim.x;  
	int localid = threadIdx.x; 
	
	float sumV[20];
	float mean[20];
	float var[20*20];
	float tmp=0;
	float tmp2[20];
	float alph[20];
	
	int voxel=0,IDX=0,IDXSC=0;
	int current_covar=0;
	curandState localState;

	if (idx < N) {  
		voxel =  vox[idx];
		IDX   = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
		
		for (int ii=0;ii<NCOVAR2;ii++) {
			mean[ii] = 0;
			//alph[ii] = alpha[ii];
		}
	}
	
	for (int isub = 0; isub < NSUBS; isub += localsize ) {
		if ((isub+localsize) > NSUBS)
		localsize = NSUBS - isub;
				
		if ((isub+localid) < NSUBS) {
			for (int j=0;j<NCOVAR;j++) 
//				shared[localid + j*localsize] = covar[(isub+localid)*NCOVAR + j]*SpatCoef[j*TOTVOXp + IDXSC];
				shared[localid + j*localsize] = covar[j*NSUBS + (isub+localid)]*SpatCoef[j*TOTVOXp + IDXSC];
		}
		__syncthreads();
				
		for (int j=0;j<localsize;j++) {
				
			tmp = Z[(isub+j)*TOTVOX+IDX];
			for (int ii=0;ii<NCOVAR;ii++) {
				tmp -= shared[j + ii*localsize]; 
			}
			for (current_covar=0;current_covar<NCOVAR2;current_covar++) {	
				mean[current_covar] += WM[current_covar*TOTVOXp + IDXSC]*tmp;
			}
		}		
		__syncthreads();
	}

	if (idx < N) {
		localState = state[idx];
		for (current_covar=0;current_covar<NCOVAR2;current_covar++) {	
			sumV[current_covar]  = beta[(current_covar*TOTVOXp+dIdxSC[voxel-1])]
			      				 + beta[(current_covar*TOTVOXp+dIdxSC[voxel+1])]
			      				 + beta[(current_covar*TOTVOXp+dIdxSC[voxel-(NROW+2)])]
				  				 + beta[(current_covar*TOTVOXp+dIdxSC[voxel+(NROW+2)])]
			      				 + beta[(current_covar*TOTVOXp+dIdxSC[voxel-(NROW+2)*(NCOL+2)])]
				  				 + beta[(current_covar*TOTVOXp+dIdxSC[voxel+(NROW+2)*(NCOL+2)])];
		}
/**** fix up dtmpvar and dPrec matrices *****/
		for (int ii=0;ii<NCOVAR2;ii++) {
			for (int jj=0;jj<NCOVAR2;jj++) {	
				mean[ii] += dPrec[jj + ii*NCOVAR2]*sumV[jj];
				var[jj + ii*NCOVAR2] = dtmpvar[jj + ii*NCOVAR2] + nbrs[idx]*dPrec[jj + ii*NCOVAR2];
			}
		}
		// decompose and invert var 
		cholesky_decompGPU(var,NCOVAR2);
		cholesky_invertGPU(NCOVAR2, var);
		
		for (int ii=0;ii<NCOVAR2;ii++) {
			tmp2[ii] = 0;
			for (int jj=0;jj<NCOVAR2;jj++)
				tmp2[ii] += var[jj + ii*NCOVAR2]*mean[jj];
		}			
		// draw MVN and with mean tmp2 and variance var --- result in mean;
		rmvnormGPU(mean,var,NCOVAR2,tmp2,&localState,0);
			
		for (int ii=0;ii<NCOVAR2;ii++) 
			beta[(ii*TOTVOXp+IDXSC)] = mean[ii];
	
		state[idx] = localState;
	}
}

__global__ void Spat_Coef_Probit(float *covar,float *covF,float *fix,float *alpha,float *Z,float *WM,float *SpatCoef,float beta,unsigned char *nbrs,int *vox,const int NSUBS,const int NROW,const int NCOL,const int NSUBTYPES,const int NCOVAR,int NCOV_FIX,const int N,const int TOTVOXp,const int TOTVOX,curandState *state,int *dIdx,int *dIdxSC)
{
	extern __shared__ float shared[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int localsize = blockDim.x;  
	int localid = threadIdx.x; 
	
	float sumV[20];
	float mean[20];
	float var[20*20];
	float tmp=0;
	float tmp2[20];
	float alph[20];
	float fixed[20];
	
	int voxel=0,IDX=0,IDXSC=0;
	float bwhtmat=0;
	int current_covar=0;
	curandState localState;

	if (idx < N) {  
		voxel =  vox[idx];
		IDX   = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
		bwhtmat = beta*WM[IDX];
		
		for (int ii=0;ii<NCOVAR;ii++) {
			mean[ii] = 0;
			//alph[ii] = alpha[ii];
		}
	//	for (int ii=0;ii<NCOV_FIX;ii++)
	//		fixed[ii] =  fix[ii]; 
	}
	
	for (int isub = 0; isub < NSUBS; isub += localsize ) {
		if ((isub+localsize) > NSUBS)
		 	localsize = NSUBS - isub;
				
		if ((isub+localid) < NSUBS) {
			for (int j=0;j<NCOVAR;j++) 
				shared[localid + j*localsize] = covar[j*NSUBS + (isub+localid)];
		}
		__syncthreads();
				
		for (int j=0;j<localsize;j++) {
				
			float ZZ = Z[(isub+j)*TOTVOX+IDX];
			tmp = ZZ - bwhtmat;
		//	for (int ii=0;ii<NCOVAR;ii++) {
		//		tmp -= alph[ii] *shared[j + ii*localsize]; 
		//	}
		//	for (int ii=0;ii<NCOV_FIX;ii++) {
		//		tmp -= fixed[ii] * covF[(isub+j)*NCOV_FIX + ii]; 
		//	}
			for (current_covar=0;current_covar<NCOVAR;current_covar++) {	
				mean[current_covar] += shared[j + current_covar*localsize]*tmp;
			}
		}		
		__syncthreads();
	}

	if (idx < N) {
		localState = state[idx];
		for (current_covar=0;current_covar<NCOVAR;current_covar++) {	
			sumV[current_covar]  = SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel-1])]
			      				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel+1])]
			      				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel-(NROW+2)])]
				  				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel+(NROW+2)])]
			      				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel-(NROW+2)*(NCOL+2)])]
				  				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel+(NROW+2)*(NCOL+2)])];
		}
// TIME TO HERE IS 38.2 ms
/**** fix up dtmpvar and dPrec matrices *****/
		for (int ii=0;ii<NCOVAR;ii++) {
			for (int jj=0;jj<NCOVAR;jj++) {	
				mean[ii] += dPrec[jj + ii*NCOVAR]*sumV[jj];
				var[jj + ii*NCOVAR] = dtmpvar[jj + ii*NCOVAR] + nbrs[idx]*dPrec[jj + ii*NCOVAR];
			}
		}
// TIME TO HERE IS 39.8  ms	
	// decompose and invert var 
		cholesky_decompGPU(var,NCOVAR);
// TIME TO HERE IS  59.2 ms	
		cholesky_invertGPU(NCOVAR, var);
// TIME TO HERE IS  127.8 ms	
		
		for (int ii=0;ii<NCOVAR;ii++) {
			tmp2[ii] = 0;
			for (int jj=0;jj<NCOVAR;jj++)
				tmp2[ii] += var[jj + ii*NCOVAR]*mean[jj];
		}			
// TIME TO HERE IS  128.1 ms	
		// draw MVN and with mean tmp2 and variance var --- result in mean;
		rmvnormGPU(mean,var,NCOVAR,tmp2,&localState,0);
// TIME TO HERE IS 151.9  ms	
			
		for (int ii=0;ii<NCOVAR;ii++) 
			SpatCoef[(ii*TOTVOXp+IDXSC)] = mean[ii];
	
		state[idx] = localState;
	}
// TIME TO HERE IS 150.83   ms	

}

__global__ void Spat_Coef(float *covar,float *covF,float *fix,float *alpha,float *Z,float *Phi,float *WM,float *SpatCoef,float beta,unsigned char *nbrs,int *vox,const int NSUBS,const int NROW,const int NCOL,const int NSUBTYPES,const int NCOVAR,int NCOV_FIX,const int N,const int TOTVOXp,const int TOTVOX,curandState *state,int *dIdx,int *dIdxSC)
{
	extern __shared__ float shared[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int localsize = blockDim.x;  
	int localid = threadIdx.x; 
	
	float sumV[20];
	float mean[20];
	float var[20*20];
	float tmp=0;
	float tmp2[20];
	float alph[20];
	float fixed[20];
	
	int voxel=0,IDX=0,IDXSC=0;
	float bwhtmat=0;
	int current_covar=0;
	curandState localState;

	if (idx < N) {  
		voxel =  vox[idx];
		IDX   = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
		bwhtmat = beta*WM[IDX];
		
		for (int ii=0;ii<NCOVAR;ii++) {
			mean[ii] = 0;
			//alph[ii] = alpha[ii];
			for (int jj=0;jj<NCOVAR;jj++)
				var[jj + ii*NCOVAR] = 0;
		}
		for (int ii=0;ii<NCOV_FIX;ii++)
			fixed[ii] =  fix[ii]; 
//	}
	
	for (int isub = 0; isub < NSUBS; isub += localsize ) {
		if ((isub+localsize) > NSUBS)
		 	localsize = NSUBS - isub;
				
		if ((isub+localid) < NSUBS) {
			for (int j=0;j<NCOVAR;j++) 
				shared[localid + j*localsize] = covar[j*NSUBS + (isub+localid)];
		}
		__syncthreads();
				
		for (int j=0;j<localsize;j++) {
				
			float ZZ = Z[(isub+j)*TOTVOX+IDX];
			float phi = Phi[(isub+j)*TOTVOX+IDX];
			tmp = ZZ - bwhtmat;
		//	for (int ii=0;ii<NCOVAR;ii++) {
		//		tmp -= alph[ii] *shared[j + ii*localsize]; 
		//	}
			for (int ii=0;ii<NCOV_FIX;ii++) {
				tmp -= fixed[ii] * covF[(isub+j)*NCOV_FIX + ii]; 
			}
			tmp *= phi;
			for (current_covar=0;current_covar<NCOVAR;current_covar++) {	
				mean[current_covar] += shared[j + current_covar*localsize]*tmp;
			}
			for (int ii=0;ii<NCOVAR;ii++) {
				float ss = shared[j + ii*localsize]*phi;
				for (int jj=ii;jj<NCOVAR;jj++) {
					var[jj + ii*NCOVAR] += ss*shared[j + jj*localsize];
				}
			}

		}		
		__syncthreads();
	}

//	if (idx < N) {
		localState = state[idx];
		for (current_covar=0;current_covar<NCOVAR;current_covar++) {	
			sumV[current_covar]  = SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel-1])]
			      				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel+1])]
			      				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel-(NROW+2)])]
				  				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel+(NROW+2)])]
			      				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel-(NROW+2)*(NCOL+2)])]
				  				 + SpatCoef[(current_covar*TOTVOXp+dIdxSC[voxel+(NROW+2)*(NCOL+2)])];
		}
/**** fix up dtmpvar and dPrec matrices *****/
		for (int ii=0;ii<NCOVAR;ii++) {
			for (int jj=0;jj<NCOVAR;jj++) {	
				mean[ii] += dPrec[jj + ii*NCOVAR]*sumV[jj];
//				var[jj + ii*NCOVAR] = dtmpvar[jj + ii*NCOVAR]*Phi[IDX] + nbrs[idx]*dPrec[jj + ii*NCOVAR];
//				var[jj + ii*NCOVAR] += nbrs[idx]*dPrec[jj + ii*NCOVAR];
			}
		}
		float ss = nbrs[idx];
		for (int ii=0;ii<NCOVAR;ii++) {
			for (int jj=ii;jj<NCOVAR;jj++) {	
				var[jj + ii*NCOVAR] += ss*dPrec[jj + ii*NCOVAR];
				var[ii + jj*NCOVAR] = var[jj + ii*NCOVAR];
			}
		}

		// decompose and invert var 
		cholesky_decompGPU(var,NCOVAR);
		cholesky_invertGPU(NCOVAR, var);
		
		for (int ii=0;ii<NCOVAR;ii++) {
			tmp2[ii] = 0;
			for (int jj=0;jj<NCOVAR;jj++)
				tmp2[ii] += var[jj + ii*NCOVAR]*mean[jj];
		}			
		// draw MVN and with mean tmp2 and variance var --- result in mean;
		rmvnormGPU(mean,var,NCOVAR,tmp2,&localState,0);
			
		for (int ii=0;ii<NCOVAR;ii++) 
			SpatCoef[(ii*TOTVOXp+IDXSC)] = mean[ii];
	
		state[idx] = localState;
	}
}


__global__ void sub_mean(float *SpatCoef,const int N,int * vox,float cmean,int *dIdxSC)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float mean;
	int voxel,IDXSC;
	
	mean = cmean;
	
	while (idx < N) { 
		voxel = vox[idx];
		IDXSC = dIdxSC[voxel];
		SpatCoef[IDXSC] -= mean;
				
		idx += stride;
	}
}

void updateSpatCoefGPU(float *covar,float *spatCoef,float *SpatCoefMean,float *SpatCoefPrec,float *alpha,float *alphaMean,float *Z,unsigned char *msk,float beta,float *WM,unsigned long *seed)
{
	int this_one;
	float *d_sum,cmean,*h_sum;

	cudaMemcpyToSymbol(dPrec,SpatCoefPrec,sizeof(float)*NCOVAR*NCOVAR);
	
	if (MODEL == 1)
		cudaMemcpyToSymbol(dtmpvar,XXprime,sizeof(float)*NCOVAR*NCOVAR);
	
	int thr_per_block = TPB; 
	int shared_memory = (NCOVAR * thr_per_block) * sizeof(float);
	for (int i=0;i<2;i++) {
		int blck_per_grid = (INDX[i].hostN+thr_per_block-1)/thr_per_block;
		
		if (MODEL != 1) {	
			Spat_Coef<<<blck_per_grid,thr_per_block,shared_memory>>>(deviceCovar,deviceCov_Fix,deviceFixMean,deviceAlphaMean,deviceZ,
			devicePhi,deviceWM,deviceSpatCoef,beta,INDX[i].deviceNBRS,INDX[i].deviceVox,NSUBS,NROW,NCOL,NSUBTYPES,NCOVAR,NCOV_FIX,INDX[i].hostN,TOTVOXp,TOTVOX,devStates,deviceIdx,deviceIdxSC);
			if ((err = cudaGetLastError()) != cudaSuccess) {printf("Error %s %s\n",cudaGetErrorString(err)," in Spat_Coef");exit(0);}
		}
		else {
			Spat_Coef_Probit<<<blck_per_grid,thr_per_block,shared_memory>>>(deviceCovar,deviceCov_Fix,deviceFixMean,deviceAlphaMean,deviceZ,deviceWM,deviceSpatCoef,beta,INDX[i].deviceNBRS,INDX[i].deviceVox,NSUBS,NROW,NCOL,NSUBTYPES,NCOVAR,NCOV_FIX,INDX[i].hostN,TOTVOXp,TOTVOX,devStates,deviceIdx,deviceIdxSC);
			if ((err = cudaGetLastError()) != cudaSuccess) {printf("Error %s %s\n",cudaGetErrorString(err)," in Spat_Coef_Probit");exit(0);}
		}

	}
	

	for (this_one=0;this_one<NCOVAR;this_one++) {
		cudaMalloc((void **)&d_sum,BPG*sizeof(float));

		reduce<<<BPG, NN >>>(&deviceSpatCoef[this_one*TOTVOXp],d_sum,TOTVOXp);

		h_sum = (float *)calloc(BPG,sizeof(float));		
	
		cudaMemcpy(h_sum,d_sum,BPG*sizeof(float),cudaMemcpyDeviceToHost);
	
		cmean = 0;
		for (int j=0;j<BPG;j++)
			cmean += h_sum[j];
			
		free(h_sum);
		cudaFree(d_sum);
		cmean /= (float)TOTVOX;

//		for (int i=0;i<2;i++)
//			sub_mean<<<(INDX[i].hostN+511)/512,512>>>(&(deviceSpatCoef[this_one*TOTVOXp]),INDX[i].hostN,INDX[i].deviceVox,cmean,deviceIdxSC);
		alphaMean[this_one] = cmean/logit_factor;
	}
}



__global__ void betaMGPU(float *dcovar,float *dcovF,float *dZ,float *dspatCoef,float *fix,float *dalpha,float *dWM,const int N,const int TOTVOXp,const int NSUBS,const int NSUBTYPES,const int NCOVAR,const int NCOV_FIX,const int TOTVOX,int * vox,float *dmean,int *dIdx,int *dIdxSC)
{
	__shared__ float cache[NN];
	int cacheIndex = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float c = 0;
	float y,t;
	float temp = 0;
	float SC[20];
	float mean=0;
	int isub,ii,voxel,IDX,IDXSC;
	while (idx < N) { 
		voxel = vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];

		for (ii=0;ii<NCOVAR;ii++)
			SC[ii] = /*dalpha[ii] +*/ dspatCoef[ii*TOTVOXp+IDXSC];
			
		mean = 0;
		for (isub=0;isub<NSUBS;isub++) {
			mean += dZ[isub*TOTVOX+IDX];
			for (ii=0;ii<NCOVAR;ii++)
				mean -= SC[ii]*dcovar[ii*NSUBS+isub];
			for (ii=0;ii<NCOV_FIX;ii++)
				mean -= fix[ii]*dcovF[ii*NSUBS+isub];
		}
		
		y = mean * dWM[IDX] - c;
		t = temp + y;
		c = (t - temp) - y;
		temp = t;		
		
		idx += stride;
	}

	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (cacheIndex < i) 
			cache[cacheIndex] += cache[cacheIndex + i];
			
		__syncthreads();
		i = i >> 1;	
	}
	
	if (cacheIndex == 0) 
		dmean[blockIdx.x] = cache[0];
}


__global__ void betaVGPU(float *dWM,const int N,int *vox,float *dvar,int *dIdx)
{
	__shared__ float cache[NN];
	int cacheIndex = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float c = 0;
	float y,t;
	float temp = 0;

	int voxel,IDX;
	while (idx < N) { 
		voxel = vox[idx];
		IDX = dIdx[voxel];
		y = dWM[IDX]*dWM[IDX] - c;
		t = temp + y;
		c = (t - temp) - y;
		temp = t;		

		idx += stride;
	}

	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i = i >> 1;	
	}
	
	if (cacheIndex == 0) {
		dvar[blockIdx.x] = cache[0];
	}

}

void updateBetaGPU(float *beta,float *dZ,float *dPhi,float *dalpha,float *dWM,float prior_mean_beta,float prior_prec_beta,float *dspatCoef,float *dcovar,unsigned long *seed)
{
	float *dmean,*dvar,*hmean,*hvar;
	float mean,var,tmp;
	
	hmean = (float *)calloc(BPG,sizeof(float));
	hvar = (float *)calloc(BPG,sizeof(float));
	cudaMalloc((void **)&dmean,BPG*sizeof(float)) ;
	cudaMalloc((void **)&dvar,BPG*sizeof(float)) ;
	
	mean = var = 0;
	for (int i=0;i<2;i++) {
		betaMGPU<<<BPG,NN>>>(dcovar,deviceCov_Fix,dZ,dspatCoef,deviceFixMean,deviceAlphaMean,deviceWM,INDX[i].hostN,TOTVOXp,NSUBS,NSUBTYPES,NCOVAR,NCOV_FIX,TOTVOX,INDX[i].deviceVox,dmean,deviceIdx,deviceIdxSC);
		cudaMemcpy(hmean,dmean,BPG*sizeof(float),cudaMemcpyDeviceToHost);

		for (int j=0;j<BPG;j++)
			mean += hmean[j];
	}
	for (int i=0;i<2;i++) {
		betaVGPU<<<BPG,NN>>>(deviceWM,INDX[i].hostN,INDX[i].deviceVox,dvar,deviceIdx);
		cudaMemcpy(hvar,dvar,BPG*sizeof(float),cudaMemcpyDeviceToHost);

		for (int j=0;j<BPG;j++)
			var += hvar[j];
	}

	var *= NSUBS;
//	var += prior_prec_beta;
	var = 1./var;
//	mean = var*(mean + prior_mean_beta*prior_prec_beta);
	mean *= var;
	tmp = (float)rnorm((double)mean,sqrt((double)var),seed);
	*beta = tmp;
	
	cudaFree(dmean);
	cudaFree(dvar);
	free(hmean);
	free(hvar);
}

__global__ void alphaGPU(float *dcovar,float *dZ,float *dspatCoef,float *dWM,float beta,const int N,const int TOTVOXp,const int NSUBS,const int NSUBTYPES,const int NCOVAR,const int TOTVOX,int * vox,float *dmean,int *dIdx,int *dIdxSC,int current_cov)
{
	__shared__ float cache[NN];
	int cacheIndex = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float c=0;
	float y,t;
	float temp = 0;
	float SC[20];
	float tmp,tmp2;
	int isub,ii,voxel,IDX,IDXSC;
	
	while (idx < N) { 
		voxel = vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
		
		float WM = beta*dWM[IDX];
		
		for (ii=0;ii<NCOVAR;ii++) 
			SC[ii] = dspatCoef[ii*TOTVOXp+IDXSC];
		
		tmp = 0;
		for (isub=0;isub<NSUBS;isub++) {
			tmp2 = dZ[isub*TOTVOX+IDX] - WM;
			for (ii=0;ii<NCOVAR;ii++)
				tmp2 -= SC[ii]*dcovar[ii*NSUBS+isub];
			tmp2 *= dcovar[current_cov*NSUBS+isub];
			tmp += tmp2;
		}
				
		y = tmp - c;
		t = temp + y;
		c = (t - temp) - y;
		temp = t;		

		idx += stride;
	}

	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (cacheIndex < i) 
			cache[cacheIndex] += cache[cacheIndex + i];
			
		__syncthreads();
		i = i >> 1;	
	}
	
	if (cacheIndex == 0) 
		dmean[blockIdx.x] = cache[0];
}

void updateAlphaMeanGPU(float *alphaMean,float Prec,float *dcovar,float *dZ,float *dspatCoef,float beta,unsigned long *seed)
{
	double *mean,*V,*tmpmean;
	float *hmean,*dmean;

	mean = (double *)calloc(NCOVAR,sizeof(double));
	tmpmean = (double *)calloc(NCOVAR,sizeof(double));
	V = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));

	for (int i=0;i<NCOVAR;i++) {
		for (int j=0;j<NCOVAR;j++) {
			V[j + i*NCOVAR] = (double)TOTVOX*(double)XXprime[j + i*NCOVAR];
		}
	}

	cholesky_decomp(V, NCOVAR);
	cholesky_invert(NCOVAR, V);

	hmean = (float *)calloc(BPG,sizeof(float));
	cudaMalloc((void **)&dmean,BPG*sizeof(float)) ;

	for (int icovar=0;icovar<NCOVAR;icovar++) {
		for (int i=0;i<2;i++) {
			alphaGPU<<<BPG,NN>>>(dcovar,dZ,dspatCoef,deviceWM,beta,INDX[i].hostN,TOTVOXp,NSUBS,NSUBTYPES,NCOVAR,TOTVOX,INDX[i].deviceVox,
									dmean,deviceIdx,deviceIdxSC,icovar);
			cudaMemcpy(hmean,dmean,BPG*sizeof(float),cudaMemcpyDeviceToHost);

			for (int j=0;j<BPG;j++)
				tmpmean[icovar] += (double)hmean[j];
		}
		
	}

	cudaFree(dmean);
	free(hmean);

	for (int i=0;i<NCOVAR;i++) {
		mean[i] = 0;
		for (int j=0;j<NCOVAR;j++)
			mean[i] += V[j + i*NCOVAR]*tmpmean[j];
	}
	
	rmvnorm(tmpmean,V,NCOVAR,mean,seed,0);
		
	for (int i=0;i<NCOVAR;i++)
		alphaMean[i] = (float)tmpmean[i];
	
	cudaMemcpy(deviceAlphaMean,alphaMean,NCOVAR*sizeof(float),cudaMemcpyHostToDevice);

	free(tmpmean);
	free(mean);
	free(V);	

}

__global__ void fixGPU(float *dcovF,float *dcovar,float *dZ,float *dspatCoef,float *dWM,float beta,const int N,const int TOTVOXp,const int NSUBS,const int NSUBTYPES,const int NCOVAR,const int NCOV_FIX,const int TOTVOX,int * vox,float *dmean,int *dIdx,int *dIdxSC,int current_cov)
{
	__shared__ float cache[NN];
	int cacheIndex = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float c=0;
	float y,t;
	float temp = 0;
	float SC[20];
	float tmp,tmp2;
	int isub,ii,voxel,IDX,IDXSC;
	
	while (idx < N) { 
		voxel = vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
		
		float WM = beta*dWM[IDX];
		
		for (ii=0;ii<NCOVAR;ii++) 
			SC[ii] = dspatCoef[ii*TOTVOXp+IDXSC];
		
		tmp = 0;
		for (isub=0;isub<NSUBS;isub++) {
			tmp2 = dZ[isub*TOTVOX+IDX] - WM;
			for (ii=0;ii<NCOVAR;ii++)
				tmp2 -= SC[ii]*dcovar[ii*NSUBS+isub];
			tmp2 *= dcovF[current_cov*NSUBS+isub];
			tmp += tmp2;
		}
				
		y = tmp - c;
		t = temp + y;
		c = (t - temp) - y;
		temp = t;		

		idx += stride;
	}

	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (cacheIndex < i) 
			cache[cacheIndex] += cache[cacheIndex + i];
			
		__syncthreads();
		i = i >> 1;	
	}
	
	if (cacheIndex == 0) 
		dmean[blockIdx.x] = cache[0];
}

void updatefixMeanGPU(float *fixMean,float Prec,float *dcovar,float *dZ,float *dspatCoef,float beta,unsigned long *seed)
{
	double *mean,*V,*tmpmean;
	float *hmean,*dmean;

	mean = (double *)calloc(NCOV_FIX,sizeof(double));
	tmpmean = (double *)calloc(NCOV_FIX,sizeof(double));
	V = (double *)calloc(NCOV_FIX*NCOV_FIX,sizeof(double));

	for (int i=0;i<NCOV_FIX;i++) {
		for (int j=0;j<NCOV_FIX;j++) {
			V[j + i*NCOV_FIX] = (double)TOTVOX*(double)XXprime_Fix[j + i*NCOV_FIX];
		}
	}

	cholesky_decomp(V, NCOV_FIX);
	cholesky_invert(NCOV_FIX, V);

	hmean = (float *)calloc(BPG,sizeof(float));
	cudaMalloc((void **)&dmean,BPG*sizeof(float)) ;

	for (int icovar=0;icovar<NCOV_FIX;icovar++) {
		for (int i=0;i<2;i++) {
			fixGPU<<<BPG,NN>>>(deviceCov_Fix,dcovar,dZ,dspatCoef,deviceWM,beta,INDX[i].hostN,TOTVOXp,NSUBS,NSUBTYPES,NCOVAR,NCOV_FIX,TOTVOX,INDX[i].deviceVox,
									dmean,deviceIdx,deviceIdxSC,icovar);
			cudaMemcpy(hmean,dmean,BPG*sizeof(float),cudaMemcpyDeviceToHost);

			for (int j=0;j<BPG;j++)
				tmpmean[icovar] += (double)hmean[j];
		}
		
	}

	cudaFree(dmean);
	free(hmean);

	for (int i=0;i<NCOV_FIX;i++) {
		mean[i] = 0;
		for (int j=0;j<NCOV_FIX;j++)
			mean[i] += V[j + i*NCOV_FIX]*tmpmean[j];
	}
	
	rmvnorm(tmpmean,V,NCOV_FIX,mean,seed,0);
		
	for (int i=0;i<NCOV_FIX;i++)
		fixMean[i] = (float)tmpmean[i];
	
	cudaMemcpy(deviceFixMean,fixMean,NCOV_FIX*sizeof(float),cudaMemcpyHostToDevice);

	free(tmpmean);
	free(mean);
	free(V);	

}


__global__ void Spat_Coef_Prec(float *SpatCoef1,float *SpatCoef2,const int N,const int NROW,const int NCOL,int *vox,float *dbeta,int *dIdxSC)
{
	__shared__ float cache[NN];
	int cacheIndex = threadIdx.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float c = 0;
	float y,t;
	float temp = 0;
	
	float tmp,SC1,SC2,SCt1,SCt2;
	int voxel,tst,IDXSC;
	while (idx < N) { 
		voxel =  vox[idx];
		IDXSC = dIdxSC[voxel];
		y = 0;
		
		SC1 = SpatCoef1[IDXSC];
		SC2 = SpatCoef2[IDXSC];
		SCt1 = SpatCoef1[dIdxSC[voxel+1]];
		SCt2 = SpatCoef2[dIdxSC[voxel+1]];
		tst = (SCt1 != 0);
		tmp = (SC1 - SCt1)*(SC2 - SCt2)*tst;
		y += tmp;

		SCt1 = SpatCoef1[dIdxSC[voxel+(NROW+2)]];
		SCt2 = SpatCoef2[dIdxSC[voxel+(NROW+2)]];
		tst = (SCt1 != 0);
		tmp = (SC1 - SCt1)*(SC2 - SCt2)*tst;
		y += tmp;

		SCt1 = SpatCoef1[dIdxSC[voxel+(NROW+2)*(NCOL+2)]];
		SCt2 = SpatCoef2[dIdxSC[voxel+(NROW+2)*(NCOL+2)]];
		tst = (SCt1 != 0);
		tmp = (SC1 - SCt1)*(SC2 - SCt2)*tst;
		y += tmp;

		y -= c;
		t = temp + y;
		c = (t - temp) -y;
		temp = t;		
				
		idx += stride;
	}   
	
	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i = i >> 1;	
	}
	
	if (cacheIndex == 0)
		dbeta[blockIdx.x] = cache[0];
}

void updateSpatCoefPrecGPU_Laplace(float *SpatCoefPrec,unsigned long *seed)
{
	float *hbeta;
	double *var;
	float *dbeta;
	double betaSqr = 2.0;//0.2;
	double tau;

	var = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
	hbeta = (float*)calloc(BPG,sizeof(float));
	cudaMalloc((void **)&dbeta,BPG*sizeof(float));
	cudaMemset(dbeta,0,BPG*sizeof(float));
	
	for (int ii=0;ii<NCOVAR;ii++) {
//		for (int jj=0;jj<=ii;jj++) {  // filling in the lower triangular part of the matrix
			var[ii + ii*NCOVAR] = 0;
			for (int i=0;i<2;i++) {
				Spat_Coef_Prec<<<BPG,NN>>>(&(deviceSpatCoef[ii*TOTVOXp]),&(deviceSpatCoef[ii*TOTVOXp]),INDX[i].hostN,NROW,NCOL,INDX[i].deviceVox,dbeta,deviceIdxSC);
				if ((err = cudaGetLastError()) != cudaSuccess) {printf("Error %s %s\n",cudaGetErrorString(err)," in Spat_Coef_Prec");exit(0);}
				CUDA_CALL( cudaMemcpy(hbeta,dbeta,BPG*sizeof(float),cudaMemcpyDeviceToHost) );
				for (int j=0;j<BPG;j++) 
					var[ii + ii*NCOVAR] += (double)hbeta[j];
			}
//			var[ii + jj*NCOVAR] = var[jj + ii*NCOVAR];
//			if (ii==jj) var[ii + ii*NCOVAR] += 1;
			SpatCoefPrec[ii + ii*NCOVAR] = rGIG(((double)TOTVOX-3.0)/2.0,var[ii+ii*NCOVAR],betaSqr,seed);
//		printf("%f %f\n",SpatCoefPrec[ii + ii*NCOVAR],var[ii+ii*NCOVAR]);
//		}
	}	
//	printf("\n\n");
	free(hbeta);
	cudaFree(dbeta);
/*
	cholesky_decomp(var, NCOVAR);
	cholesky_invert(NCOVAR, var);
	cholesky_decomp(var,NCOVAR);
	
	double *tmp = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
	rwishart2(tmp,var,NCOVAR,TOTVOX-1,seed);

	for (int ii=0;ii<NCOVAR;ii++) 
		for (int jj=0;jj<=ii;jj++)   // filling in the lower triangular part of the matrix
			SpatCoefPrec[jj + ii*NCOVAR] = SpatCoefPrec[ii + jj*NCOVAR] = (float)tmp[jj + ii*NCOVAR];

	free(tmp);*/
	free(var);
}

void updateSpatCoefPrecGPU(float *SpatCoefPrec,unsigned long *seed)
{
	float *hbeta;
	double *var;
	float *dbeta;


	var = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
	hbeta = (float*)calloc(BPG,sizeof(float));
	cudaMalloc((void **)&dbeta,BPG*sizeof(float));
	cudaMemset(dbeta,0,BPG*sizeof(float));
	
	for (int ii=0;ii<NCOVAR;ii++) {
		for (int jj=0;jj<=ii;jj++) {  // filling in the lower triangular part of the matrix
//			var[ii + jj*NCOVAR] = 0;
			for (int i=0;i<2;i++) {
				Spat_Coef_Prec<<<BPG,NN>>>(&(deviceSpatCoef[ii*TOTVOXp]),&(deviceSpatCoef[jj*TOTVOXp]),INDX[i].hostN,NROW,NCOL,INDX[i].deviceVox,dbeta,deviceIdxSC);
				if ((err = cudaGetLastError()) != cudaSuccess) {printf("Error %s %s\n",cudaGetErrorString(err)," in Spat_Coef_Prec");exit(0);}
				CUDA_CALL( cudaMemcpy(hbeta,dbeta,BPG*sizeof(float),cudaMemcpyDeviceToHost) );
				for (int j=0;j<BPG;j++) 
					var[jj + ii*NCOVAR] += (double)hbeta[j];
			}
			var[ii + jj*NCOVAR] = var[jj + ii*NCOVAR];
			if (ii==jj) var[ii + ii*NCOVAR] += 1;
		}
	}	
	
	free(hbeta);
	cudaFree(dbeta);
/*	for (int ii=0;ii<NCOVAR;ii++) {
		for (int jj=0;jj<=ii;jj++) {  // filling in the lower triangular part of the matrix
			printf("%g ",var[jj + ii*NCOVAR]);
		}
		printf("\n");
	}
*/	
	cholesky_decomp(var, NCOVAR);
	cholesky_invert(NCOVAR, var);
	cholesky_decomp(var,NCOVAR);
	
	double *tmp = (double *)calloc(NCOVAR*NCOVAR,sizeof(double));
	rwishart2(tmp,var,NCOVAR,TOTVOX+20-1,seed);

	for (int ii=0;ii<NCOVAR;ii++) 
		for (int jj=0;jj<=ii;jj++)   // filling in the lower triangular part of the matrix
			SpatCoefPrec[jj + ii*NCOVAR] = SpatCoefPrec[ii + jj*NCOVAR] = (float)tmp[jj + ii*NCOVAR];

	free(tmp);
	free(var);
}

__device__ float truncNormLeft(float a,curandState *state)
{
	/* Computes a truncated N(0,1) on x>a */
	float x,a_star;
	int stop;
//	double rnorm(double,double,unsigned long *);
//	double kiss(unsigned long *);
	
	
	if (a<0.0f)
	{
		stop=0;
		while (stop==0)
		{
			x = curand_normal(state);
			if( x>a ) stop=1;
			else stop=0;
		}
	}
	else
	{
		a_star=0.5f*(a+sqrtf(a*a+4.0f));
		stop=0;
		while (stop==0)
		{
			x=a-__logf(curand_uniform(state))/a_star;
			if( __logf(curand_uniform(state))<x*(a_star-0.5f*x)-a_star*a_star*0.5f ) stop=1;
			else stop=0;
		}
	}
	
	
	return(x);
}


__device__ float truncNormGPU(float mu,float std,float u, float v,curandState *state)
{
	/*__________________Optimal truncated normal N(mu,std), a<x<b
	 (Robert, Stat&Computing, 1995)        ____________*/
	float x,a,b,boun,boo;
	int stop;
	
	u = (u-mu)/std;
	v = (v-mu)/std;
	if (v<0)
	{
		a=-v; b=-u;
	}
	else
	{
		a=u; b=v;
	}
	if (b>a+3*__expf(-a*a-1/(a*a))/(a+sqrtf(a*a+4.0f)))
	{
		stop=0;
		while (stop==0)
		{
			x=truncNormLeft(a,state);
			if( x < b ) stop=1;
			else stop = 0;
		}
	}
	else
	{
		boo=0.0f;
		if (b<0.0f)
			boo=b*b;
		if (a>0.0f)
			boo=a*a;
		
		
		stop=0;
		while (stop==0)
		{
			x=(b-a)*curand_uniform(state)+a;
			boun=boo-x*x;
			if( 2.0f*__logf(curand_uniform(state))<boun ) stop=1;
			else stop=0;
		}
	}
	if (v <= 0)
		x = -x;
	return (std*x + mu);
}

__device__ inline float discrete_N(int n,float *ChiSqHist,curandState *state)
{
  float U;
 
  U = curand_uniform(state);
  for (int i=0;i<n;i++) {
	if (U <= ChiSqHist[i]) {
	   U *= (float)n;
	   break;
	}
}    return U;
// return len-1;
}

__device__ inline float rexpGPU(float beta,curandState *state)
{
 return -logf(curand_uniform(state))/beta;
}

__device__ inline float fsignfGPU(float num, float sign )
/* Transfers sign of argument sign to argument num */
{
	if ( ( sign>0.0f && num<0.0f ) || ( sign<0.0f && num>0.0f ) )
		return -num;
	else return num;
}

__device__ float  sgammaGPU(float a, curandState *state) {
	const float q1 = 0.0416666664f;
	const float q2 = 0.0208333723f;
	const float q3 = 0.0079849875f;
	const float q4 = 0.0015746717f;
	const float q5 = -0.0003349403f;
	const float q6 = 0.0003340332f;
	const float q7 = 0.0006053049f;
	const float q8 = -0.0004701849f;
	const float q9 = 0.0001710320f;
	const float a1 = 0.333333333f;
	const float a2 = -0.249999949f;
	const float a3 = 0.199999867f;
	const float a4 = -0.166677482f;
	const float a5 = 0.142873973f;
	const float a6 = -0.124385581f;
	const float a7 = 0.110368310f;
	const float a8 = 0.112750886f;
	const float a9 = 0.104089866f;
	const float e1 = 1.000000000f;
	const float e2 = 0.499999994f;
	const float e3 = 0.166666848f;
	const float e4 = 0.041664508f;
	const float e5 = 0.008345522f;
	const float e6 = 0.001353826f;
	const float e7 = 0.000247453f;
	float aa = 0.0f;
	float aaa = 0.0f;
	const float sqrt32 = 5.65685424949238f;
	float sgamma,s2,s,d,t,x,u,r,q0,b,si,c,v,q,e,w,p;

   if(a == aa) 
		goto S2;
    if(a < 1.0f) 
		goto S13;

//S1: // STEP  1:  RECALCULATIONS OF S2,S,D IF A HAS CHANGED
    aa = a;
    s2 = a-0.5f;
    s = sqrtf(s2);
    d = sqrt32-12.0f*s;
	
S2: // STEP  2:  T=STANDARD NORMAL DEVIATE,	 X=(S,1/2)-NORMAL DEVIATE.	 IMMEDIATE ACCEPTANCE (I)
    t = curand_normal(state);
    x = s+0.5f*t;
    sgamma = x*x;
    if(t >= 0.0f) 
		return sgamma;

//S3: // STEP  3:  U= 0,1 -UNIFORM SAMPLE. SQUEEZE ACCEPTANCE (S)
    u = curand_uniform(state);
    if(d*u <= t*t*t) 
		return sgamma;

//S4: // STEP  4:  RECALCULATIONS OF Q0,B,SI,C IF NECESSARY
	if (a != aaa) {
		aaa = a;
		r = 1.0f/ a;
		q0 = r*(q1+r*(q2+r*(q3+r*(q4+r*(q5+r*(q6+r*(q7+r*(q8+r*q9))))))));
		if (a <= 3.686f) {
			b = 0.463f+s+0.178f*s2;
			si = 1.235f;
			c = 0.195f/s-0.079f+0.16f*s;
		}
		else if (a <= 13.022f) {
			b = 1.654f+0.0076f*s2;
			si = 1.68f/s+0.275f;
			c = .062/s+0.024f;
		}
		else {
			b = 1.77f;
			si = 0.75f;
			c = 0.1515f/s;			
		}
	}

//S5: //  NO QUOTIENT TEST IF X NOT POSITIVE
    if(x <= 0.0f) 
		goto S8;
	
//S6: // CALCULATION OF V AND QUOTIENT Q
    v = t/(s+s);
    if(fabsf(v) > 0.25f) 
		q = q0-s*t+0.25f*t*t+(s2+s2)*log1pf(v);
	else
		q = q0+0.5f*t*t*v*(a1+v*(a2+v*(a3+v*(a4+v*(a5+v*(a6+v*(a7+v*(a8+v*a9))))))));

//S7: //  QUOTIENT ACCEPTANCE (Q)
	if(log1pf(-u) <= q) return sgamma;

S8: // E=STANDARD EXPONENTIAL DEVIATE  U= 0,1 -UNIFORM DEVIATE 	 T=(B,SI)-DOUBLE EXPONENTIAL (LAPLACE) SAMPLE
    e = rexpGPU(1.0f,state);
    u = curand_uniform(state);
    u += (u-1.0f);
    t = b+fsignfGPU(si*e,u);

//S9: //   REJECTION IF T .LT. TAU(1) = -.71874483771719
	if(t <= -0.71874483771719f) 
		goto S8;

//S10: // CALCULATION OF V AND QUOTIENT Q
	v = t/(s+s);
	if(fabsf(v) > 0.25f) 
		q = q0-s*t+0.25f*t*t+(s2+s2)*log1pf(v);
	else
		q = q0+0.5f*t*t*v*(a1+v*(a2+v*(a3+v*(a4+v*(a5+v*(a6+v*(a7+v*(a8+v*a9))))))));

//S11: // HAT ACCEPTANCE (H) (IF Q NOT POSITIVE GO TO STEP 8)
	if (q <= 0.5f)
		w = q*(e1+q*(e2+q*(e3+q*(e4+q*(e5+q*(e6+q*e7))))));
	else 
		w = expf(q)-1.0f;
   if(q <= 0.0f || c*fabs(u) > w*expf(e-0.5f*t*t)) 
		goto S8;
//S12: 
    x = s+0.5f*t;
    sgamma = x*x;
    return sgamma;
	
S13: // ALTERNATE METHOD FOR PARAMETERS A BELOW 1  (.3678794=EXP(-1.))
    aa = 0.0f;
    b = 1.0f+0.3678794f*a;

S14:
    p = b*curand_uniform(state);
    if(p >= 1.0f) 
		goto S15;
    sgamma = expf(logf(p)/ a);
    if(rexpGPU(1.0f,state) < sgamma) 
		goto S14;
    return sgamma;

S15:
    sgamma = -logf((b-p)/ a);
    if(rexpGPU(1.0f,state) < (1.0f-a)*logf(sgamma)) 
		goto S14;
    return sgamma;	

}


__global__ void Z_GPU1(float *dZ,unsigned char *data,float *covar,float *covarF,float *fix,float *alpha,float *WM,float *SpatCoef,const float beta,int *vox,const int NSUBS,const int NSUBTYPES,const int NCOVAR,const int NCOV_FIX,const int N,const int TOTVOXp,const int TOTVOX,curandState *state,int *dIdx,int *dIdxSC,const float sqrt2,unsigned int *dR)
{
	extern __shared__ float shared[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int localsize = blockDim.x;  
	int localid = threadIdx.x; 
	
	float mean[2000];
	float tmp=0;
	float SC[20];
	int voxel=0,IDX=0,IDXSC=0;
	curandState localState;

	if (idx < N) {
		voxel =  vox[idx];
		IDX   = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
			
		tmp = beta*WM[IDX];
	
		for (int ii=0;ii<NCOVAR;ii++)
			SC[ii] = /*alpha[ii] +*/ SpatCoef[ii*TOTVOXp+IDXSC];
	}

	for (int isub = 0; isub < NSUBS; isub += localsize ) {
		if ((isub+localsize) > NSUBS)
			localsize = NSUBS - isub;
				
		if ((isub+localid) < NSUBS) {
			for (int j=0;j<NCOVAR;j++) 
//				shared[localid + j*localsize] = covar[(isub+localid)*NCOVAR + j];
				shared[localid + j*localsize] = covar[j*NSUBS + (isub+localid)];
		}
		__syncthreads();
				
		for (int j=0;j<localsize;j++) {
			mean[isub+j] = tmp;	
			for (int ii=0;ii<NCOVAR;ii++)
				mean[isub+j] += SC[ii]*shared[j + ii*localsize];
			for (int ii=0;ii<NCOV_FIX;ii++)
				mean[isub+j] += fix[ii]*covarF[(isub+j)*NCOV_FIX + ii];
		}		
		__syncthreads();
	}

	if (idx < N) {
		float2 lim[2];
		lim[0].x = -500.0f;
		lim[0].y = -0.000001f; 
		lim[1].x =  0.000001f;
		lim[1].y = 500.0f; 
		localState = state[idx];
		for (int isub=0;isub<NSUBS;isub++) {
			int itmp = (int)data[isub*TOTVOX+IDX];
			dZ[isub*TOTVOX+IDX] = truncNormGPU(mean[isub],1.0f,lim[itmp].x,lim[itmp].y,&localState);
//			dZ[isub*TOTVOX+IDX] = ((float)data[isub*TOTVOX+IDX]-1.0f)*truncNormGPU(fabsf(mean[isub]),1.0f,0.0f,500.0f,&localState);
//			dR[isub*TOTVOX+IDX] += (fabsf(dZ[isub*TOTVOX+IDX] - mean[isub]) > 4.0f);
		}
		state[idx] = localState;
	}
}

__global__ void Z_GPU2(float *dZ,unsigned char *data,float *covar,float *covarF,float *fix,float *alpha,float *WM,float *SpatCoef,const float beta,int *vox,const int NSUBS,const int NSUBTYPES,const int NCOVAR,const int NCOV_FIX,const int N,const int TOTVOXp,const int TOTVOX,curandState *state,int *dIdx,int *dIdxSC,const float sqrt2,float *dPhi,unsigned int *dR,const float alp,const float df)
{
	extern __shared__ float shared[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int localsize = blockDim.x;  
	int localid = threadIdx.x; 
	
	float a = alp;
	float mean[2000];
	float tmp=0;
	float SC[20];
	int voxel=0,IDX=0,IDXSC=0;
	curandState localState;

	if (idx < N) {  
		voxel =  vox[idx];
		IDX   = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
			
		tmp = beta*WM[IDX];
	
		for (int ii=0;ii<NCOVAR;ii++)
			SC[ii] = /*alpha[ii] +*/ SpatCoef[ii*TOTVOXp+IDXSC];
	}

	for (int isub = 0; isub < NSUBS; isub += localsize ) {
		if ((isub+localsize) > NSUBS)
			localsize = NSUBS - isub;
				
		if ((isub+localid) < NSUBS) {
			for (int j=0;j<NCOVAR;j++) 
//				shared[localid + j*localsize] = covar[(isub+localid)*NCOVAR + j];
				shared[localid + j*localsize] = covar[j*NSUBS + (isub+localid)];
		}
		__syncthreads();
				
		for (int j=0;j<localsize;j++) {
			mean[isub+j] = tmp;	
			for (int ii=0;ii<NCOVAR;ii++)
				mean[isub+j] += SC[ii]*shared[j + ii*localsize];
			for (int ii=0;ii<NCOV_FIX;ii++)
				mean[isub+j] += fix[ii]*covarF[(isub+j)*NCOV_FIX + ii];
		}		
		__syncthreads();
	}

	if (idx < N) {
		float2 lim[2];
		lim[0].x = -500.0f;
		lim[0].y = -0.000001f; 
		lim[1].x =  0.000001f;
		lim[1].y = 500.0f; 
		localState = state[idx];
		float b;
		for (int isub=0;isub<NSUBS;isub++) {
			float Z = dZ[isub*TOTVOX+IDX];
			float M = mean[isub];
			float P = rsqrtf(dPhi[isub*TOTVOX+IDX]);
			
			int itmp = (int)data[isub*TOTVOX+IDX];
			Z = truncNormGPU(M,P,lim[itmp].x,lim[itmp].y,&localState);
//			Z = ((float)data[isub*TOTVOX+IDX]-1.0f)*truncNormGPU(fabsf(M),rsqrtf(P),0.0f,500.0f,&localState);
//			dR[isub*TOTVOX+IDX] += (fabsf(Z - M) > 4.0f);
			
			b = Z - M;
			b = (df + b*b)/2.0f;
			P = sgammaGPU(a,&localState)/b;
			
			dZ[isub*TOTVOX+IDX] = Z;
			dPhi[isub*TOTVOX+IDX] = P;
		}
		state[idx] = localState;
	}
}


__global__ void Phi_GPU(float *dZ,unsigned char *data,float *covar,float *covarF,float *fix,float *alpha,float *WM,float *SpatCoef,const float beta,int *vox,const int NSUBS,const int NSUBTYPES,const int NCOVAR,const int NCOV_FIX,const int N,const int TOTVOXp,const int TOTVOX,curandState *state,int *dIdx,int *dIdxSC,const float sqrt2,float *dPhi,const float alp,const float df)
{
	extern __shared__ float shared[];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int localsize = blockDim.x;  
	int localid = threadIdx.x; 
	
	const float a = alp;
	float beta2[2000];
	float tmp=0;
	float SC[20];
	int voxel=0,IDX=0,IDXSC=0;
	curandState localState;

	if (idx < N) {  
		voxel =  vox[idx];
		IDX   = dIdx[voxel];
		IDXSC = dIdxSC[voxel];
			
		tmp = beta*WM[IDX];
	
		for (int ii=0;ii<NCOVAR;ii++)
			SC[ii] = /*alpha[ii] +*/ SpatCoef[ii*TOTVOXp+IDXSC];
	}

	for (int isub = 0; isub < NSUBS; isub += localsize ) {
		if ((isub+localsize) > NSUBS)
			localsize = NSUBS - isub;
				
		if ((isub+localid) < NSUBS) {
			for (int j=0;j<NCOVAR;j++) 
//				shared[localid + j*localsize] = covar[(isub+localid)*NCOVAR + j];
				shared[localid + j*localsize] = covar[j*NSUBS + (isub+localid)];
		}
		__syncthreads();
				
		for (int j=0;j<localsize;j++) {
			beta2[isub+j] = tmp;	
			for (int ii=0;ii<NCOVAR;ii++)
				beta2[isub+j] += SC[ii]*shared[j + ii*localsize];
//			for (int ii=0;ii<NCOV_FIX;ii++)
//				beta2[isub+j] -= fix[ii]*covarF[(isub+j)*NCOV_FIX + ii];
		}		
		__syncthreads();
	}

	if (idx < N) {
		localState = state[idx];
		float b;
		//float sg = sgammaGPU(a,&localState);
		for (int isub=0;isub<NSUBS;isub++) {
			b = dZ[isub*TOTVOX+IDX] - beta2[isub];
			b = (df + b*b)/2.0f;
			dPhi[isub*TOTVOX+IDX] = sgammaGPU(a,&localState)/b;
		}
		state[idx] = localState;
	}
}

void updatePhiGPU(float beta,float *Phi)
{
	float sqrt2 = sqrtf(2);
	float alpha = (t_df + 1.0f)/2.0f;
	int N=0;
	for (int i=0;i<2;i++)
		N = (INDX[i].hostN > N) ? INDX[i].hostN:N;

	if (MODEL != 1) {		 

		for (int i=0;i<2;i++) {
			int thr_per_block = TPB2; 
			int shared_memory = (NCOVAR * thr_per_block) * sizeof(float);
			int blck_per_grid = (INDX[i].hostN+thr_per_block-1)/thr_per_block;
	
			Phi_GPU<<<blck_per_grid,thr_per_block,shared_memory>>>(deviceZ,deviceData,deviceCovar,deviceCov_Fix,deviceFixMean,deviceAlphaMean,
			deviceWM,deviceSpatCoef,beta,INDX[i].deviceVox,NSUBS,NSUBTYPES,NCOVAR,NCOV_FIX,
			INDX[i].hostN,TOTVOXp,TOTVOX,devStates,deviceIdx,deviceIdxSC,sqrt2,devicePhi,alpha,t_df);
			if ((err = cudaGetLastError()) != cudaSuccess) {printf("Error %s %s\n",cudaGetErrorString(err)," in Phi_GPU");exit(0);}
		}
		
	}
}

void updatePhi(unsigned char *data,float *Z,float *Phi,unsigned char *msk,float beta,float *WM,float *spatCoef,float *covar,unsigned long *seed) 
{
	double alpha = ((double)t_df + 1)/2;
	unsigned char d;
	double beta2,mx= -10,mZ=-10,mtmp,mZ2=-10;

	CUDA_CALL( cudaMemcpy(spatCoef,deviceSpatCoef,TOTVOXp*NCOVAR*sizeof(float),cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaMemcpy(Z,deviceZ,TOTVOX*NSUBS*sizeof(float),cudaMemcpyDeviceToHost) );
	for (int i=0;i<2;i++) {
		for (int j=0;j<INDX[i].hostN;j++) {
			int vox = INDX[i].hostVox[j];
			int IDX = hostIdx[vox];
			int IDXSC = hostIdxSC[vox];
			for (int isub=0;isub<NSUBS;isub++) {
				float tmp = beta*WM[IDX];
				for (int ii=0;ii<NCOVAR;ii++)
					tmp += spatCoef[ii*TOTVOXp+IDXSC]*covar[ii*NSUBS+isub];
				beta2 = Z[isub*TOTVOX+IDX] - tmp;
				if (mx < fabs(beta2))  {
					mx = fabs(beta2);
					mZ = Z[isub*TOTVOX+IDX];
					mtmp = tmp;
					d = data[isub*TOTVOX+IDX];
				}
				mZ2 = (mZ2 >fabs(Z[isub*TOTVOX+IDX])) ? mZ2:fabs(Z[isub*TOTVOX+IDX]);
				beta2 = ((double)t_df + beta2*beta2)/2.0;
				Phi[isub*TOTVOX+IDX] = (float)rgamma(alpha,beta2,seed);
		//		Phi[isub*TOTVOX+IDX] = (float)rgamma(0.5*(double)t_df,0.5*(double)t_df,seed);
			//	printf("%lf\n",Phi[isub*TOTVOX+IDX]);
			}
		}
	}
	CUDA_CALL( cudaMemcpy(devicePhi,Phi,NSUBS*TOTVOX*sizeof(float),cudaMemcpyHostToDevice) );	
	printf("beta2_max = %lf %lf %lf\t %lf %d\n",mx,mZ,mtmp,mZ2,(int)d);fflush(NULL);
}

void updateZGPU(unsigned char *data,float beta,unsigned long *seed)
{
	float alpha = (t_df + 1)/2;
	float sqrt2 = sqrtf(2);
//	float *rchisq,*drchisq;
//	float dfdiv2 = t_df/2;
	int N=0;
	for (int i=0;i<2;i++)
		N = (INDX[i].hostN > N) ? INDX[i].hostN:N;

	if (MODEL != 1) {		 
//		rchisq = (float *)malloc(N*sizeof(float));
//		cudaMalloc((void **)&drchisq,N*sizeof(float));

		for (int i=0;i<2;i++) {
//			for (int j=0;j<INDX[i].hostN;j++)
//				rchisq[j] = rgamma(dfdiv2,dfdiv2,seed);
//			cudaMemcpy(drchisq,rchisq,INDX[i].hostN*sizeof(float),cudaMemcpyHostToDevice);
			int thr_per_block = TPB2; 
			int shared_memory = (NCOVAR * thr_per_block) * sizeof(float);
			int blck_per_grid = (INDX[i].hostN+thr_per_block-1)/thr_per_block;
	
			Z_GPU2<<<blck_per_grid,thr_per_block,shared_memory>>>(deviceZ,deviceData,deviceCovar,deviceCov_Fix,deviceFixMean,deviceAlphaMean,
			deviceWM,deviceSpatCoef,beta,INDX[i].deviceVox,NSUBS,NSUBTYPES,NCOVAR,NCOV_FIX,
			INDX[i].hostN,TOTVOXp,TOTVOX,devStates,deviceIdx,deviceIdxSC,sqrt2,devicePhi,deviceResid,alpha,t_df);
			if ((err = cudaGetLastError()) != cudaSuccess) {printf("Error %s %s\n",cudaGetErrorString(err)," in Z_GPU2");exit(0);}		
		}
//		free(rchisq);
//		cudaFree(drchisq);
	}
	else {

		for (int i=0;i<2;i++) {
			int thr_per_block = TPB2; 
			int shared_memory = (NCOVAR * thr_per_block) * sizeof(float);
			int blck_per_grid = (INDX[i].hostN+thr_per_block-1)/thr_per_block;

			Z_GPU1<<<blck_per_grid,thr_per_block,shared_memory>>>(deviceZ,deviceData,deviceCovar,deviceCov_Fix,deviceFixMean,
			deviceAlphaMean,deviceWM,deviceSpatCoef,beta,INDX[i].deviceVox,NSUBS,NSUBTYPES,NCOVAR,NCOV_FIX,
			INDX[i].hostN,TOTVOXp,TOTVOX,devStates,deviceIdx,deviceIdxSC,sqrt2,deviceResid);
			if ((err = cudaGetLastError()) != cudaSuccess) {printf("Error %s %s\n",cudaGetErrorString(err)," in Z_GPU1");exit(0);}		
	
		}
	}
}


__device__ inline float dProbSDNormGPU(const float x)
//Calculate the cumulative probability of standard normal distribution;
{
	const float b1 =  0.319381530f;
	const float b2 = -0.356563782f;
	const float b3 =  1.781477937f;
	const float b4 = -1.821255978f;
	const float b5 =  1.330274429f;
	const float p  =  0.2316419f;
	const float c  =  0.39894228f;
	float t,sgnx,out,y;
	
	sgnx = copysignf(1.0f,x);
	y = sgnx*x;
	
	t = 1.0f / ( 1.0f + p * y );
		
	out = ((x >= 0.0f) - sgnx*c * expf( -y * y / 2.0f ) * t *
				( t *( t * ( t * ( t * b5 + b4 ) + b3 ) + b2 ) + b1 ));
			
	return out;
}

__global__ void ProbLogitGPU(float *prb,float *alpha,float *SpatCoef,float beta,float *WM,int *dIdx,int *dIdxSC,int *vox,const int TOTVOX,const int TOTVOXp,const int NSUBTYPES,const int N,const float logit_factor)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float x;
	float bwhtmat=0;
	int voxel,ii,IDX,IDXSC;
	
	while (idx < N) { 
		voxel =  vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];

		bwhtmat = beta*WM[IDX];
		
		for (ii=0;ii<NSUBTYPES;ii++) {
//			x = /*alpha[ii] + */ SpatCoef[ii*TOTVOXp+IDXSC] + bwhtmat;	// average covar is zero (covars are centered) 
																			// so don't need to add in the spatially varying parms	
//			prb[ii*TOTVOX+IDX] = dProbSDNormGPU(x);								// so don't need to add in the spatially varying parms	
			x = expf(SpatCoef[ii*TOTVOXp+IDXSC]/logit_factor);
			prb[ii*TOTVOX+IDX] = x/(1.f+x);
		}
		
		idx += stride;
	}
}

__global__ void ProbSDNormGPU(float *prb,float *alpha,float *SpatCoef,float beta,float *WM,int *dIdx,int *dIdxSC,int *vox,const int TOTVOX,const int TOTVOXp,const int NSUBTYPES,const int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float x;
	float bwhtmat=0;
	int voxel,ii,IDX,IDXSC;
	
	while (idx < N) { 
		voxel =  vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];

		bwhtmat = beta*WM[IDX];
		
		for (ii=0;ii<NSUBTYPES;ii++) {
			x = /*alpha[ii] + */ SpatCoef[ii*TOTVOXp+IDXSC] + bwhtmat;	// average covar is zero (covars are centered) 
																			// so don't need to add in the spatially varying parms	
			prb[ii*TOTVOX+IDX] = dProbSDNormGPU(x);								// so don't need to add in the spatially varying parms	
		}
		
		idx += stride;
	}
}


__global__ void ProbLogitGPU_prediction(float *covar,float *covarFix,float *pred,float *fix,float *alpha,float *SpatCoef,float beta,float *WM,int *dIdx,int *dIdxSC,int *vox,const int TOTVOX,const int TOTVOXp,const int NSUBTYPES,const int NCOVAR,const int NSUBS,const int NCOV_FIX,const int isub,const int N,const float logit_factor)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float x,y;
	float bwhtmat=0;
	int voxel,ii,IDX,IDXSC;
	while (idx < N) { 
		voxel =  vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];

		bwhtmat = beta*WM[IDX]/logit_factor;
		
		y = 0;
		for (ii=NSUBTYPES;ii<NCOVAR;ii++) {
//			y += (/*alpha[ii] + */SpatCoef[ii*TOTVOXp+IDXSC]/logit_factor)*covar[isub*NCOVAR+ii];
			y += (/*alpha[ii] + */SpatCoef[ii*TOTVOXp+IDXSC]/logit_factor)*covar[ii*NSUBS + isub];
		}
		for (ii=0;ii<NCOV_FIX;ii++)
			y += fix[ii]*covarFix[isub*NCOV_FIX+ii]/logit_factor;
		for (ii=0;ii<NSUBTYPES;ii++) {
//			x = /*alpha[ii] + */ SpatCoef[ii*TOTVOXp+IDXSC] + bwhtmat + y;	
//			pred[ii*TOTVOX+IDX] = dProbSDNormGPU(x);									
			x = expf(SpatCoef[ii*TOTVOXp+IDXSC]/logit_factor + bwhtmat + y);	
			pred[ii*TOTVOX+IDX] = x/(1.f+x);									
		}
		
		idx += stride;
	}
}

__global__ void ProbSDNormGPU_prediction(float *covar,float *covarFix,float *pred,float *fix,float *alpha,float *SpatCoef,float beta,float *WM,int *dIdx,int *dIdxSC,int *vox,const int TOTVOX,const int TOTVOXp,const int NSUBTYPES,const int NCOVAR,const int NSUBS,const int NCOV_FIX,const int isub,const int N)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x*gridDim.x;
	
	float x,y;
	float bwhtmat=0;
	int voxel,ii,IDX,IDXSC;
	while (idx < N) { 
		voxel =  vox[idx];
		IDX = dIdx[voxel];
		IDXSC = dIdxSC[voxel];

		bwhtmat = beta*WM[IDX];
		
		y = 0;
		for (ii=NSUBTYPES;ii<NCOVAR;ii++) {
//			y += (/*alpha[ii] + */SpatCoef[ii*TOTVOXp+IDXSC])*covar[isub*NCOVAR+ii];
			y += (/*alpha[ii] + */SpatCoef[ii*TOTVOXp+IDXSC])*covar[ii*NSUBS + isub];
		}
		for (ii=0;ii<NCOV_FIX;ii++)
			y += fix[ii]*covarFix[isub*NCOV_FIX+ii];
		for (ii=0;ii<NSUBTYPES;ii++) {
			x = /*alpha[ii] + */ SpatCoef[ii*TOTVOXp+IDXSC] + bwhtmat + y;	
			pred[ii*TOTVOX+IDX] = dProbSDNormGPU(x);								
		}
		
		idx += stride;
	}
}

void compute_prbGPU(float beta)
{
	if (MODEL != 1) {
		for (int i=0;i<2;i++)
			ProbLogitGPU<<<(INDX[i].hostN+511)/512, 512 >>>(devicePrb,deviceAlphaMean,deviceSpatCoef,beta,deviceWM,deviceIdx,deviceIdxSC,INDX[i].deviceVox,TOTVOX,TOTVOXp,NSUBTYPES,INDX[i].hostN,logit_factor);		
	}
	else {
		for (int i=0;i<2;i++)
			ProbSDNormGPU<<<(INDX[i].hostN+511)/512, 512 >>>(devicePrb,deviceAlphaMean,deviceSpatCoef,beta,deviceWM,deviceIdx,deviceIdxSC,INDX[i].deviceVox,TOTVOX,TOTVOXp,NSUBTYPES,INDX[i].hostN);		
	}
}

__global__ void reduce_for_Mpredict(float *in,float *out,unsigned char *data,const int N)
{
	__shared__ float cache[NN];
	int cacheIndex = threadIdx.x;
	int ivox = threadIdx.x + blockIdx.x * blockDim.x;
	float c = 0;
	float y,t;
	float temp = 0;
	
	while (ivox < N) {
		int tmp = (int)data[ivox];
		float inv = in[ivox];
		y = tmp*logf(inv + 1E-35f) + (1 - tmp)*logf(1.0f - inv + 1E-35f) - c;
		t = temp + y;
		c = (t - temp) - y;
		temp = t;
		ivox += blockDim.x * gridDim.x;
	}
			
	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i = i >> 1;	
	}
	
	if (cacheIndex == 0)
		out[blockIdx.x] = cache[0];
}

double compute_predictGPU(float beta,unsigned char *data,unsigned char *msk,float *covar,float *predict,double **Qhat,int first)
{
	double *Mpredict,DIC;
	float *hf_sum,*df_sum;
	
	Mpredict = (double *)calloc(NSUBTYPES,sizeof(double));
	
	CUDA_CALL( cudaMalloc((void **)&df_sum,BPG*sizeof(float)) );
	hf_sum = (float *)calloc(BPG,sizeof(float));		

	DIC = 0;
	for (int isub=0;isub<NSUBS;isub++) {
		if (MODEL != 1) {
			for (int i=0;i<2;i++) {
				ProbLogitGPU_prediction<<<(INDX[i].hostN+511)/512, 512 >>>(deviceCovar,deviceCov_Fix,devicePredict,deviceFixMean,
				deviceAlphaMean,deviceSpatCoef,beta,deviceWM,deviceIdx,deviceIdxSC,INDX[i].deviceVox,TOTVOX,TOTVOXp,
				NSUBTYPES,NCOVAR,NSUBS,NCOV_FIX,isub,INDX[i].hostN,logit_factor);
			}
		}
		else {
			for (int i=0;i<2;i++) {
				ProbSDNormGPU_prediction<<<(INDX[i].hostN+511)/512, 512 >>>(deviceCovar,deviceCov_Fix,devicePredict,deviceFixMean,
				deviceAlphaMean,deviceSpatCoef,beta,deviceWM,deviceIdx,deviceIdxSC,INDX[i].deviceVox,TOTVOX,TOTVOXp,
				NSUBTYPES,NCOVAR,NSUBS,NCOV_FIX,isub,INDX[i].hostN);
			}
		}
		
		//  Compute Bayes Factor
/*		reduce<<<BPG, NN >>>(devicePredict,df_sum,NSUBTYPES*TOTVOX);

		cudaMemcpy(hf_sum,df_sum,BPG*sizeof(float),cudaMemcpyDeviceToHost);
	
		for (int j=0;j<BPG;j++)
			f += (double)hf_sum[j];*/
		//  End Compute Bayes Factor
		
		//  Compute Prediction for each subtype	
		for (int ii=0;ii<NSUBTYPES;ii++) {
			Mpredict[ii] = 0;
			
			reduce_for_Mpredict<<<BPG, NN >>>(&devicePredict[ii*TOTVOX],df_sum,&deviceData[isub*TOTVOX],TOTVOX);
			CUDA_CALL( cudaMemcpy(hf_sum,df_sum,BPG*sizeof(float),cudaMemcpyDeviceToHost) );
			
			for (int j=0;j<BPG;j++)
				Mpredict[ii] += (double)hf_sum[j];
		}
/*		cudaMemcpy(predict,devicePredict,NSUBTYPES*TOTVOX*sizeof(float),cudaMemcpyDeviceToHost);			

		for (int i=0;i<NSUBTYPES;i++)
			Mpredict[i] = 0;
		for (int k=1;k<NDEPTH+1;k++) {
			for (int j=1;j<NCOL+1;j++) {
				for (int i=1;i<NROW+1;i++) {
					int IDX = idx(i,j,k);
					if (msk[IDX]) {
						for (int ii=0;ii<NSUBTYPES;ii++) {
							if (data[isub*TOTVOX+hostIdx[IDX]])
								Mpredict[ii] += log((double)predict[ii*TOTVOX+hostIdx[IDX]] + DBL_MIN);
							else
								Mpredict[ii] += log((1 - (double)predict[ii*TOTVOX+hostIdx[IDX]]) + DBL_MIN);
						}
					}						
				}
			}
		}*/
		int subtype=0;
		for (int i=0;i<NSUBTYPES;i++) {
			if (covar[i*NSUBS + isub]) {
				subtype = i;
				break;
			}
		}
		DIC += Mpredict[subtype];

		if (first) {
			double max = -DBL_MAX + 0.5;
			Qhat[isub][0] = Mpredict[0] - Mpredict[subtype];
			for (int i=1;i<NSUBTYPES;i++) {
				max = ((Mpredict[i] - Mpredict[subtype]) > Qhat[isub][i]) ? (Mpredict[i] - Mpredict[subtype]) : Qhat[isub][i];
				Qhat[isub][i] = log(exp(Qhat[isub][i] - max) + exp((Mpredict[i] - Mpredict[subtype]) - max)) + max;
			}
		}
		else {
			double max = -DBL_MAX + 0.5;
			for (int i=0;i<NSUBTYPES;i++) {
				max = ((Mpredict[i] - Mpredict[subtype]) > Qhat[isub][i]) ? (Mpredict[i] - Mpredict[subtype]) : Qhat[isub][i];
				Qhat[isub][i] = log(exp(Qhat[isub][i] - max) + exp((Mpredict[i] - Mpredict[subtype]) - max)) + max;
			}
		}
	}	
	
	free(Mpredict);	
    	CUDA_CALL( cudaFree(df_sum) );
	free(hf_sum);
	
	return(DIC);
}



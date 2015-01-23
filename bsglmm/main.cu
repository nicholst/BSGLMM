#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h>
#include <float.h>
#include "randgen.h"
#include "binCAR.h"

int PREC=64;
int NSUBS;
int GPU;
int NROW;
int NCOL;
int NDEPTH;
int TOTVOX;
int TOTVOXp;  // total voxels plus boundary
int UPDATE_WM=1;

int *hostIdx;
int *deviceIdx;
int *hostIdxSC;
int *deviceIdxSC;

float *covar;
int NSUBTYPES;
int NCOVAR;
int NCOV_FIX = 0;
int MAXITER = 100000;
int BURNIN  =  50000;
int RESTART;
float *deviceCovar;
float *hostCovar;
float *XXprime;
float *deviceCov_Fix;
float *hostCov_Fix;
float *XXprime_Fix;
float logit_factor;
float t_df;
int MODEL = 1;    // logistic = 0; Probit = 1, t = 2;

//float *deviceChiSqHist;
//int   ChiSqHist_N;

#define IMAGE_DIR "./images"
#define DEFAULT_MASK "mask.nii.gz"
#define DEFAULT_WM "avg152T1_white.nii.gz"

curandState *devStates;
INDEX *INDX;

double M = exp(-PREC*log(2.0));

#define IDX(i,j,k) (i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k)

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) {printf("\nCUDA Error: %s (err_num=%d) \n",cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}

int main (int argc, char * const argv[]) {
	int rtn,cnt,cnt0,cnt1,cntp;
	char *WM_name=(char*)calloc(300,sizeof(char));
	char *mask_name=(char*)calloc(300,sizeof(char));
	unsigned char *data;
	unsigned char *msk,*mskp;
	unsigned long *seed;
	float *WM;
	FILE *fseed;

//	unsigned char *read_nifti1_mask(char *,char *);
	unsigned char *read_nifti1_mask(char *);
	unsigned char *read_nifti1_image(unsigned char *,char *);
	float *read_nifti1_WM(const char *,const unsigned char *);
	float *read_covariates(char *);
	float *read_covariates_fix();
	void mcmc(float *,float *,unsigned char *,float *,unsigned char *,unsigned long *);
	void write_empir_prb(unsigned char *,float *,unsigned char *,int *);
	unsigned char *get_WM_mask(float *,unsigned char *);



	if (argc !=7 && argc !=9) {
		printf("Usage: %s  NTypes  NCov  GPU  Design Mask WM [MaxIter BurnIn]  \n",argv[0]);
		printf("  NTypes  - Number of groups\n");
		printf("  NCov    - Number of covariates (count must include groups)\n");
		printf("  GPU     - 1 use GPU; 0 use CPU (CPU not tested! Use with caution)\n");
		printf("  Design  - Text file, tab or space separated data file\n");
		printf("  Mask    - Filename of mask image in '%s' directory, or '1' to use default (%s)\n",IMAGE_DIR,DEFAULT_MASK);
		printf("  WM      - Filename of WM image in '%s' directory, '0' to use none, or '1' to use default (%s)\n",IMAGE_DIR,DEFAULT_WM);
		printf("  MaxIter - Number of iterations (defaults to 1,000,000)\n");
		printf("  BurnIn  - Number of burn-in iterations (defaults to 500,000)\n");
		printf("For documentation see: http://warwick.ac.uk/tenichols/BSGLMM\n");
		exit(1);
	}

	NSUBTYPES = atoi(argv[1]);
	NCOVAR = atoi(argv[2]);
	GPU = atoi(argv[3]);
	if (strcmp(argv[5],"1")==0)
		sprintf(mask_name,"%s/%s",IMAGE_DIR,DEFAULT_MASK);
	else
		sprintf(mask_name,"%s/%s",IMAGE_DIR,argv[5]);
	if (strcmp(argv[6],"0")==0)
		UPDATE_WM=0;
	else if  (strcmp(argv[6],"1")==0)
		sprintf(WM_name,"%s/%s",IMAGE_DIR,DEFAULT_WM);
	else
		sprintf(WM_name,"%s/%s",IMAGE_DIR,argv[6]);
	if (argc >7)  {
		MAXITER = atoi(argv[7]);
		BURNIN  = atoi(argv[8]);
		if (MAXITER<1000)
		   printf("WARNING: Silly-small number of iterations used; recommend abort and use more\n");
		if (BURNIN>MAXITER)
		   printf("WARNING: Burn-in exceeds MAXITER, no results will be saved\n");
	}


	RESTART = 0;
	int deviceCount = 0;
   	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

   	 if (error_id != cudaSuccess){
        	printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
        	exit(EXIT_FAILURE);
    	}
	if (deviceCount == 0) {
	        printf("There are no available device(s) that support CUDA\n");
	}
	else {
	        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}
	int best = 0;
	int major = 100;
	for (int dev = 0; dev < deviceCount; ++dev) {
        	cudaSetDevice(dev);
        	cudaDeviceProp deviceProp;
        	cudaGetDeviceProperties(&deviceProp, dev);
	        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		printf("Compute capability %d.%d \n",deviceProp.major,deviceProp.minor);
		if (deviceProp.major >= major) {
			major = deviceProp.major;
			best = dev;
		}
	}
       	cudaSetDevice(best);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, best);
	printf("Device used is %s \n\n",deviceProp.name);

	RESTART = 0;

	seed = (unsigned long *)calloc(3,sizeof(unsigned long));
	fseed = fopen("seed.dat","r");
	if (fseed == NULL){
		printf("No seed.dat found; using [ 1 2 3 ]\n");
		seed[0]=1L;seed[1]=2L;seed[2]=2L;
	} else {
		rtn = fscanf(fseed,"%lu %lu %lu\n",&(seed[0]),&(seed[1]),&(seed[2]));
		if (rtn==0)
		  exit(1);
		rtn = fclose(fseed);
		if (rtn != 0)
		  exit(1);
	}


	logit_factor = 1.0f;
	t_df = 1000.0f;
	if (MODEL == 0) {
		logit_factor = 0.634f;
		t_df = 8.0f;
	}
	if (MODEL == 2)
		t_df = 3.0f;


	msk = read_nifti1_mask(mask_name);

	WM = read_nifti1_WM(WM_name,msk);

	data = read_nifti1_image(msk,argv[4]);

	if (RESTART)
		BURNIN = 1;


  	mskp = (unsigned char *)calloc((NROW+2)*(NCOL+2)*(NDEPTH+2),sizeof(unsigned char));
	for (int k=1;k<NDEPTH+1;k++) {
		for (int j=1;j<NCOL+1;j++) {
			for (int i=1;i<NROW+1;i++) {
				int idx1 = i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k;
				if (msk[idx1]) {
					mskp[idx1] = 1;
					mskp[idx1 - 1] = 1;
					mskp[idx1 + 1] = 1;
					mskp[idx1 - (NROW+2)] = 1;
					mskp[idx1 + (NROW+2)] = 1;
					mskp[idx1 - (NROW+2)*(NCOL+2)] = 1;
					mskp[idx1 + (NROW+2)*(NCOL+2)] = 1;
				}
			}
		}
	}

	TOTVOXp = 0;
	for (int k=0;k<NDEPTH+2;k++) {
		for (int j=0;j<NCOL+2;j++) {
			for (int i=0;i<NROW+2;i++) {
				if (mskp[i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k]) TOTVOXp++;
			}
		}
	}

	INDX = (INDEX *)calloc(2,sizeof(INDEX));
 	for (int k=1;k<NDEPTH+1;k++) {
 		for (int j=1;j<NCOL+1;j++) {
 			for (int i=1;i<NROW+1;i++) {
 				if (msk[i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k]) {
 					if (k%2) {
 						if (j%2) {
 							if (i%2)
 								INDX[0].hostN++;
 							else
 								INDX[1].hostN++;
 						}
 						else {
  							if (i%2)
 								INDX[1].hostN++;
 							else
 								INDX[0].hostN++;
						}
 					}
 					else {
						if (j%2) {
  							if (i%2)
 								INDX[1].hostN++;
 							else
 								INDX[0].hostN++;
						}
						else {
 							if (i%2)
 								INDX[0].hostN++;
 							else
 								INDX[1].hostN++;
						}
 					}
 				}
 			}
 		}
 	}
/*	for (int kk=1;kk<3;kk++) {
		for (int jj=1;jj<3;jj++) {
			for (int ii=1;ii<3;ii++) {
				for (int k = kk;k<NDEPTH+1;k+=2) {
					for (int j=jj;j<NCOL+1;j+=2) {
						for (int i=ii;i<NROW+1;i+=2) {
							if (msk[i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k])
								INDX[idx].hostN++;
						}
					}
				}
				idx++;
			}
		}
	}*/

	float  	NSIZE = (NROW+2)*(NCOL+2)*(NDEPTH+2);

	if (GPU) {
		for (int i=0;i<2;i++) {
			INDX[i].hostVox = (int *)calloc(INDX[i].hostN,sizeof(int));
//			cudaHostAlloc((void **)&INDX[i].hostVox, INDX[i].hostN*sizeof(int),cudaHostAllocDefault);
			CUDA_CALL( cudaMalloc((void **)&INDX[i].deviceVox,INDX[i].hostN*sizeof(int)) );
			CUDA_CALL( cudaMemset(INDX[i].deviceVox,0,INDX[i].hostN*sizeof(int)) );
		}
		for	(int i=0;i<2;i++) {
			INDX[i].hostNBRS = (unsigned char *)calloc(INDX[i].hostN,sizeof(unsigned char));
//			cudaHostAlloc((void **)&INDX[i].hostNBRS, INDX[i].hostN*sizeof(unsigned char),cudaHostAllocDefault);
			CUDA_CALL( cudaMalloc((void **)&INDX[i].deviceNBRS,INDX[i].hostN*sizeof(unsigned char)) );
			CUDA_CALL( cudaMemset(INDX[i].deviceNBRS,0,INDX[i].hostN*sizeof(unsigned char)) );
		}
		hostIdx = (int *)calloc(NSIZE,sizeof(int));
//		cudaHostAlloc((void **)&hostIdx, NSIZE*sizeof(int),cudaHostAllocDefault);
		CUDA_CALL( cudaMalloc((void **)&deviceIdx,NSIZE*sizeof(int)) );
		CUDA_CALL( cudaMemset(deviceIdx,0,NSIZE*sizeof(int)) );

		hostIdxSC = (int *)calloc(NSIZE,sizeof(int));
//		cudaHostAlloc((void **)&hostIdxSC, NSIZE*sizeof(int),cudaHostAllocDefault);
		CUDA_CALL( cudaMalloc((void **)&deviceIdxSC,NSIZE*sizeof(int)) );
		CUDA_CALL( cudaMemset(deviceIdxSC,0,NSIZE*sizeof(int)) );

	}
	else {
	     fprintf(stderr,"WARNING!!!\n");
	     fprintf(stderr,"WARNING!!! CPU code not tested!  Results might not be right!\n");
	     fprintf(stderr,"WARNING!!!\n");
		for (int i=0;i<2;i++) {
			INDX[i].hostVox = (int *)calloc(INDX[i].hostN,sizeof(int));
		}
		for	(int i=0;i<2;i++) {
			INDX[i].hostNBRS = (unsigned char *)calloc(INDX[i].hostN,sizeof(unsigned char));
		}
		hostIdx = (int *)calloc(NSIZE,sizeof(int));
		hostIdxSC = (int *)calloc(NSIZE,sizeof(int));
	}
	for (int i=0;i<NSIZE;i++) {
		hostIdx[i] = -1;
		hostIdxSC[i] = -1;
	}

	cnt = 0;
	cntp=0;
	for (int k=0;k<NDEPTH+2;k++) {
		for (int j=0;j<NCOL+2;j++) {
			for (int i=0;i<NROW+2;i++) {
				int iidx = i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k;
				if (msk[iidx]) {
					hostIdx[iidx] = cnt;
					cnt++;
				}
				if (mskp[iidx]) {
					hostIdxSC[iidx] = cntp;
					cntp++;
				}
			}
		}
	}

	void cnt_nbrs(int,int,int *,unsigned char *);

	cnt0 = cnt1 = 0;
 	for (int k=1;k<NDEPTH+1;k++) {
 		for (int j=1;j<NCOL+1;j++) {
 			for (int i=1;i<NROW+1;i++) {
 				int iidx = i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k;
				if (msk[iidx]) {
 					if (k%2) {
 						if (j%2) {
 							if (i%2) {
 								cnt_nbrs(0,iidx,&cnt0,msk);
 //								INDX[0].hostVox[cnt0] = iidx;
 //								cnt0++;
 							}
 							else {
 								cnt_nbrs(1,iidx,&cnt1,msk);
 //								INDX[1].hostVox[cnt1] = iidx;
 //								cnt1++;
 							}
 						}
 						else {
  							if (i%2) {
								cnt_nbrs(1,iidx,&cnt1,msk);
 //								INDX[1].hostVox[cnt1] = iidx;
 //								cnt1++;
 							}
 							else {
 								cnt_nbrs(0,iidx,&cnt0,msk);
 //								INDX[0].hostVox[cnt0] = iidx;
//								cnt0++;
 							}
 						}
 					}
 					else {
						if (j%2) {
   							if (i%2) {
 								cnt_nbrs(1,iidx,&cnt1,msk);
 //								INDX[1].hostVox[cnt1] = iidx;
 //								cnt1++;
 							}
 							else {
								cnt_nbrs(0,iidx,&cnt0,msk);
// 								INDX[0].hostVox[cnt0] = iidx;
//								cnt0++;
 							}
 						}
						else {
 							if (i%2) {
 								cnt_nbrs(0,iidx,&cnt0,msk);
 //								INDX[0].hostVox[cnt0] = iidx;
//								cnt0++;
 							}
  							else {
 								cnt_nbrs(1,iidx,&cnt1,msk);
// 								INDX[1].hostVox[cnt1] = iidx;
// 								cnt1++;
 							}
 						}
 					}
 				}
 			}
 		}
 	}

	int max = 0;
	if (GPU) {
		for (int i=0;i<2;i++)
			max = (max > INDX[i].hostN) ? max:INDX[i].hostN;
//printf("max = %d\n",max);
		for (int i=0;i<2;i++)
			 CUDA_CALL( cudaMemcpy(INDX[i].deviceVox,INDX[i].hostVox,INDX[i].hostN*sizeof(int),cudaMemcpyHostToDevice) );
		for (int i=0;i<2;i++)
			CUDA_CALL( cudaMemcpy(INDX[i].deviceNBRS,INDX[i].hostNBRS,INDX[i].hostN*sizeof(unsigned char),cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(deviceIdx,hostIdx,NSIZE*sizeof(int),cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(deviceIdxSC,hostIdxSC,NSIZE*sizeof(int),cudaMemcpyHostToDevice) );
	}

	if (GPU) {
		__global__ void setup_kernel(curandState *,unsigned long long, const int);
		unsigned long long devseed;
//		devseed = (unsigned long long)runiform_long_n(18446744073709551615ULL,seed);
		devseed = (unsigned long long)runiform_long_n(ULONG_MAX,seed);
		CUDA_CALL( cudaMalloc (( void **) &devStates , max * sizeof ( curandState ) ) );
		setup_kernel<<<512,512 >>>(devStates,devseed,max);
	//	cutilCheckMsg("setup_kernel failed:");
	}

	write_empir_prb(msk,covar,data,hostIdx);

	mcmc(covar,hostCov_Fix,data,WM,msk,seed);

	free(msk);
	free(mskp);
	free(WM);
	free(data);
	free(WM_name);


	fseed = fopen("seed.dat","w");
	fprintf(fseed,"%lu %lu %lu\n",seed[0],seed[1],seed[2]);
	fclose(fseed);
	free(seed);

//	cudaFree(deviceChiSqHist);
	if (GPU) {
		free(hostIdx);
   		cudaFree(deviceIdx);
		free(hostIdxSC);
   		cudaFree(deviceIdxSC);
   		for (int i=0;i<2;i++) {
   			free(INDX[i].hostVox);
   			cudaFree(INDX[i].deviceVox);
   			free(INDX[i].hostNBRS);
   			cudaFree(INDX[i].deviceNBRS);
   		}
 		cudaFree(deviceCovar);
		if (NCOV_FIX > 0) {
			cudaFree(deviceCov_Fix);
	 	}
  	}
   	else {
 		free(covar);
		if (NCOV_FIX > 0)
			free(hostCov_Fix);
  		free(hostIdx);
   		free(deviceIdx);
   		free(hostIdxSC);
   		free(deviceIdxSC);
   		for (int i=0;i<2;i++) {
   			free(INDX[i].hostVox);
   			free(INDX[i].hostNBRS);
   		}
  	}
	free(INDX);

    return 0;
}


void cnt_nbrs(int idx,int iidx,int *cnt,unsigned char *msk)
{
	int tmp;

	INDX[idx].hostVox[*cnt] = iidx;

	tmp = iidx-1;
	if (msk[tmp])
		INDX[idx].hostNBRS[*cnt]++;

	tmp = iidx+1;
	if (msk[tmp])
		INDX[idx].hostNBRS[*cnt]++;

	tmp = iidx - (NROW+2);
	if (msk[tmp])
		INDX[idx].hostNBRS[*cnt]++;

	tmp = iidx + (NROW+2);
	if (msk[tmp])
		INDX[idx].hostNBRS[*cnt]++;

	tmp = iidx - (NROW+2)*(NCOL+2);
	if (msk[tmp])
		INDX[idx].hostNBRS[*cnt]++;

	tmp = iidx + (NROW+2)*(NCOL+2);
	if (msk[tmp])
		INDX[idx].hostNBRS[*cnt]++;

	(*cnt)++;
}

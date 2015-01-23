/*
 *  read_data.cpp
 *  BinCAR
 *
 *  Created by Timothy Johnson on 4/14/12.
 *  Copyright 2012 University of Michigan. All rights reserved.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cuda_runtime.h>
//#include <cutil_inline.h>
#include "nifti1.h"
#include "binCAR.h"

extern float *covar;
extern int GPU;
extern int NSUBS;
extern int NROW;
extern int NCOL;
extern int NDEPTH;
extern int TOTVOX;
extern int TOTVOXp;
extern int NCOVAR;
extern int NCOV_FIX;
extern int NSUBTYPES;
extern int UPDATE_WM;
extern float *deviceCovar;
extern float *deviceCov_Fix;
extern float *XXprime;
extern float *XXprime_Fix;

typedef double MY_DATATYPE;
double *grp_prior;
char **SS;

#define CUDA_CALL(x) {const cudaError_t a = (x); if (a != cudaSuccess) {printf("\nCUDA Error: %s (err_num=%d) \n",cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}

int test_ext(const char *img_name,const char *ext)
{
        char *ptr;
        int Found=0;
	
	ptr = strstr(img_name,ext); 
	if (ptr!=NULL) {
		if (img_name - ptr == strlen(img_name) - strlen(ext))
			Found=1;
	}
	
	return(Found);
}
void nifti_basenm(const char *img_name,char *basenm)
{
        char *ptr;
	
	strcpy(basenm,img_name);
	ptr = strstr(basenm,".gz"); 
	if (ptr!=NULL)
		*ptr=0;
	
	ptr = strstr(basenm,".nii"); 
	if (ptr!=NULL)
		*ptr=0;
	else {
		ptr = strstr(basenm,".img"); 
		if (ptr!=NULL)
			*ptr=0;
		else {
			ptr = strstr(basenm,".hdr"); 
			if (ptr!=NULL)
				*ptr=0; 
		}
	}
}

FILE *nifti_expand(const char *img_name,char *exp_name)
{
	FILE *data;
	int FileType=0;

	if (test_ext(img_name,".nii.gz")) 
		FileType=1;
	if (test_ext(img_name,".nii")) 
		FileType=2;
	if (FileType==0) {
		printf("Image ('%s') is not single file NIFTI (.nii or .nii.gz)\n",img_name);
		exit(1);
	}
	data=fopen(img_name,"r");
	if (data==NULL) {
		printf("Cannot open '%s'\n",img_name);
		exit(1);
	}
	
	if (FileType==1) {
		fclose(data);
		nifti_basenm(img_name,exp_name);
		strcat(exp_name,".nii");

		char *RR = (char *)calloc(300,sizeof(char));

		RR = strcpy(RR,"gunzip ");
		RR = strcat(RR,(const char *)img_name);		
		int rtn = system(RR);
		if (!rtn) {
			printf("Error unzipping file '%s'\n",img_name);
			exit(rtn);
		}
		data=fopen(RR,"r");
		if (data==NULL) {
			printf("Cannot open unzipped file '%s'\n",RR);
			exit(1);
		}
		free(RR);
	} else {
		strcpy(exp_name,img_name);
	}
}
void nifti_compress(const char *img_name,const char *exp_name)
{

	if (strcmp(img_name,exp_name)!=0) {

		char *RR = (char *)calloc(300,sizeof(char));

		RR = strcpy(RR,"gzip ");
		RR = strcat(RR,exp_name);
		int rtn = system(RR);
		if (!rtn) {
			printf("WARNING: Error re-zipping '%s'\n",exp_name);
		}

		free(RR);
	}
}
void itoa(int n,char *s)
{
	int i,sign;
	void reverse(char *s);
	
	if ((sign = n) < 0)
		n = -n;
	i = 0;
	do {
		s[i++] = n % 10 + '0';
	} while ((n /= 10) > 0);
	if (sign < 0)
		s[i++] = '-';
	s[i] = '\0';
	reverse(s);
}

void reverse(char *s)
{
	int c,i,j;
	
	for (i=0,j = strlen(s)-1;i<j;i++,j--) {
		c = s[i];
		s[i] = s[j];
		s[j] = c;
	}
}


unsigned char *read_nifti1_mask(char *mask_name)
{
	int i,j,k,rtn;
	double *image;
	char *RR,*basenm,*tok2;
	unsigned char *mask;
	struct nifti_1_header *nifti_head;
	FILE *data;
	int FileType=0;

	char *mask_name_exp = (char *)calloc(300,sizeof(char));
	data = nifti_expand(mask_name,mask_name_exp);

	nifti_head = (struct nifti_1_header *)malloc(352);
	rtn = fread(nifti_head,352,1,data);

	if (!(nifti_head->datatype == NIFTI_TYPE_FLOAT64)) {
		printf("Incorrect data type in file: %d\n",nifti_head->datatype);
		printf("Must be float (NIFTI_TYPE_FLOAT64)\n");
		exit(0);
	}
	else {
		image = (double *)malloc(nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3] * sizeof(double));
		int ret = fread(image, sizeof(double), nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3], data);
		if (ret != nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3]) {
			printf("\nError reading volume 1 from %s (%d)\n",mask_name,ret);
			free(image);
			exit(0);
		}
	}

	fclose(data);

	nifti_compress(mask_name,mask_name_exp);

	NROW = nifti_head->dim[1];
	NCOL = nifti_head->dim[2];
	NDEPTH = nifti_head->dim[3];

	mask = (unsigned char *)calloc((NROW+2)*(NCOL+2)*(NDEPTH+2),sizeof(unsigned char));

	int t = 0,ii;
	TOTVOX = 0;
	for(k=1; k<NDEPTH+1;k++) {
		for(j=1; j<NCOL+1; j++) {
			for(i=1;i<NROW+1; i++) {
				ii = i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k;
				mask[ii] = (unsigned char)*(image+t);
				t++;
				if (mask[ii]) TOTVOX++;
			}
		}
	}
	
	free(mask_name_exp);
	free(basenm);
	free(image);

	return mask;
}

unsigned char *read_nifti1_image(unsigned char *msk,char *file_name)
{
	int ii,jj,kk,*list;
	int i,j,k,rtn,nrow,ncol,ndepth,cnt,t,subtype_cnt;
	unsigned char *img2;
	float *imagef;
	double *imaged;
	char *S,*T,*TT,*RR,**img_list;
	struct nifti_1_header *nifti_head;
	FILE *data,*fdir,*hdr,*demo;
	int write_nifti_file(int NROW,int NCOL,int NDEPTH,int NTIME,char *hdr_file,char *data_file,MY_DATATYPE *data);

	S = (char *)calloc(500,sizeof(char));
	T = (char *)calloc(500,sizeof(char));
	SS = (char **)calloc(NCOVAR,sizeof(char *));
	for (i=0;i<NCOVAR;i++)
		SS[i] = (char *)calloc(100,sizeof(char));

	demo = fopen(file_name,"r");

	for (i=0;i<(NCOVAR+2);i++) {
		rtn = fscanf(demo,"%s ",S);
	}

	NSUBS = 0;

	while (fscanf(demo,"%s\n",S) != EOF) {
		NSUBS++;
	}
	NSUBS /= NCOVAR+2;

	img_list = (char **)calloc(NSUBS,sizeof(char *));
	for (i=0;i<NSUBS;i++)
		img_list[i] = (char *)calloc(500,sizeof(char));

	rewind(demo);
	
	subtype_cnt = 0;
	for (i=0;i<(NCOVAR+2);i++) {
		rtn = fscanf(demo,"%s ",S);
		if ((i>0) && (i<NCOVAR+1)) {
			SS[i-1] = strcpy(SS[i-1],S);
		}
	}

	cnt = 0;
	int sub_cnt = 0;
	grp_prior = (double *)calloc(NSUBTYPES,sizeof(double));

	covar = (float *)calloc(NSUBS*NCOVAR,sizeof(float));
	while (fscanf(demo,"%s\n",S) != EOF) {
		if ((cnt > 0) && (cnt < NCOVAR+1)) {
			covar[(cnt-1)*NSUBS+sub_cnt] = atof(S);
			if (cnt < NSUBTYPES+1)
				grp_prior[cnt-1] += (int)atof(S);
		}
		if (cnt == NCOVAR+1) {
			T = strcpy(T,"./images/");
			T = strcpy(T,S);
			img_list[sub_cnt] = strcpy(img_list[sub_cnt],T);
		}
		cnt++;
		if (cnt == NCOVAR+2) {
			cnt = 0;
			sub_cnt++;
		}
	}
	fclose(demo);
	for (int i=0;i<NSUBTYPES;i++) {
		grp_prior[i] /= (double)NSUBS;
	}
	XXprime = (float *)calloc(NCOVAR*NCOVAR,sizeof(float));
	for (int isub=0;isub<NSUBS;isub++) {
		for (int i=0;i<NCOVAR;i++) {
			for (int j=0;j<NCOVAR;j++)
				XXprime[j+i*NCOVAR] += covar[i*NSUBS+isub]*covar[j*NSUBS+isub];
		}
	}

	if (GPU) {
		CUDA_CALL( cudaMalloc((void **)&deviceCovar,NCOVAR*NSUBS*sizeof(float)) );
		CUDA_CALL( cudaMemcpy(deviceCovar,covar,NCOVAR*NSUBS*sizeof(float),cudaMemcpyHostToDevice) );
	}

	img2 = (unsigned char *)calloc(NSUBS*TOTVOX,sizeof(unsigned char));

	char *img_nm,*img_nm_exp;
	img_nm = (char *)calloc(500,sizeof(char));
	img_nm_exp = (char *)calloc(500,sizeof(char));
	for (int isub=0;isub<NSUBS;isub++) {

		strcpy(img_nm,img_list[isub]);		

		data = nifti_expand(img_nm,img_nm_exp);

		nifti_head = (struct nifti_1_header *)malloc(352);
		rtn = fread(nifti_head,352,1,data);
		if (!((nifti_head->datatype == NIFTI_TYPE_FLOAT32) || (nifti_head->datatype == NIFTI_TYPE_FLOAT64))) {
			printf("Incorrect data type in file %s: %d\n",img_list[isub],nifti_head->datatype);
			printf("Must be float (NIFTI_TYPE_FLOAT32) or double (NIFTI_TYPE_FLOAT64)\n");
			exit(0);
		}
		if (nifti_head->datatype == NIFTI_TYPE_FLOAT32) {
			imagef = (float *)malloc(nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3] * sizeof(float));
			int ret = fread(imagef, sizeof(float), nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3], data);
			if (ret != nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3]) {
				printf("\nError reading volume 1 from %s (%d)\n",img_list[isub],ret);
				free(imagef);
				exit(0);
			}	
		}
		if (nifti_head->datatype == NIFTI_TYPE_FLOAT64) {
			imaged = (double *)malloc(nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3] * sizeof(double));
			int ret = fread(imaged, sizeof(double), nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3], data);
			if (ret != nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3]) {
				printf("\nError reading volume 1 from %s (%d)\n",img_list[isub],ret);
				free(imaged);
				exit(0);
			}
		}
		fclose(data);
		nifti_compress(img_nm,img_nm_exp);
		int idx2; 
		int cnt2=0;
		cnt = 0;

		for(k=1;k<NDEPTH+1;k++) {
			for(j=1; j<NCOL+1; j++) {
				for(i=1;i<NROW+1; i++) {
					idx2 = i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k;
					if (msk[idx2]) {
						switch (nifti_head->datatype) {
							case NIFTI_TYPE_FLOAT32: default:
								img2[isub*TOTVOX+cnt] = (unsigned char)*(imagef+cnt2);
								break;
							case NIFTI_TYPE_FLOAT64:
								img2[isub*TOTVOX+cnt] = (unsigned char)*(imaged+cnt2);
						}
						cnt++;
					}
					cnt2++;
				}
			}
		}

		switch (nifti_head->datatype) {
			case NIFTI_TYPE_FLOAT32: default:
				free(imagef);
				break;
			case NIFTI_TYPE_FLOAT64:
				free(imaged);
				break;
			}
		free(nifti_head);
	}


	free(img_nm);
	free(img_nm_exp);
	free(S);
	free(T);

	return img2;
}

void write_empir_prb(unsigned char *msk,float *covar,unsigned char *data,int *hostIdx)
{
	void itoa(int,char *);

	int idx;
	char *S,*RR;
	double *sum,N,*sum2,N2;
	FILE *fout;
	
	int write_nifti_file(int NROW,int NCOL,int NDEPTH,int NTIME,char *hdr_file,char *data_file,MY_DATATYPE *data);

	S = (char *)calloc(100,sizeof(char));
	RR = (char *)calloc(500,sizeof(char));

	sum = (double *)calloc(NROW*NCOL*NDEPTH,sizeof(double));
	sum2 = (double *)calloc(NROW*NCOL*NDEPTH,sizeof(double));
	for (int ii=0;ii<NSUBTYPES;ii++) {
	
		int cnt = 0;
		for (int k=1;k<NDEPTH+1;k++) {
			for (int j=1;j<NCOL+1;j++) {
				for (int i=1;i<NROW+1;i++) {
					idx = i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k;
					sum[cnt] = 0;
					if (msk[idx]) {
						N = 0;
						N2 = 0;
						for (int isub=0;isub<NSUBS;isub++) {
							if (covar[ii*NSUBS+isub]) {
								sum[cnt] += data[isub*TOTVOX+hostIdx[idx]];
								N++;
							}
						}
						sum2[cnt] += sum[cnt];
						sum[cnt] /= N;
					}
					cnt++;
				}
			}
		}
		S = strcpy(S,"empir_prb_");
		S = strcat(S,SS[ii]);
		S = strcat(S,".nii");
		int rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,sum);

		RR = strcpy(RR,"gzip -f ");
		RR = strcat(RR,(const char *)S);
		rtn = system(RR);
	}
	S = strcpy(S,"total_lesion_cnt.nii");
	int rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,sum2);

	RR = strcpy(RR,"gzip -f ");
	RR = strcat(RR,(const char *)S);
	rtn = system(RR);

	for (int ii=0;ii<NROW*NCOL*NDEPTH;ii++)
		sum2[ii] /= NSUBS;
	S = strcpy(S,"total_empir_prb.nii");
	rtn = write_nifti_file(NROW,NCOL,NDEPTH,1,S,S,sum2);

	RR = strcpy(RR,"gzip -f ");
	RR = strcat(RR,(const char *)S);
	rtn = system(RR);
	
	free(sum2);
	free(sum);
	free(S);
	free(RR);
}

float *read_nifti1_WM(const char *WM_name,const unsigned char *msk)
{
	int i,j,k,rtn;
	unsigned char *image;
	float *WM;
	struct nifti_1_header *nifti_head;
	FILE *data,*hdr,*qconv;

	char *WM_name_exp = (char *)calloc(300,sizeof(char));
	data = nifti_expand(WM_name,WM_name_exp);

	nifti_head = (struct nifti_1_header *)malloc(352);
	rtn = fread(nifti_head,352,1,data);
	if (!(nifti_head->datatype == NIFTI_TYPE_UINT8)) {
		printf("Incorrect data type in file: %d\n",nifti_head->datatype);
		printf("Must be unsigned char (NIFTI_TYPE_UINT8)\n");
		exit(0);
	}
	if (nifti_head->datatype == NIFTI_TYPE_UINT8) {
		image = (unsigned char *)malloc(nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3] * sizeof(unsigned char));
		int ret = fread(image, sizeof(unsigned char), nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3], data);
		if (ret != nifti_head->dim[1]*nifti_head->dim[2]*nifti_head->dim[3]) {
			printf("\nError reading volume 1 from (%d)\n",ret);
			free(image);
			exit(0);
		}
		
	}
	fclose(data);
	nifti_compress(WM_name,WM_name_exp);

	WM = (float *)calloc(TOTVOX,sizeof(float));

	int t = 0,idx,cnt=0;
	for(k=1; k<NDEPTH+1;k++) {
		for(j=1; j<NCOL+1; j++) {
			for(i=1;i<NROW+1; i++) {
				idx = i + (NROW+2)*j + (NROW+2)*(NCOL+2)*k;
				if (msk[idx]) { 
					if (UPDATE_WM)
						WM[cnt] = (float)(*(image+t))/255.0f;
					else
						WM[cnt] = 1.0f;
					cnt++;
				}
				t++;
			}
		}
	}
	free(image);
	
	return WM;
}

int is_numeric(const char *p) {
     if (*p) {
          char c;
          while ((c=*p++)) {
                if (!isdigit(c) && strcmp(&c,".")) return 0;
          }
          return 1;
      }
      return 0;
}

float *read_covariates(char *file_name)
{
	int i,j,rtn,sub_cnt;
	char *S,*T,*U;
	float *covar,x[NCOVAR];
	FILE *demo;
	
	grp_prior = (double *)calloc(NSUBTYPES,sizeof(double));
	covar = (float *)calloc(NCOVAR*NSUBS,sizeof(float));
	if (GPU) {
	//	cudaHostAlloc((void **)&covar, NCOVAR*NSUBS*sizeof(float),cudaHostAllocDefault);
		CUDA_CALL( cudaMalloc((void **)&deviceCovar,NCOVAR*NSUBS*sizeof(float)) );
//		CUDA_CALL( cudaMalloc((void **)&device_subject_subtype,NSUBS*sizeof(unsigned char)) );
	}
//	cudaHostGetDevicePointer((void **)&deviceCovar,covar,0);
//	cudaMemset(deviceCovar,0,NCOVAR*NSUBS*sizeof(float));
//	cudaThreadSynchronize();
//printf("0\n");fflush(NULL);
	S = (char *)calloc(20,sizeof(char));
	T = (char *)calloc(20,sizeof(char));
	U = (char *)calloc(20,sizeof(char));
	SS = (char **)calloc(NCOVAR,sizeof(char *));
	for (i=0;i<NCOVAR;i++)
		SS[i] = (char *)calloc(20,sizeof(char));
//	SS[0] = strcpy(SS[0],"CIS");
	SS[0] = strcpy(SS[0],"RLRM");
	SS[1] = strcpy(SS[1],"PRP");
	SS[2] = strcpy(SS[2],"SCP");
	SS[3] = strcpy(SS[3],"PRL");

	demo = fopen(file_name,"r");
//	demo = fopen("/home/tdjtdj/projects/MS/demo-data/cov5_RLRM_AGE.dat","r");
//	demo = fopen("/home/tdjtdj/projects/MS/demo-data/covarSansCIS.dat","r");
//	demo = fopen("/home/tdjtdj/projects/MS/demo-data/covar4.dat","r");
	for (i=0;i<NCOVAR-NSUBTYPES+2;i++) {
		rtn = fscanf(demo,"%s",S);
		if ((i > 0) && (i < NCOVAR-NSUBTYPES+1)){
			SS[i+NSUBTYPES-1] = strcpy(SS[i+NSUBTYPES-1],S);
//		printf("%s\n",SS[i+NSUBTYPES-1]);fflush(NULL);
	}
	}
	for (i=0;i<NCOVAR;i++)
		printf("%s\n",SS[i]);
//	rtn = fscanf(demo,"%s ",S);
//	rtn = fscanf(demo,"%s ",S);
//	rtn = fscanf(demo,"%s ",S);
//	rtn = fscanf(demo,"%s ",S);
	sub_cnt = 0;
//	while (fscanf(demo,"%d %s\n",&i,S) != EOF) {
//	while (fscanf(demo,"%d %f %s\n",&i,&x[0],S) != EOF) {
//	while (fscanf(demo,"%d %f %f %f %s\n",&i,&x[0],&x[1],&x[2],S) != EOF) {
//	while (fscanf(demo,"%d %f %f %f %f %f %s\n",&i,&x[0],&x[1],&x[2],&x[3],&x[4],S) != EOF) {
	while (fscanf(demo,"%d %f %f %f %f %f %f %f %f %f %f %f %s\n",&i,&x[0],&x[1],&x[2],&x[3],&x[4],&x[5],&x[6],&x[7],&x[8],&x[9],&x[10],S) != EOF) {
//	while (fscanf(demo,"%d %s %f %f %f %f %f %f %f %f %f %f %f %s\n",&i,&x[0],&x[1],&x[2],&x[3],&x[4],&x[5],&x[6],&x[7],&x[8],&x[9],&x[10],&x[11],S) != EOF) {
	
//		if (!strcmp(S,"CIS"))
			sub_cnt++;
	}
	fclose(demo);
	demo = fopen(file_name,"r"); 
	for (i=0;i<NCOVAR-NSUBTYPES+2;i++) {
		rtn = fscanf(demo,"%s ",S);
	}

	float mean[NCOVAR];
	for (int i=0;i<NCOVAR;i++)
		mean[i] = 0;
	
	char **typelist;
	typelist = (char **)calloc(NSUBTYPES,sizeof(char *));
	for (int i=0;i<NSUBTYPES;i++)
		typelist[i] = (char *)calloc(10,sizeof(char));
//	typelist[0] = strcpy(typelist[0],"CIS");
	typelist[0] = strcpy(typelist[0],"RLRM");
	typelist[1] = strcpy(typelist[1],"PRP");
	typelist[2] = strcpy(typelist[2],"SCP");
	typelist[3] = strcpy(typelist[3],"PRL");
	sub_cnt = 0;

	unsigned char *sub_subt;
	sub_subt = (unsigned char *)calloc(NSUBS,sizeof(unsigned char *));
//printf("OK %d\n",demo);fflush(NULL);
//	while (fscanf(demo,"%d %s\n",&j,S) != EOF) {
//	while (fscanf(demo,"%d %f %s\n",&j,&x[0],S) != EOF) {
//	while (fscanf(demo,"%d %f %f %f %s\n",&i,&x[0],&x[1],&x[2],S) != EOF) {
//	while (fscanf(demo,"%d %f %f %f %f %f %s\n",&i,&x[0],&x[1],&x[2],&x[3],&x[4],S) != EOF) {
	while (fscanf(demo,"%d %f %f %f %f %f %f %f %f %f %f %f %s\n",&i,&x[0],&x[1],&x[2],&x[3],&x[4],&x[5],&x[6],&x[7],&x[8],&x[9],&x[10],S) != EOF) {
//	while (fscanf(demo,"%d %s %f %f %f %f %f %f %f %f %f %f %f %s\n",&i,&x[0],&x[1],&x[2],&x[3],&x[4],&x[5],&x[6],&x[7],&x[8],&x[9],&x[10],&x[11],S) != EOF) {
//		if (is_numeric(T) || !is_numeric(U)) { printf("wrong data\n"); exit(0);}
//		else {
//			if (!strcmp(S,"CIS")) {
//				printf("sub_cnt = %d %f %s\n",sub_cnt,x[0],S);
				//int i;
				for (i=0;i<NSUBTYPES;i++) {
					if (!strcmp(S,typelist[i])) {
//						covar[sub_cnt*NCOVAR+i] = 1;
						covar[i*NSUBS+sub_cnt] = 1;
						sub_subt[sub_cnt] = i;
						grp_prior[i] += 1;
					}
					else
//						covar[sub_cnt*NCOVAR+i] = 0;
						covar[i*NSUBS+sub_cnt] = 0;
				}
				i = NSUBTYPES;
				/*if (!strcmp(T,"F"))
					covar[sub_cnt*NCOVAR+i] = 0;
				else
					covar[sub_cnt*NCOVAR+i] = 1;

				i += 1;*/
				for (int ii=i;ii<NCOVAR;ii++) {
					covar[ii*NSUBS + sub_cnt] = x[ii-i];
//					covar[sub_cnt*NCOVAR + ii] = (x[ii-i] > 0);
//					mean[ii] += covar[ii*NSUBS+sub_cnt];
				}
				sub_cnt++;
	
//			}
//		}
	}
/*	grp_prior[0] = 0.8;
	grp_prior[1] = 0.05;
	grp_prior[2] = 0.10;
	grp_prior[3] = 0.05;*/
	for (int i=0;i<NSUBTYPES;i++) {
		grp_prior[i] /= (double)NSUBS;
		printf("Proportion of Subjects of MS subtype %s = %lf\n",typelist[i],grp_prior[i]);
	}
//printf("OK-2\n");fflush(NULL);
	for (int i=0;i<NSUBTYPES;i++) 
		free(typelist[i]);
	free(typelist);

/*	for (int i=0;i<NCOVAR-NSUBTYPES;i++)
		mean[NSUBTYPES+i] /= (float)NSUBS;
 	for (int isub=0;isub<NSUBS;isub++) {
		for (int i=0;i<NCOVAR-NSUBTYPES;i++) {
			covar[(NSUBTYPES+i)*NSUBS+isub] -= mean[NSUBTYPES+i];
		}
	}
	*/

/*	for (int i=0;i<NCOVAR-NSUBTYPES;i++)
		mean[NSUBTYPES+i] /= (float)NSUBS;
	for (int isub=0;isub<NSUBS;isub++) {
		for (int i=0;i<NCOVAR-NSUBTYPES;i++) {
			covar[isub*NCOVAR + (NSUBTYPES+i)] -= mean[NSUBTYPES+i];
		}
	}*/

	XXprime = (float *)calloc(NCOVAR*NCOVAR,sizeof(float));
	for (int isub=0;isub<NSUBS;isub++) {
		for (int i=0;i<NCOVAR;i++) {
			for (int j=0;j<NCOVAR;j++)
//				XXprime[j+i*NCOVAR] += covar[isub*NCOVAR+i]*covar[isub*NCOVAR+j];
				XXprime[j+i*NCOVAR] += covar[i*NSUBS+isub]*covar[j*NSUBS+isub];
		}
	}
/*	for (int i=0;i<NCOVAR;i++) {
		for (int j=0;j<NCOVAR;j++)
			printf("%0.4lf ",XXprime[j+i*NCOVAR]);
		printf("\n");
	}
	exit(0);*/
/*	for (int isub=0;isub<NSUBS;isub++) {
		covar[isub*NCOVAR + (NSUBTYPES)] += 1;
		for (int i=2;i<NCOVAR-NSUBTYPES;i++) {
			covar[isub*NCOVAR + (NSUBTYPES+i)] += 1;
		}
	}*/	
	
	free(S);
	free(T);
	fclose(demo);
	
/*for (int isub=0;isub<NSUBS;isub++) {
		for (int icov=0;icov<NCOVAR;icov++)
			printf("%f ",covar[icov*NSUBS+isub]);
		printf("\n");
	}	
exit(0);*/
//printf("hmm\n");fflush(NULL);
	if (GPU) {
		CUDA_CALL( cudaMemcpy(deviceCovar,covar,NCOVAR*NSUBS*sizeof(float),cudaMemcpyHostToDevice) );
//		CUDA_CALL( cudaMemcpy(device_subject_subtype,sub_subt,NSUBS*sizeof(unsigned char),cudaMemcpyHostToDevice) );
	}
	free(sub_subt);
//printf("damn\n");fflush(NULL);
	return(covar);
}

float *read_covariates_fix()
{
	int i,rtn,sub_cnt;
	char *S,*T,*U;
	float *covar,x[10];
	FILE *demo;
	
	covar = (float *)calloc(NCOV_FIX*NSUBS,sizeof(float));
	if (GPU) {
//		cudaHostAlloc((void **)&covar, NCOV_FIX*NSUBS*sizeof(float),cudaHostAllocDefault);
		CUDA_CALL( cudaMalloc((void **)&deviceCov_Fix,NCOV_FIX*NSUBS*sizeof(float)) );
	}
//	cudaHostGetDevicePointer((void **)&deviceCovar,covar,0);
//	cudaMemset(deviceCovar,0,NCOVAR*NSUBS*sizeof(float));
//	cudaThreadSynchronize();

	S = (char *)calloc(20,sizeof(char));
	T = (char *)calloc(20,sizeof(char));
	U = (char *)calloc(20,sizeof(char));

	demo = fopen("/home/tdjtdj/projects/MS/demo-data/cov1_fix.dat","r");
//	demo = fopen("/home/tdjtdj/projects/MS/demo-data/covarSansCIS.dat","r");
//	demo = fopen("/home/tdjtdj/projects/MS/demo-data/covar4.dat","r");
	for (i=0;i<NCOV_FIX+1;i++) {
		rtn = fscanf(demo,"%s ",S);
	}
	sub_cnt = 0;
	while (fscanf(demo,"%d %f\n",&i,&x[0]) != EOF) {
			sub_cnt++;
	}

	rewind(demo);

	for (i=0;i<NCOV_FIX+1;i++) {
		rtn = fscanf(demo,"%s ",S);
	}
	float mean[NCOV_FIX];
	for (int i=0;i<NCOV_FIX;i++)
		mean[i] = 0;
	
	sub_cnt = 0;

	
	while (fscanf(demo,"%d %f\n",&i,&x[0]) != EOF) {
		for (int ii=0;ii<NCOV_FIX;ii++) {
			covar[sub_cnt*NCOV_FIX + ii] = x[ii];
			mean[ii] += covar[sub_cnt*NCOV_FIX + ii];
		}
		sub_cnt++;
	}

	for (int i=0;i<NCOV_FIX;i++)
		mean[i] /= (float)NSUBS;
	for (int isub=0;isub<NSUBS;isub++) {
		for (int i=0;i<NCOV_FIX;i++) {
			covar[isub*NCOV_FIX + i] -= mean[i];
			if (covar[isub*NCOV_FIX + i] == 0) covar[isub*NCOV_FIX + i] = 0.001;
		}
	}
	

	XXprime_Fix = (float *)calloc(NCOV_FIX*NCOV_FIX,sizeof(float));
	for (int isub=0;isub<NSUBS;isub++) {
		for (int i=0;i<NCOV_FIX;i++)
			for (int j=0;j<NCOV_FIX;j++)
				XXprime_Fix[j+i*NCOV_FIX] += covar[isub*NCOV_FIX + i]*covar[isub*NCOV_FIX + j];
	}
	
	free(S);
	free(T);
	fclose(demo);
	
/*	for (int isub=0;isub<NSUBS;isub++) {
		for (int icov=0;icov<NCOV_FIX;icov++)
			printf("%f ",covar[isub*NCOV_FIX+icov]);
		printf("\n");
	}*/
//	exit(0);

	if (GPU) {
		CUDA_CALL( cudaMemcpy(deviceCov_Fix,covar,NCOV_FIX*NSUBS*sizeof(float),cudaMemcpyHostToDevice) );
	}

	return(covar);
}




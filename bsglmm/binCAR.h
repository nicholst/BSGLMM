#define blocks_per_grid  512
#define threads_per_block  512


typedef struct subdata{
	float *hostCovar;
	float *deviceCovar;
	float *hostZ;
	float *deviceZ;
} SUB;

typedef struct idxtype{
	int *hostVox;
	int *deviceVox;
	unsigned char *hostNBRS;
	unsigned char *deviceNBRS;
	int hostN;
	int devN;
} INDEX;



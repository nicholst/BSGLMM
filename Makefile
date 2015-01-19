################################################################################
#
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# include /home/tdjtdj/NVIDIA_CUDA-5.5_Samples/common/findcudalib.mk


# OS Name (Linux or Darwin)                                                                                                        
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform                                                                                     
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings                                                                                           
ifeq ($(i386),1)
        OS_SIZE = 32
        OS_ARCH = i686
endif

ifeq ($(x86_64),1)
        OS_SIZE = 64
        OS_ARCH = x86_64
endif


# Location of the CUDA Toolkit
CUDA_PATH       ?= /usr/local/cuda-5.0
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin


# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++


# internal flags
NVCCFLAGS   := -m${OS_SIZE}
CCFLAGS     := -O3
NVCCLDFLAGS :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

# OS-specific build flags
ifneq ($(DARWIN),) 
  LDFLAGS += -rpath $(CUDA_PATH)/lib
  CCFLAGS += -arch $(OS_ARCH) $(STDLIB)
else
  ifeq ($(OS_ARCH),armv7l)
    ifeq ($(abi),gnueabi)
      CCFLAGS += -mfloat-abi=softfp
    else
      # default to gnueabihf
      override abi := gnueabihf
      LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
      CCFLAGS += -mfloat-abi=hard
    endif
  endif
endif

ifeq ($(ARMv7),1)
NVCCFLAGS += -target-cpu-arch ARM
ifneq ($(TARGET_FS),) 
CCFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)
endif
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../../common/inc 
LIBRARIES := 

################################################################################

# CUDA code generation flags
ifneq ($(OS_ARCH),armv7l)
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
endif
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30)

################################################################################

# Target rules
all: build

build: BinCar

main.o: main.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

randgen.o: randgen.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

read_data.o: read_data.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mcmc.o: mcmc.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

covar.o: covar.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

nifti1_read_write.o: nifti1_read_write.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

cholesky.o: cholesky.cpp 
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

covarGPU.o: covarGPU.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -o $@ -c $<

BinCar: main.o mcmc.o randgen.o read_data.o covar.o nifti1_read_write.o cholesky.o covarGPU.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES) 
	cp $@ $(HOME)/bin/

run: build
	./BinCar

clean:
	rm -f BinCarGPU main.o mcmc.o randgen.o read_data.o covar.o nifti1_read.write.o cholesky.o covarGPU.o
	#rm -rf ../../bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))/matrixMul

clobber: clean

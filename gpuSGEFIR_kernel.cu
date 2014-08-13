#ifndef _GPUSGEFIR_KERNEL_H_
#define _GPUSGEFIR_KERNEL_H_

#include <stdio.h>

#define unrolled_loop(_)\
	_(0);	_(1);				\
	_(2);	_(3);				\
	_(4);	_(5);				\
	_(6);	_(7)


__device__ __constant__ float H[COEFFICIENT_TAPS];

__global__ void gpuSGEFIR_opt (float* X, float* Y, int nChannelCount, int nSamplePoint)
//__global__ void gpuSGEFIR_opt (float* H, float* X, float* Y, int nChannelCount, int nSamplePoint)
{
    __shared__ float hTile[BLOCK_SIZE_UNROLL][BLOCK_SIZE_POINT];
	__shared__ float xTile[BLOCK_SIZE_POINT];

	#define op_decl(__)\
		float out##__##0 = 0.0f;\
		float out##__##1 = 0.0f
	unrolled_loop (op_decl);

	for (int k_slow =0; k_slow < nSamplePoint; k_slow +=BLOCK_SIZE_POINT) {
		
        __syncthreads();
		if(threadIdx.x*SIMULTANEONUS_READ < BLOCK_SIZE_POINT){ 
        	for(int y=0; y<SIMULTANEONUS_UNROLL; y++){
				int hTile_y = threadIdx.y*SIMULTANEONUS_UNROLL + y;
				int hTile_x = threadIdx.x*SIMULTANEONUS_READ;

				int H_y = blockIdx.y*BLOCK_SIZE_UNROLL + hTile_y;

				int H_x = k_slow + hTile_x;

				int H_index = H_y - H_x;

				if(H_index >= 0  && H_index< COEFFICIENT_TAPS)
				{
					hTile[hTile_y][hTile_x] =  H[H_index];
					hTile[hTile_y][hTile_x+1] = (H_index == 0)? 0 : H[H_index-1];
				}
				else
				{
					hTile[hTile_y][hTile_x] = 0;
					hTile[hTile_y][hTile_x+1] = (H_index == COEFFICIENT_TAPS) ? H[H_index-1] : 0 ;
				}
				
			}
		}
        __syncthreads();

		//* Perform calculations
		float a;
		int X_y, X_x;
		float2 b2, b1;

		for (int k=0; k<BLOCK_SIZE_POINT; k+=1){

			X_y = k_slow + k;
			X_x = BLOCK_SIZE_CHANNEL*blockIdx.x + SIMULTANEONUS_READ*threadIdx.x;
			b1 = *((float2*)&(X[X_y*nChannelCount + X_x]));

			#define op_compute(__)\
				a = hTile[threadIdx.y*SIMULTANEONUS_UNROLL + __][k];\
				out##__##0 += b1.x * a;\
				out##__##1 += b1.y * a
			unrolled_loop (op_compute);
		}
    }


	int Y_x, Y_y;
	#define op_save(__)																\
    	Y_y = blockIdx.y*BLOCK_SIZE_UNROLL + threadIdx.y*SIMULTANEONUS_UNROLL + __;					\
		Y_x = blockIdx.x*BLOCK_SIZE_CHANNEL + threadIdx.x*SIMULTANEONUS_READ;						\
		Y[Y_y*nChannelCount + Y_x] = out##__##0;												\
		Y[Y_y*nChannelCount + Y_x+1] = out##__##1
	unrolled_loop (op_save);
}


#endif // #ifndef _GPUSGEFIR_KERNEL_H_

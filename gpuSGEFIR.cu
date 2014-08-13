// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <windows.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "gpuSGEFIR.h"
#include "gpuSGEFIR_kernel.cu"


void cpuFIR_DirectII_MULCH(float *x,float *h,float *y,int n,int m,int c);

void exeCalFIR(int ,int);
void genSamplePoint(float*, int,int);
void genCoeffient(float*, int);
void exeTranspose(float*,int,int);
void printDiff(float*, float*, int, int);


#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif

int
main(int argc, char** argv)
{
    if(!InitCUDA()) {
		return ;
	}

	int nSamplePoint   =  SAMPLEPOINT;  
	int nChannelCount  =  CHANNELCOUNT;

	exeCalFIR(nSamplePoint, nChannelCount);

    CUT_EXIT(argc, argv);
}



void
exeCalFIR(int nSamplePoint,int nChannelCount)
{
	int size_H = COEFFICIENT_TAPS;
	int size_X = nSamplePoint*nChannelCount;
	int size_Y = nSamplePoint*nChannelCount;
	int mem_size_H = size_H *sizeof(float);
	int mem_size_X = size_X *sizeof(float);
	int mem_size_Y = size_Y *sizeof(float);

    // allocate host memory for matrices A and B
    float* h_H = (float*) malloc (mem_size_H);
    float* h_X = (float*) malloc (mem_size_X);
    float* h_Y = (float*) malloc (mem_size_Y);

    // initialize host memory

	genCoeffient(h_H, size_H);
	genSamplePoint(h_X, nSamplePoint,nChannelCount);


    // allocate device memory
    //float* d_H;
    //CUDA_SAFE_CALL(cudaMalloc((void**) &d_H, mem_size_H));
    float* d_X;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_X, mem_size_X));
    float* d_Y;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_Y, mem_size_Y));
	
    // copy host memory to device
    //CUDA_SAFE_CALL(cudaMemcpy(d_H, h_H, mem_size_H, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(H, h_H, mem_size_H) );
    CUDA_SAFE_CALL(cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice) );
   
	//set kernel parameters
	dim3 threads(BLOCK_SIZE_CHANNEL/SIMULTANEONUS_READ, BLOCK_SIZE_UNROLL/SIMULTANEONUS_UNROLL);
	dim3 grid(nChannelCount/BLOCK_SIZE_CHANNEL, nSamplePoint/BLOCK_SIZE_UNROLL);


	//warmup
		//gpuSGEFIR_opt<<<grid, threads>>>(d_H, d_X, d_Y, nChannelCount, nSamplePoint);
		gpuSGEFIR_opt<<<grid, threads>>>(d_X, d_Y, nChannelCount, nSamplePoint);
		cudaThreadSynchronize();

	///再拷贝一遍数据，为了方便计算时间
	double gpustart = (double)clock();
    // copy host memory to device
    //CUDA_SAFE_CALL(cudaMemcpy(d_H, h_H, mem_size_H, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(H, h_H, mem_size_H) );
    CUDA_SAFE_CALL(cudaMemcpy(d_X, h_X, mem_size_X, cudaMemcpyHostToDevice) );
 
		//create and start timer
		unsigned int timer = 0;
		cutCreateTimer(&timer);
		cutStartTimer(timer);
	//execute
		double const ops = 2.0e11;
		int const reps = (int)(ops/(2.0*nSamplePoint*nChannelCount*nSamplePoint));
		for(int i=0; i< reps; i++)
			gpuSGEFIR_opt<<<grid, threads>>>(d_X, d_Y, nChannelCount, nSamplePoint);
			//gpuSGEFIR_opt<<<grid, threads>>>(d_H, d_X, d_Y, nChannelCount, nSamplePoint);
		cudaThreadSynchronize();
    //stop and destroy timer
		cutStopTimer(timer);
		float duration = cutGetTimerValue(timer);
		cutDeleteTimer(timer);
		CUDA_SAFE_CALL(cudaMemcpy(h_Y, d_Y, mem_size_Y, cudaMemcpyDeviceToHost) );

		double gpufinish=(double)clock();

		//printf("\n");
		//printf("Total size:  %d MB  %.1f Billion calculations\n", (mem_size_H+mem_size_X+mem_size_Y)/1024/1024, (2.0*nSamplePoint*nChannelCount*nSamplePoint)/1e9);
		//printf("Iterations: %d\n", reps);
		//printf("Average processing time: %f (ms) \n", duration / reps );
		printf("Point:%d, Channel:%d ,Taps:%d \n",nSamplePoint,nChannelCount,COEFFICIENT_TAPS);
		printf("Gpu total time: %f (ms), kernel time:%.2fms, %f (GFLOPS) \n", (gpufinish-gpustart) - duration+ duration / reps,duration / reps,(2.*nSamplePoint*nChannelCount*nSamplePoint)/duration/1e6 *reps );

		if (1) {
		
		float* reference = (float*) malloc(mem_size_Y);
		double start=(double)clock();
		float iter = 1.0f;
		for(int r=0;r<iter;r++)
			cpuFIR_DirectII_MULCH(h_X,h_H, reference,nSamplePoint, COEFFICIENT_TAPS, nChannelCount);
		double finish=(double)clock();
		CUTBoolean res = cutCompareL2fe(reference, h_Y, size_Y, 1e-6f);
		printf("Cpu exetime:%.2fms,  %f (GFLOPS) \n",(finish-start)/iter,(2.*nSamplePoint*nChannelCount*COEFFICIENT_TAPS)/(finish-start)/1e6 *iter);
		printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");
		/*FILE *fo;
		fo = fopen("Out_result.txt", "wb");
		for(int row=0;row<nSamplePoint;row++)
		{
			for(int col=0;col<nChannelCount;col++)
				fprintf(fo,"%f  ",h_Y[row*nChannelCount +col]); 
			fprintf(fo,"\n"); 
		}
		fclose(fo);*/

		FILE *flog;
		flog = fopen("Out_log.txt", "ab+");
		//fseek ( flog ,0 , SEEK_END );
		fprintf(flog,"Point:%d, Channel:%d ,Taps:%d \n",nSamplePoint,nChannelCount,COEFFICIENT_TAPS);
		fprintf(flog,"Gpu total time: %f (ms), kernel time:%.2fms, %f (GFLOPS) \n", (gpufinish-gpustart) - duration + duration / reps,duration / reps,(2.*nSamplePoint*nChannelCount*nSamplePoint)/duration/1e6 *reps);  
		fprintf(flog,"Cpu exetime:%.2fms,  %f (GFLOPS) \n",(finish-start)/iter,(2.*nSamplePoint*nChannelCount*COEFFICIENT_TAPS)/(finish-start)/1e6 *iter);  
		fprintf(flog,"\n");
		fclose(flog);

		free(reference);

	}
	
    // clean up memory
   // CUDA_SAFE_CALL(cudaFree(d_H));
    CUDA_SAFE_CALL(cudaFree(d_X));
    CUDA_SAFE_CALL(cudaFree(d_Y));
    free(h_H);
    free(h_X);
    free(h_Y);
}

void genSamplePoint(float* data, int point,int channel)
{
	///按行生成通道，1行1个通道
	for(int i=0;i<point*channel;i++)
		data[i]= 0;
	for(int j=0;j<point;j++)
		data[j]= j;
	///转置
	exeTranspose(data,channel,point);
}
void genCoeffient(float* data, int size)
{
    for (int i = 0; i < size; i++)
        data[i] = 0.001f;
}
void exeTranspose(float* x,int row,int col)
{
	int   i,j; 
	float *y; 
	y=(float *)malloc(row*col*sizeof(float)); 

	for(i=0;i <row;i++) 
		for(j=0;j <col;j++) 
			y[j*row+i]=x[i*col+j];   

	memcpy(x,y,row*col*sizeof(float));

	free(y);

}
void printDiff(float *data1, float *data2, int width, int height)
{
  int x, y, k;
  int error_count=0;
  for (y=0; y<height; y++) {
    for (x=0; x<width; x++) {
      k = y*width+x;
      //if (data1[k] - data2[k] > 1e-2f) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f \n", x,y, data1[k], data2[k]);
		 // printf("diff(%d,%d) \t", x,j);
         error_count++;
      //}
    }
	 printf("\n");
  }
  printf("\nTotal Errors = %d \n", error_count);
}


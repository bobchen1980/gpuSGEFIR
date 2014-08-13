#include <stdlib.h>
#include <stdio.h>
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void cpuFIR_DirectII(float *x,float *h,float *y,int n,int m);
void
cpuFIR_DirectII_MULCH(float *x,float *h,float *y,int n,int m,int c);
////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param x          reference data, computed but preallocated
//! @param h          filter coefficient
//! @param y          ouput data
//! @param n          input data length
//! @param m          coefficient taps
////////////////////////////////////////////////////////////////////////////////
void
cpuFIR_DirectII(float *x,float *h,float *y,int n,int m)
{
	for(int i=0;i<n;i++)
	{
		float sum = 0.0f;
		for(int k=0;k<m;k++) 
			if (i-k >=0)
				sum+= h[k] * x[i-k];
		y[i]=sum;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param x          reference data, computed but preallocated
//! @param h          filter coefficient
//! @param y          ouput data
//! @param n          input data length
//! @param m          coefficient taps
//! @param c          channel count
//! X按列存储，1个通道存1列,Y按行存储
////////////////////////////////////////////////////////////////////////////////
void
cpuFIR_DirectII_MULCH(float *x,float *h,float *y,int n,int m,int c)
{
	float *_x,*_y;
	_x = (float *)malloc(sizeof(float)*n);
	_y = (float *)malloc(sizeof(float)*n);
	for(int ch=0;ch<c;ch++)
	{
		for(int i=0;i<n;i++)
			_x[i] = x[i*c+ch];

		cpuFIR_DirectII(_x,h,_y,n,m);

		for(int j=0;j<n;j++)
			y[j*c +ch] =_y[j];
	}
	
	free(_x);
	free(_y);
}

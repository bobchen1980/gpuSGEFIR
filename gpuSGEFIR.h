#ifndef _GPUSEFIR_H_
#define _GPUSEFIR_H_

// Thread block size
#define SIMULTANEONUS_READ		2 
#define SIMULTANEONUS_UNROLL	8 

#define BLOCK_SIZE_UNROLL	8

#define BLOCK_SIZE_CHANNEL	256  //通道分块数
#define BLOCK_SIZE_POINT	128  //采样点数分块


#define COEFFICIENT_TAPS   256  //滤波器阶数   //128
/////可以改变的参数
#define	SAMPLEPOINT  1024*8
#define CHANNELCOUNT  1024*2

#endif // _GPUSEFIR_H_



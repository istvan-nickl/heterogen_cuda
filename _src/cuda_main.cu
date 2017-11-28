#include <stdio.h>   
#include "time.h"
#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"

#include "defs.h"
#include "func.h"

__constant__ int filter_laplace[5][5] = {-1, -1, -1, -1, -1,
                                         -1, -1, -1, -1, -1,
                                         -1, -1, 24, -1, -1,
                                         -1, -1, -1, -1, -1,
                                         -1, -1, -1, -1, -1};

__constant__ float filter_laplace_f[5][5] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                             -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                             -1.0f, -1.0f, 24.0f, -1.0f, -1.0f,
                                             -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                             -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};


// Globális memóriát használó (triviális :)) megoldás
__global__ void kernel_conv_global(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = threadIdx.x + blockDim.x * blockIdx.x;   // a szál melyik oszlopban levõ kimeneti pixelt számolja
  
  // konvolúció 3 komponensre
  int acc_r = 0, acc_g = 0, acc_b = 0;

  for (int a = 0; a < FILTER_H; a++){
	  for (int b = 0; b < FILTER_W; b++) {
		  acc_r += gInput[((row + a) *imgWidthF + col + b) * 3] * filter_laplace[a][b];
		  acc_g += gInput[((row + a) *imgWidthF + col + b) * 3 + 1] * filter_laplace[a][b];
		  acc_b += gInput[((row + a) *imgWidthF + col + b) * 3 + 2] * filter_laplace[a][b];
	  }
  }

  // kimenet írása
  if (acc_r > 255) gOutput[(row*imgWidth + col) * 3] = 255;
  else if (acc_r < 0) gOutput[(row*imgWidth + col) * 3] = 0;
  else gOutput[(row*imgWidth + col) * 3] = acc_r;

  if (acc_g > 255) gOutput[(row*imgWidth + col) * 3 + 1] = 255;
  else if (acc_g < 0) gOutput[(row*imgWidth + col) * 3 + 1] = 0;
  else gOutput[(row*imgWidth + col) * 3 + 1] = acc_g;

  if (acc_b > 255) gOutput[(row*imgWidth + col) * 3 + 2] = 255;
  else if (acc_b < 0) gOutput[(row*imgWidth + col) * 3 + 2] = 0;
  else gOutput[(row*imgWidth + col) * 3 + 2] = acc_b;

}

// Shared memóriát használó megoldás (1.)
// Shared memória adattípus: unsigned char
// Számítás adattípusa: integer
__global__ void kernel_conv_sh_uchar_int(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = threadIdx.x + blockDim.x * blockIdx.x;   // a szál melyik oszlopban levõ kimeneti pixelt számolja
	int base = (blockIdx.y * blockDim.y * imgWidthF + blockIdx.x * blockDim.x) * 3;


	// Sharde Memory deklaráció

	__shared__ unsigned char mem[20 * 20 * 3];

	// Shared Memory feltöltés
	int th1D = blockDim.x * threadIdx.y + threadIdx.x;	// lieáris szál-azonosító a Thread Block-on belül
	if (th1D < 240)
		for (int d = 0; d < 5; d++) mem[th1D + 240 * d] = gInput[base + th1D%60 + (4 * d + th1D / 60) * imgWidthF * 3];



	// Szál szinkronizáció
	__syncthreads();


	// konvolúció 3 komponensre
	int acc_r = 0, acc_g = 0, acc_b = 0;

	for (int a = 0; a < FILTER_H; a++){
		for (int b = 0; b < FILTER_W; b++) {
			acc_r += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3] * filter_laplace[a][b];
			acc_g += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3 + 1] * filter_laplace[a][b];
			acc_b += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3 + 2] * filter_laplace[a][b];
		}
	}

	// kimenet írása közvetlenül a globális memóriába
	if (acc_r > 255)
		gOutput[(row*imgWidth + col) * 3] = 255;
	else if (acc_r < 0)
		gOutput[(row*imgWidth + col) * 3] = 0;
	else
		gOutput[(row*imgWidth + col) * 3] = acc_r;

	if (acc_g > 255)
		gOutput[(row*imgWidth + col) * 3 + 1] = 255;
	else if (acc_g < 0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 1] = acc_g;

	if (acc_b > 255)
		gOutput[(row*imgWidth + col) * 3 + 2] = 255;
	else if (acc_b < 0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 2] = acc_b;

}

// Shared memóriát használó megoldás (2.)
// Shared memória adattípus: unsigned char
// Számítás adattípusa: float
// A töltés tömb indexelés helyett pointer + offset megoldással
__global__ void kernel_conv_sh_uchar_float(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = threadIdx.x + blockDim.x * blockIdx.x;   // a szál melyik oszlopban levõ kimeneti pixelt számolja
	int base = (blockIdx.y * blockDim.y * imgWidthF + blockIdx.x * blockDim.x) * 3;


	// Sharde Memory deklaráció

	__shared__ unsigned char mem[20 * 20 * 3];

	// Shared Memory feltöltés
	int th1D = blockDim.x * threadIdx.y + threadIdx.x;	// lieáris szál-azonosító a Thread Block-on belül
	if (th1D < 240)
		for (int d = 0; d < 5; d++) mem[th1D + 240 * d] = gInput[base + th1D % 60 + (4 * d + th1D / 60) * imgWidthF * 3];



	// Szál szinkronizáció
	__syncthreads();


	// konvolúció 3 komponensre
	float acc_r = 0, acc_g = 0, acc_b = 0;

	for (int a = 0; a < FILTER_H; a++){
		for (int b = 0; b < FILTER_W; b++) {
			acc_r += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3] * filter_laplace[a][b];
			acc_g += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3 + 1] * filter_laplace[a][b];
			acc_b += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3 + 2] * filter_laplace[a][b];
		}
	}

	// kimenet írása közvetlenül a globális memóriába
	if (acc_r > 255.0)
		gOutput[(row*imgWidth + col) * 3] = 255;
	else if (acc_r < 0.0)
		gOutput[(row*imgWidth + col) * 3] = 0;
	else
		gOutput[(row*imgWidth + col) * 3] = acc_r;

	if (acc_g > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 255;
	else if (acc_g < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 1] = acc_g;

	if (acc_b > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 255;
	else if (acc_b < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 2] = acc_b;
}


// Shared memóriát használó megoldás (3.)
// Shared memória adattípus: float
// Számítás adattípusa: float
__global__ void kernel_conv_sh_float_float(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = threadIdx.x + blockDim.x * blockIdx.x;   // a szál melyik oszlopban levõ kimeneti pixelt számolja
	int base = (blockIdx.y * blockDim.y * imgWidthF + blockIdx.x * blockDim.x) * 3;


	// Sharde Memory deklaráció

	__shared__ float mem[20 * 20 * 3];

	// Shared Memory feltöltés
	int th1D = blockDim.x * threadIdx.y + threadIdx.x;	// lieáris szál-azonosító a Thread Block-on belül
	if (th1D < 240)
		for (int d = 0; d < 5; d++) mem[th1D + 240 * d] = gInput[base + th1D % 60 + (4 * d + th1D / 60) * imgWidthF * 3];

	// Szál szinkronizáció
	__syncthreads();


	// konvolúció 3 komponensre
	float acc_r = 0, acc_g = 0, acc_b = 0;

	for (int a = 0; a < FILTER_H; a++){
		for (int b = 0; b < FILTER_W; b++) {
			acc_r += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3] * filter_laplace[a][b];
			acc_g += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3 + 1] * filter_laplace[a][b];
			acc_b += mem[(20 * (threadIdx.y + a) + threadIdx.x + b) * 3 + 2] * filter_laplace[a][b];
		}
	}

	// kimenet írása közvetlenül a globális memóriába
	if (acc_r > 255.0)
		gOutput[(row*imgWidth + col) * 3] = 255;
	else if (acc_r < 0.0)
		gOutput[(row*imgWidth + col) * 3] = 0;
	else
		gOutput[(row*imgWidth + col) * 3] = acc_r;

	if (acc_g > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 255;
	else if (acc_g < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 1] = acc_g;

	if (acc_b > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 255;
	else if (acc_b < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 2] = acc_b;

}



// Ugyanaz mint az elõbb, módosított blokk méretekkel (16*16) az olvasási shared memory bank konfliktus elkerüléséhez
__global__ void kernel_conv_sh_float_float_nbc(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = threadIdx.x + blockDim.x * blockIdx.x;   // a szál melyik oszlopban levõ kimeneti pixelt számolja
	int base = (blockIdx.y * blockDim.y * imgWidthF + blockIdx.x * blockDim.x) * 3;


	// Sharde Memory deklaráció

	__shared__ float mem[20 * 65];

	// Shared Memory feltöltés
	int th1D = blockDim.x * threadIdx.y + threadIdx.x;	// lieáris szál-azonosító a Thread Block-on belül
	if (th1D < 240)
		for (int d = 0; d < 5; d++) mem[th1D + 65 * 4 * d + (th1D / 60 * 5)] = gInput[base + th1D % 60 + (4 * d + th1D / 60) * imgWidthF * 3];

	// Szál szinkronizáció
	__syncthreads();


	// konvolúció 3 komponensre
	float acc_r = 0, acc_g = 0, acc_b = 0;

	for (int a = 0; a < FILTER_H; a++){
		for (int b = 0; b < FILTER_W; b++) {
			acc_r += mem[65 * (threadIdx.y + a) + (threadIdx.x + b) * 3] * filter_laplace[a][b];
			acc_g += mem[65 * (threadIdx.y + a) + (threadIdx.x + b) * 3 + 1] * filter_laplace[a][b];
			acc_b += mem[65 * (threadIdx.y + a) + (threadIdx.x + b) * 3 + 2] * filter_laplace[a][b];
		}
	}

	/*acc_r = mem[65 * (threadIdx.y) + (threadIdx.x) * 3];
	acc_g = mem[65 * (threadIdx.y) + (threadIdx.x) * 3 + 1];
	acc_b = mem[65 * (threadIdx.y) + (threadIdx.x) * 3 + 2];
	*/
	// kimenet írása közvetlenül a globális memóriába
	if (acc_r > 255.0)
		gOutput[(row*imgWidth + col) * 3] = 255;
	else if (acc_r < 0.0)
		gOutput[(row*imgWidth + col) * 3] = 0;
	else
		gOutput[(row*imgWidth + col) * 3] = acc_r;

	if (acc_g > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 255;
	else if (acc_g < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 1] = acc_g;

	if (acc_b > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 255;
	else if (acc_b < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 2] = acc_b;
}


// Ugyanaz mint az elõbb, módosított blokk méretekkel (32x8) az olvasási shared memory bank konfliktus elkerüléséhez
__global__ void kernel_conv_sh_float_float_nbc_easy(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = threadIdx.x + blockDim.x * blockIdx.x;   // a szál melyik oszlopban levõ kimeneti pixelt számolja
	int base = (blockIdx.y * blockDim.y * imgWidthF + blockIdx.x * blockDim.x) * 3;


	// Sharde Memory deklaráció

	__shared__ float mem[12][36][3];

	// Shared Memory feltöltés
	int th1D = blockDim.x * threadIdx.y + threadIdx.x;	// lieáris szál-azonosító a Thread Block-on belül
	if (th1D < 216)
		for (int d = 0; d < 6; d++)
			mem[th1D / (36 * 3) + d * 2][(th1D / 3) % 36][th1D % 3] = gInput[base + th1D % 108 + (th1D / (36 * 3) + d * 2) * imgWidthF * 3];

	// Szál szinkronizáció
	__syncthreads();


	// konvolúció 3 komponensre
	float acc_r = 0, acc_g = 0, acc_b = 0;

	for (int a = 0; a < FILTER_H; a++){
		for (int b = 0; b < FILTER_W; b++) {
			acc_r += mem[threadIdx.y + a][threadIdx.x + b][0] * filter_laplace[a][b];
			acc_g += mem[threadIdx.y + a][threadIdx.x + b][1] * filter_laplace[a][b];
			acc_b += mem[threadIdx.y + a][threadIdx.x + b][2] * filter_laplace[a][b];
		}
	}

	/*acc_r = mem[65 * (threadIdx.y) + (threadIdx.x) * 3];
	acc_g = mem[65 * (threadIdx.y) + (threadIdx.x) * 3 + 1];
	acc_b = mem[65 * (threadIdx.y) + (threadIdx.x) * 3 + 2];
	*/
	// kimenet írása közvetlenül a globális memóriába
	if (acc_r > 255.0)
		gOutput[(row*imgWidth + col) * 3] = 255;
	else if (acc_r < 0.0)
		gOutput[(row*imgWidth + col) * 3] = 0;
	else
		gOutput[(row*imgWidth + col) * 3] = acc_r;

	if (acc_g > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 255;
	else if (acc_g < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 1] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 1] = acc_g;

	if (acc_b > 255.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 255;
	else if (acc_b < 0.0)
		gOutput[(row*imgWidth + col) * 3 + 2] = 0;
	else
		gOutput[(row*imgWidth + col) * 3 + 2] = acc_b;
}

// Ugyanaz mint az elõbb, módosított blokk méretekkel (16*16) az olvasási shared memory bank konfliktus elkerüléséhez
__global__ void kernel_median(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = threadIdx.x + blockDim.x * blockIdx.x;   // a szál melyik oszlopban levõ kimeneti pixelt számolja
	int base = (blockIdx.y * blockDim.y * imgWidthF + blockIdx.x * blockDim.x) * 3;


	// Sharde Memory deklaráció

	__shared__ float mem[20 * 65];

	// Shared Memory feltöltés
	int th1D = blockDim.x * threadIdx.y + threadIdx.x;	// lieáris szál-azonosító a Thread Block-on belül
	if (th1D < 240)
		for (int d = 0; d < 5; d++) mem[th1D + 65 * 4 * d + (th1D / 60 * 5)] = gInput[base + th1D % 60 + (4 * d + th1D / 60) * imgWidthF * 3];

	// Szál szinkronizáció
	__syncthreads();


	// konvolúció 3 komponensre
	float acc_r = 0, acc_g = 0, acc_b = 0;

	for (int a = 0; a < FILTER_H; a++){
		for (int b = 0; b < FILTER_W; b++) {
			acc_r += mem[65 * (threadIdx.y + a) + (threadIdx.x + b) * 3] * filter_laplace[a][b];
			acc_g += mem[65 * (threadIdx.y + a) + (threadIdx.x + b) * 3 + 1] * filter_laplace[a][b];
			acc_b += mem[65 * (threadIdx.y + a) + (threadIdx.x + b) * 3 + 2] * filter_laplace[a][b];
		}
	}

	// kimenet írása közvetlenül a globális memóriába

		gOutput[(row*imgWidth + col) * 3] = acc_r;
		gOutput[(row*imgWidth + col) * 3 + 1] = acc_g;
		gOutput[(row*imgWidth + col) * 3 + 2] = acc_b;
}


__global__ void kernel_median_char(unsigned char* gInput, unsigned char* gOutput, int imgWidth, int imgWidthF)
{
	register unsigned int tmp;
#define swap(a, b) {if(a>b) {tmp=a; a=b; b=tmp;}}

	register unsigned int sort[25];


	int row = threadIdx.y + blockDim.y * blockIdx.y;	// a szál melyik sorban levõ kimeneti pixelt számolja
	int col = 2 * (threadIdx.x + blockDim.x * blockIdx.x);   // a szál melyik oszlopban levõ kimeneti pixelt számolja
	int base = (blockIdx.y * blockDim.y * imgWidthF + 2 * blockIdx.x * blockDim.x) * 3;

	// Sharde Memory deklaráció
	__shared__ unsigned char mem[36 * 20 * 3];

	// Shared Memory feltöltés
	int th1D = blockDim.x * threadIdx.y + threadIdx.x;	// lieáris szál-azonosító a Thread Block-on belül
	if (th1D < 216)
		for (int d = 0; d < 10; d++)
			mem[th1D + 216 * d] = gInput[base + th1D % 108 + (2 * d + th1D / 108) * imgWidthF * 3];

	// Szál szinkronizáció
	__syncthreads();

	for (int i = 0; i < 3; i++){
		for (int a = 0; a < FILTER_H; a++)
			for (int b = 0; b < FILTER_W; b++)
				sort[a * 5 + b] = mem[(36 * (threadIdx.y + a) + 2 * threadIdx.x + b) * 3 + i];
		//1
		swap(sort[1], sort[2]); swap(sort[3], sort[4]); swap(sort[6], sort[7]); swap(sort[8], sort[9]); swap(sort[11], sort[12]); swap(sort[13], sort[14]); swap(sort[16], sort[17]);
		swap(sort[1], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[8]); swap(sort[7], sort[9]); swap(sort[11], sort[13]); swap(sort[12], sort[14]);
		swap(sort[1], sort[6]); swap(sort[4], sort[9]); swap(sort[11], sort[16]); swap(sort[14], sort[17]);
		swap(sort[1], sort[11]); swap(sort[9], sort[17]);
		
		//2
		swap(sort[18], sort[2]); swap(sort[3], sort[4]); swap(sort[6], sort[7]); swap(sort[8], sort[9]); swap(sort[11], sort[12]); swap(sort[13], sort[14]);
		swap(sort[18], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[8]); swap(sort[7], sort[9]); swap(sort[11], sort[13]); swap(sort[12], sort[14]);
		swap(sort[18], sort[6]); swap(sort[4], sort[9]); swap(sort[11], sort[16]); swap(sort[14], sort[16]);
		swap(sort[18], sort[11]); swap(sort[9], sort[16]);
		
		//3
		swap(sort[19], sort[2]); swap(sort[3], sort[4]); swap(sort[6], sort[7]); swap(sort[8], sort[9]); swap(sort[11], sort[12]); swap(sort[13], sort[14]);
		swap(sort[19], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[8]); swap(sort[7], sort[9]); swap(sort[11], sort[13]); swap(sort[12], sort[14]);
		swap(sort[19], sort[6]); swap(sort[4], sort[9]); swap(sort[11], sort[14]);
		swap(sort[19], sort[11]); swap(sort[9], sort[14]);
		
		//4
		swap(sort[21], sort[2]); swap(sort[3], sort[4]); swap(sort[6], sort[7]); swap(sort[8], sort[9]); swap(sort[11], sort[12]);
		swap(sort[21], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[8]); swap(sort[7], sort[9]); swap(sort[12], sort[13]);
		swap(sort[21], sort[6]); swap(sort[4], sort[9]); swap(sort[11], sort[13]);
		swap(sort[21], sort[11]); swap(sort[9], sort[13]);
		
		//5
		swap(sort[22], sort[2]); swap(sort[3], sort[4]); swap(sort[6], sort[7]); swap(sort[8], sort[9]);
		swap(sort[22], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[8]); swap(sort[7], sort[9]);
		swap(sort[22], sort[6]); swap(sort[4], sort[9]); swap(sort[11], sort[12]);
		swap(sort[22], sort[11]); swap(sort[9], sort[12]);
		
		//6
		swap(sort[23], sort[2]); swap(sort[3], sort[4]); swap(sort[6], sort[7]); swap(sort[8], sort[9]);
		swap(sort[23], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[8]); swap(sort[7], sort[9]);
		swap(sort[23], sort[6]); swap(sort[4], sort[9]);
		swap(sort[23], sort[11]); swap(sort[9], sort[11]);
		
		//7
		swap(sort[24], sort[2]); swap(sort[3], sort[4]); swap(sort[6], sort[7]); swap(sort[8], sort[9]);
		swap(sort[24], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[8]); swap(sort[7], sort[9]);
		swap(sort[24], sort[6]); swap(sort[4], sort[9]);
		swap(sort[24], sort[9]);
		
		//Counting second pixel, too
		//0,2,3,4,6,7,8,5,10,15,20
		sort[11] = sort[2];
		sort[12] = sort[3];
		sort[13] = sort[4];
		sort[14] = sort[6];
		sort[16] = sort[7];
		sort[17] = sort[8];
		sort[18] = mem[(36 * (threadIdx.y) + 2*threadIdx.x + FILTER_W) * 3 + i];
		sort[19] = mem[(36 * (threadIdx.y + 1) + 2*threadIdx.x + FILTER_W) * 3 + i];
		sort[21] = mem[(36 * (threadIdx.y + 2) + 2*threadIdx.x + FILTER_W) * 3 + i];
		sort[22] = mem[(36 * (threadIdx.y + 3) + 2*threadIdx.x + FILTER_W) * 3 + i];
		sort[23] = mem[(36 * (threadIdx.y + 4) + 2*threadIdx.x + FILTER_W) * 3 + i];
		
		//8
		swap(sort[0], sort[2]); swap(sort[3], sort[4]); swap(sort[14], sort[7]);
		swap(sort[0], sort[3]); swap(sort[2], sort[4]); swap(sort[14], sort[8]); swap(sort[7], sort[8]);
		swap(sort[0], sort[14]); swap(sort[4], sort[8]);
		swap(sort[0], sort[8]);
		//8-2
		swap(sort[18], sort[11]); swap(sort[12], sort[13]); swap(sort[6], sort[16]);
		swap(sort[18], sort[12]); swap(sort[11], sort[13]); swap(sort[6], sort[17]); swap(sort[16], sort[17]);
		swap(sort[18], sort[6]); swap(sort[13], sort[17]);
		swap(sort[18], sort[17]);
		
		//9
		swap(sort[5], sort[2]); swap(sort[3], sort[4]);
		swap(sort[5], sort[3]); swap(sort[2], sort[4]); swap(sort[6], sort[7]);
		swap(sort[5], sort[6]); swap(sort[4], sort[7]);
		swap(sort[5], sort[7]);
		//9-2
		swap(sort[19], sort[11]); swap(sort[12], sort[13]);
		swap(sort[19], sort[12]); swap(sort[11], sort[13]); swap(sort[14], sort[16]);
		swap(sort[19], sort[14]); swap(sort[13], sort[16]);
		swap(sort[19], sort[16]);
		
		//10
		swap(sort[10], sort[2]); swap(sort[3], sort[4]);
		swap(sort[10], sort[3]); swap(sort[2], sort[4]);
		swap(sort[4], sort[6]);
		swap(sort[10], sort[6]);
		//10-2
		swap(sort[21], sort[11]); swap(sort[12], sort[13]);
		swap(sort[21], sort[12]); swap(sort[11], sort[13]);
		swap(sort[13], sort[14]);
		swap(sort[21], sort[14]);
		
		//11
		swap(sort[15], sort[2]); swap(sort[3], sort[4]);
		swap(sort[15], sort[3]); swap(sort[2], sort[4]);
		swap(sort[15], sort[4]);
		//11-2
		swap(sort[22], sort[11]); swap(sort[12], sort[13]);
		swap(sort[22], sort[12]); swap(sort[11], sort[13]);
		swap(sort[22], sort[13]);
		
		//12
		swap(sort[20], sort[2]);
		swap(sort[2], sort[3]);
		swap(sort[20], sort[3]);
		//12-2
		swap(sort[23], sort[11]);
		swap(sort[11], sort[12]);
		swap(sort[23], sort[12]);

		gOutput[(row*imgWidth + col) * 3 + i] =  sort[2];
		gOutput[(row*imgWidth + col + 1) * 3 + i] = sort[11];
	}
}


void cudaMain(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF,
			  int imgFOfssetH, int imgFOfssetW,
			  unsigned char *imgSrc, unsigned char *imgDst)

{
    double s0, e0;
    double d0;


    int size_in  = imgWidthF*imgHeightF*sizeof(unsigned char) * 3;
	int size_out = imgWidth*imgHeight*sizeof(unsigned char) * 3;

    unsigned char *gInput, *gOutput;
	cudaMalloc((void**)&gInput, size_in);
	cudaMalloc((void**)&gOutput, size_out);

	dim3 thrBlock(16, 16);
    dim3 thrGrid(imgWidth/16, imgHeight/16);

	dim3 thrBlock2(32, 8);
	dim3 thrGrid2(imgWidth / 32, imgHeight / 8);
	
	dim3 thrBlock3(16, 16);
	dim3 thrGrid3(imgWidth / 32, imgHeight / 16);

	cudaMemcpy(gInput, imgSrc, size_in, cudaMemcpyHostToDevice); 

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	s0 = time_measure(1);
	for (int i = 0; i < KERNEL_RUNS; i++)
	{
		//kernel_conv_global << <thrGrid, thrBlock >> >(gInput, gOutput, imgWidth, imgWidthF);
		//kernel_conv_sh_uchar_int << <thrGrid, thrBlock >> >(gInput, gOutput, imgWidth, imgWidthF);
		//kernel_conv_sh_uchar_float << <thrGrid, thrBlock >> >(gInput, gOutput, imgWidth, imgWidthF);
		//kernel_conv_sh_float_float << <thrGrid, thrBlock >> >(gInput, gOutput, imgWidth, imgWidthF);
		//kernel_conv_sh_float_float_nbc << <thrGrid, thrBlock >> >(gInput, gOutput, imgWidth, imgWidthF);
		//kernel_conv_sh_float_float_nbc_easy << <thrGrid2, thrBlock2 >> >(gInput, gOutput, imgWidth, imgWidthF);
		//kernel_median << <thrGrid, thrBlock >> >(gInput, gOutput, imgWidth, imgWidthF);
		kernel_median_char << <thrGrid3, thrBlock3 >> >(gInput, gOutput, imgWidth, imgWidthF);
	}
	cudaThreadSynchronize();
	e0 = time_measure(2);

    cudaMemcpy(imgDst, gOutput, size_out, cudaMemcpyDeviceToHost);
	
    cudaFree(gInput); cudaFree(gOutput);

	cudaDeviceReset();

    d0 = (double)(e0-s0)/(CLOCKS_PER_SEC*KERNEL_RUNS);
	double mpixel = (imgWidth*imgHeight / d0) / 1000000;
    printf("CUDA single kernel time: %4.4f\n", d0);
	printf("CUDA Mpixel/s: %4.4f\n", mpixel);
}
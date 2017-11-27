#include "omp.h"

#include "emmintrin.h"
#include "nmmintrin.h"
#include "immintrin.h"


#include "defs.h"

void conv_filter(int imgHeight, int imgWidth, int imgHeightF, int imgWidthF,
				 int imgFOfssetH, int imgFOfssetW,
				 float *filter, unsigned char *imgSrc, unsigned char *imgDst)
{

#if USE_OMP==1
	#pragma omp parallel for
#else
#endif
	for (int row=imgFOfssetH; row<(imgHeight+imgFOfssetH); row++)
	{
		int wr_base = (row*imgWidthF + imgFOfssetH)*3;
		int rd_base = (row-imgFOfssetW)*imgWidthF*3;
		for (int col=imgFOfssetW; col<(imgWidth+imgFOfssetW); col++)
		{
			float fval[3] = {0,0,0};
			int rd_offset = 0;
			for (int fy=0; fy<FILTER_H; fy++)
			{
				for (int fx=0; fx<FILTER_W; fx++)
				{
					for (int rgb=0; rgb<3; rgb++)
					{
						int pixel_src = rd_base + rd_offset;
						fval[rgb] = fval[rgb] + *(filter+fy*FILTER_W+fx) * (float)(*(imgSrc+pixel_src+rgb));
					}
					rd_offset = rd_offset + 3;
				}
				rd_offset = rd_offset - 4*3 + imgWidthF*3;
				
			}

			for (int rgba=0; rgba<3; rgba++)
			{
				if (fval[rgba]>255.0) fval[rgba]=255.0;
				if (fval[rgba]<0.0)   fval[rgba]=0.0;
			}

			for (int rgb=0; rgb<3; rgb++)
			{
				*(imgDst+wr_base+rgb) = fval[rgb];
			}
			wr_base = wr_base + 3;
			rd_base = rd_base + 3;
		}
		wr_base = wr_base + 4*3;
		rd_base = rd_base + 4*3;
	}
}


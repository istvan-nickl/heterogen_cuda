#include <stdio.h>
#include <windows.h>

double time_measure(int mode)
{
	static double PCFreq = 0.0;
	static __int64 CounterStart = 0;

	if (mode == 1)	// start
	{
		LARGE_INTEGER li;
		if (!QueryPerformanceFrequency(&li))
			printf("QueryPerformanceFrequency failed!\n");

		PCFreq = double(li.QuadPart) / 1000.0;

		QueryPerformanceCounter(&li);
		CounterStart = li.QuadPart;
		return(-1.0);
	}
	else if (mode == 2)	// end
	{
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return double(li.QuadPart - CounterStart) / PCFreq;
	}
}
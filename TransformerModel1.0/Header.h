#pragma once
#include <iostream>
#include <vector>
#include <chrono>

using std::cout;
using std::endl;
using std::vector;

using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

static uint32_t seed;
static uint32_t kn[128];
static float fn[128];
static float wn[128];

static void Seed(uint32_t Seed = 0x8a777e69) {
	double dn = 3.442619855899;
	int i;
	const double m1 = 2147483648.0;
	double q;
	double tn = 3.442619855899;
	const double vn = 9.91256303526217E-03;

	q = vn / exp(-0.5 * dn * dn);

	kn[0] = (uint32_t)((dn / q) * m1);
	kn[1] = 0;

	wn[0] = (float)(q / m1);
	wn[127] = (float)(dn / m1);

	fn[0] = 1.0;
	fn[127] = (float)(exp(-0.5 * dn * dn));

	for (i = 126; 1 <= i; i--)
	{
		dn = sqrt(-2.0 * log(vn / dn + exp(-0.5 * dn * dn)));
		kn[i + 1] = (uint32_t)((dn / tn) * m1);
		tn = dn;
		fn[i] = (float)(exp(-0.5 * dn * dn));
		wn[i] = (float)(dn / m1);
	}

	seed = Seed ^
		duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count() ^
		duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

static uint32_t uintRand()
{
	uint32_t temp;

	temp = seed;
	seed = (seed ^ (seed << 13));
	seed = (seed ^ (seed >> 17));
	seed = (seed ^ (seed << 5));

	return temp + seed;
}

static float floatRand()
{
	uint32_t temp;

	temp = seed;
	seed = (seed ^ (seed << 13));
	seed = (seed ^ (seed >> 17));
	seed = (seed ^ (seed << 5));

	return fmod(0.5 + (float)(temp + seed) * 2.32830643654e-10, 1.0);
}

static float normalRand()
{
	int hz;
	uint32_t iz;
	const float r = 3.442620;
	float value;
	float x;
	float y;

	hz = (int)uintRand();
	iz = (hz & 127);

	if (fabs(hz) < kn[iz]) {
		return (float)(hz)*wn[iz];
	}

	while (true) {
		if (iz == 0) {
			do {
				x = -0.2904764 * log(floatRand());
				y = -log(floatRand());
			} while (x * x > y + y);

			return (hz <= 0) * (-r - x) + (hz > 0) * (r + x);
		}

		x = (float)(hz)*wn[iz];

		if (fn[iz] + floatRand() * (fn[iz - 1] - fn[iz]) < exp(-0.5 * x * x))
		{
			return x;
		}

		hz = (int)uintRand();
		iz = (hz & 127);

		if (fabs(hz) < kn[iz])
		{
			return (float)(hz)*wn[iz];
		}
	}
}
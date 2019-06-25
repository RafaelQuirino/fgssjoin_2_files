/*
#ifdef __cplusplus
extern "C" {
#endif
*/

#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#ifdef __MACH__
	#include <mach/clock.h>
	#include <mach/mach.h>
#endif

using namespace std;

#define FALSE 0
#define TRUE  1



void msleep (unsigned long ms);
void current_utc_time (struct timespec *ts);
unsigned long getTimeInMicroseconds ();
unsigned long getTimeInMiliseconds ();
double intervalInMiliseconds (unsigned long t0, unsigned long t1);

void print_separator (const char*, int size);

unsigned bernstein ( void *key, int len );
unsigned long bernstein_hash (char *str);
unsigned long djb2_hash (char *str);
unsigned long sdbm_hash (char *str);



vector<string> split (const string &s, char delim, vector<string> &elems);
vector<string> split (const string &s, char delim);
void print_str_vec (vector<string> vec);

#endif /* _UTIL_H_ */

/*
#ifdef __cplusplus
}
#endif
*/

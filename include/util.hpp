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



/* DOCUMENTATION
 *
 */
void ut_msleep (unsigned long ms);



/* DOCUMENTATION
 *
 */
void ut_current_utc_time (struct timespec *ts);



/* DOCUMENTATION
 *
 */
unsigned long ut_get_time_in_microseconds ();



/* DOCUMENTATION
 *
 */
unsigned long ut_get_time_in_miliseconds ();



/* DOCUMENTATION
 *
 */
double ut_interval_in_miliseconds (unsigned long t0, unsigned long t1);



/* DOCUMENTATION
 *
 */
void ut_print_separator (const char*, int size);



/* DOCUMENTATION
 *
 */
unsigned ut_bernstein ( void *key, int len );



/* DOCUMENTATION
 *
 */
unsigned long ut_bernstein_hash (char *str);



/* DOCUMENTATION
 *
 */
unsigned long ut_djb2_hash (char *str);



/* DOCUMENTATION
 *
 */
unsigned long ut_sdbm_hash (char *str);



/* DOCUMENTATION
 *
 */
vector<string> ut_split (const string &s, char delim, vector<string> &elems);



/* DOCUMENTATION
 *
 */
vector<string> ut_split (const string &s, char delim);



/* DOCUMENTATION
 *
 */
void ut_print_str_vec (vector<string> vec);



#endif /* _UTIL_H_ */

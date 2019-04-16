#include "util.hpp"



/* DOCUMENTATION
 *
 */
void ut_msleep(unsigned long ms)
{
	usleep(1000*ms);
}



/* DOCUMENTATION
 *
 */
void ut_current_utc_time(struct timespec *ts)
{
    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
        clock_serv_t cclock;
        mach_timespec_t mts;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &mts);
        mach_port_deallocate(mach_task_self(), cclock);
        ts->tv_sec = mts.tv_sec;
        ts->tv_nsec = mts.tv_nsec;
    #else
        clock_gettime(CLOCK_REALTIME, ts);
    #endif
}



/* DOCUMENTATION
 *
 */
unsigned long ut_get_time_in_microseconds()
{
    unsigned long   us; // Microseconds
    time_t          s;  // Seconds
    struct timespec spec;

    ut_current_utc_time(&spec);
    s  = spec.tv_sec;
    us = round(spec.tv_nsec / 1.0e3); // Convert nanoseconds to microseconds

    unsigned long x = (long)(intmax_t)s;

    return x*1.0e6 + us;
}



/* DOCUMENTATION
 *
 */
unsigned long ut_get_time_in_miliseconds()
{
    unsigned long   ms; // Milliseconds
    time_t          s;  // Seconds
    struct timespec spec;

    ut_current_utc_time(&spec);
    s  = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds

    unsigned long x = (long)(intmax_t)s;

    return x*1000 + ms;
}



/* DOCUMENTATION
 *
 */
double ut_interval_in_miliseconds (unsigned long t0, unsigned long t1)
{
    return (double)(t1-t0)/1000.0;
}



/* DOCUMENTATION
 *
 */
void ut_print_separator (const char* str, int size)
{
    int i; // size = 90;
    for (i = 0; i < size; i++) printf("%s", str);
    printf("\n");
}



/* DOCUMENTATION
 *
 */
unsigned ut_bernstein (void *key, int len)
{
    unsigned char *p = (unsigned char*) key;
    unsigned h = 0;
    int i;

    for (i = 0; i < len; i++)
        h = 33 * h + p[i];

    return h;
}



/* DOCUMENTATION
 *
 */
unsigned long ut_bernstein_hash (char *str)
{
	unsigned char* ustr = (unsigned char*)str;
	int len = (int) strlen(str);
	return ut_bernstein((void*)str,len);
}



/* DOCUMENTATION
 *
 */
unsigned long ut_djb2_hash (char *str)
{
	unsigned char* ustr = (unsigned char*)str;
    unsigned long hash = 5381;
    int c;

    while (c = *ustr++)
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}



/* DOCUMENTATION
 *
 */
unsigned long ut_sdbm_hash (char *str)
{

	unsigned char *ustr = (unsigned char*)str;
    unsigned long hash = 0;
    int c;

    while (c = *ustr++)
        hash = c + (hash << 6) + (hash << 16) - hash;

    return hash;
}



/* DOCUMENTATION
 *
 */
uint32_t ut_murmur3_32 (const uint8_t* key, size_t len, uint32_t seed) {
    uint32_t h = seed;
    
    if (len > 3) {
        const uint32_t* key_x4 = (const uint32_t*) key;
        size_t i = len >> 2;
        do {
            uint32_t k = *key_x4++;
            k *= 0xcc9e2d51;
            k = (k << 15) | (k >> 17);
            k *= 0x1b873593;
            h ^= k;
            h = (h << 13) | (h >> 19);
            h = (h * 5) + 0xe6546b64;
        } while (--i);
        key = (const uint8_t*) key_x4;
    }

    if (len & 3) {
        size_t i = len & 3;
        uint32_t k = 0;
        key = &key[i - 1];
        do {
            k <<= 8;
            k |= *key--;
        } while (--i);
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
    }

    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}



/* DOCUMENTATION
 *
 */
vector<string> ut_split (const string &s, char delim, vector<string> &elems)
{
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}



/* DOCUMENTATION
 *
 */
vector<string> ut_split (const string &s, char delim)
{
    vector<string> elems;
    ut_split(s, delim, elems);
    return elems;
}



/* DOCUMENTATION
 *
 */
void ut_print_str_vec (vector<string> vec)
{
	int i;
	for (i = 0; i < vec.size(); i++)
	{
		printf("%s\n", vec[i].c_str());
	}
}

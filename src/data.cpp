#include "io.hpp"
#include "util.hpp"
#include "data.hpp"



/* DOCUMENTATION
 *
 */
vector<string>
dat_get_input_data (string file_path, int input_size)
{
    if (input_size == 0) fprintf (stderr, "\tReading %s...\n", file_path.c_str());
	else fprintf (stderr, "\tReading %d lines from %s...\n", input_size, file_path.c_str());
	unsigned long t0 = ut_get_time_in_microseconds();

	vector<string> input;
	if (input_size == 0) input = io_get_input_from_file (file_path);
	else input = io_get_partial_input_from_file (file_path, input_size);

	unsigned long t1 = ut_get_time_in_microseconds();
	fprintf (stderr, "\t> Done in %gms.\n", ut_interval_in_miliseconds (t0,t1));

    return input;
}



/* DOCUMENTATION
 * Auxiliar function
 */
char dat_easytolower(char in){
    if (in <= 'Z' && in >= 'A')
        return in - ('Z'-'z');
    return in;
}



/* DOCUMENTATION
 *
 */
vector<string>
dat_get_proc_data (vector<string> input_data)
{
    // I hope this makes a copy...
    vector<string> proc_data = input_data;

    fprintf (stderr, "\tProcessing data...\n");
    unsigned long t0 = ut_get_time_in_microseconds();

    for (unsigned i = 0; i < input_data.size(); i++) {
        transform (
            proc_data[i].begin(), proc_data[i].end(), proc_data[i].begin(),
            dat_easytolower
        );
    }

    unsigned long t1 = ut_get_time_in_microseconds();
    fprintf (stderr, "\t> Done in %gms.\n", ut_interval_in_miliseconds (t0,t1));

    return proc_data;
}

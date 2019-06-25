#include "data.h"
#include "util.h"
#include "io.h"

vector<string>
get_input_data (string file_path, int input_size)
{
    if (input_size == 0)
        fprintf(stderr, "\tReading %s...\n", file_path.c_str());
	else
        fprintf(stderr, "\tReading %d lines from %s...\n",
            input_size, file_path.c_str());

    unsigned long t0 = getTimeInMicroseconds();

	vector<string> input;
	if (input_size == 0) input = getInputFromFile (file_path);
	else input = getPartialInputFromFile (file_path, input_size);

	unsigned long t1 = getTimeInMicroseconds();
	fprintf(stderr, "\t> Done in %gms.\n", intervalInMiliseconds(t0,t1));

    return input;
}

// Auxiliar function
char _easytolower(char in){
    if (in <= 'Z' && in >= 'A')
        return in - ('Z'-'z');
    return in;
}

vector<string>
get_proc_data (vector<string> input_data)
{
    // I hope this makes a copy...
    vector<string> proc_data = input_data;

    fprintf(stderr, "\tProcessing data...\n");
    unsigned long t0 = getTimeInMicroseconds();

    for (unsigned i = 0; i < input_data.size(); i++) {
        transform(
            proc_data[i].begin(), proc_data[i].end(), proc_data[i].begin(),
            _easytolower);
    }

    unsigned long t1 = getTimeInMicroseconds();
    fprintf(stderr, "\t> Done in %gms.\n", intervalInMiliseconds(t0,t1));

    return proc_data;
}

void proc_data (vector<string>& input_data)
{
    fprintf(stderr, "\tProcessing data...\n");
    unsigned long t0 = getTimeInMicroseconds();

    for (unsigned i = 0; i < input_data.size(); i++) {
        transform(
            input_data[i].begin(), input_data[i].end(), input_data[i].begin(),
            _easytolower);
    }

    unsigned long t1 = getTimeInMicroseconds();
    fprintf(stderr, "\t> Done in %gms.\n", intervalInMiliseconds(t0,t1));
}

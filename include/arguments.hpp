#ifdef _cplusplus
extern "C" {
#endif

#ifndef _ARGUMENTS_H_
#define _ARGUMENTS_H_



#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#include <string>



using namespace std;



/* DOCUMENTATION
 *
 */
struct Arguments {
	string file_path;
	int input_size;
	int qgram;
	string output_path;
	float threshold;
};



/* DOCUMENTATION
 *
 */
bool is_number (char number[])
{
    int i = 0;

    if (number[0] == '-')
        i = 1;

    for (; number[i] != 0; i++)
    {
        if (!isdigit(number[i]))
            return false;
    }

    return true;
}



/* DOCUMENTATION
 *
 */
bool is_dir (char* path)
{
	struct stat s;
	int err = stat(path, &s);
	if(-1 == err) {
	    if(ENOENT == errno) {
	        /* does not exist */
	        return false;
	    } else {
	        //perror("stat");
	        //exit(1);
	        return false;
	    }
	} else {
	  if(S_ISDIR(s.st_mode)) {
	        /* it's a dir */
	        return true;
	  } else {
	        /* exists but is no dir */
	        return false;
	  }
	}
}



/* DOCUMENTATION
 *
 */
void print_usage()
{
    fprintf(stderr, "usage: path/exec -f file_path -q qgram_size -t threshold");
    fprintf(stderr, " [-o output_path -n #_of_lines]\n");
}



/* DOCUMENTATION
 *
 */
Arguments get_arguments (int argc, char** argv)
{
	int opt = 0;
	char *fname_arg = NULL;
	char *qgram_arg = NULL;
	char *nlines_arg = NULL;
	char *outpath_arg = NULL;
	char *threshold_arg = NULL;

	while ((opt = getopt(argc, argv, "f:n:q:o:t:")) != -1) {
		switch(opt) {
			case 'f': fname_arg = optarg;
			break;

			case 'q': qgram_arg = optarg;
			break;

			case 'n': nlines_arg = optarg;
			break;

			case 'o': outpath_arg = optarg;
			break;

			case 't': threshold_arg = optarg;
			break;

			case '?':
				if (optopt == 'n') {
					print_usage();
					fprintf(stderr, "Ignoring option...\n");
				}
			break;
		}
	}

	if (fname_arg == NULL) {
		print_usage();
		exit(1);
	}

	if (qgram_arg == NULL) {
		print_usage();
		exit(1);
	}

	if (threshold_arg == NULL) {
		print_usage();
		exit(1);
	}

	if( access( fname_arg, F_OK ) == -1 ) {
		fprintf(stderr, "File doesn't exist.\nAborting...\n");
		exit(1);
	}

	if (!is_number(qgram_arg)) {
		fprintf(stderr, "Qgram size is not integer.\nAborting...\n");
		exit(1);
	} else if (atoi(qgram_arg) <= 0) {
		fprintf(stderr, "Qgram size must be a positive integer.\nAborting...\n");
		exit(1);
	}

	if (outpath_arg == NULL)
		outpath_arg = (char*) ".";
	else if (!is_dir(outpath_arg))
		fprintf(stderr, "Output path not found...\n");

	int flag = 0;
	if (nlines_arg != NULL) {
		flag = 1;
		if (!is_number(nlines_arg)) {
			fprintf(stderr, "Input size is not integer. Ignoring option...\n");
			flag = 0;
		} else if (atoi(nlines_arg) <= 0) {
			fprintf(stderr, "Input size must be a positive integer. Ignoring option...\n");
			flag = 0;
		}
	}

	Arguments args;
	args.file_path = fname_arg;
	args.input_size = flag == 1 ? atoi(nlines_arg) : 0;
	args.qgram = atoi(qgram_arg);
	args.output_path = outpath_arg;
	args.threshold = atof(threshold_arg);

	return args;
}



#endif // _ARGUMENTS_H_

#ifdef _cplusplus
}
#endif

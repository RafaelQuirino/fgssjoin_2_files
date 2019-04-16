#fgssjoin: Filtering GPU-based Set Similarity Join
---

Parallel Set Similarity Join algorithms for CUDA.

The data/ directory contains test data, from the dblp dataset.

To **compile** the project just run the script compile.sh. 
You must have CUDA environment installed, and the first variable 
in the Makefile inside src/ directory (CUDA_INSTALL_PATH) properly configured.

Examples:
---

* Standar compilation:

  - $ ./compile.sh

* Compile/recompile the whole project:

  - $ ./compile.sh all

* Compile specific files:

  - $ ./compile.sh file1 file2 etc...
  
* Clean executable and object files

  - $ ./compile.sh clean

To **run** the project, execute bin/fgssjoin executable file, created
after the compilation process, with options -f (data file, with each
record in one line), -q (size of the qgrams, 3 is a good value) and
-t (similarity threshold, between 0.0 and 1.0).

* Execution example, printing result to STDOUT

  - $ bin/fgssjoin -f data/dblp_t_18k.txt -q 3 -t 0.9

* Exeution example, printing result to an output file

  - $ bin/fgssjoin -r data/dblp_t_18k.txt -q 3 -t 0.9 > output

Reference:
---
Quirino R., Junior S., Ribeiro L. and Martins W. (2017). fgssjoin: A GPU-based Algorithm for Set Similarity Joins . In Proceedings of the 19th International Conference on Enterprise Information Systems - Volume 1: ICEIS, ISBN 978-989-758-247-9, pages 152-161. DOI: 10.5220/0006339001520161

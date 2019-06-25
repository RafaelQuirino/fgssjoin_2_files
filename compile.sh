#!/bin/bash

# Name of the executable
EXEC="fgssjoin"

# Name of object's directory
OBJ=".obj"
SRC="./src"
BIN="./bin"

# Create obj/ directory if none
if [ ! -d $OBJ ]; then
	mkdir $OBJ
fi

# Create bin/ directory if none
if [ ! -d $BIN ]; then
	mkdir $BIN
fi

# If [clean] option given, remove objects and executable
if [ "$1" == "clean" ]; then
	nfiles=$(ls ./$OBJ/|grep '\.o$'|wc -l)
	if [ $nfiles != 0 ]; then
		rm $OBJ/*.o
	fi
	if [ -a $BIN/$EXEC ]; then
		rm $BIN/$EXEC
	fi
	exit
fi

# If [all] option given, remove objects and executable;
# or else only the given targets
if [ "$1" == "all" ]; then
	nfiles=$(ls ./$OBJ/|grep '\.o$'|wc -l)
	if [ $nfiles != 0 ]; then
		rm $OBJ/*.o
	fi
	if [ -a $BIN/$EXEC ]; then
		rm $BIN/$EXEC
	fi
else
	for target in "$@"; do
		[ -f ./$OBJ/$target.o ] && rm $OBJ/$target.o
	done
fi

# Bring all objects to src directory
nfiles=$(ls ./$OBJ/|grep '\.o$'|wc -l)
if [ $nfiles != 0 ]; then
	for fname in "./$OBJ/*.o"; do
		mv $fname ./src
	done
fi

# Bring executable to src directory
if [ -a $BIN/$EXEC ]; then
	mv $BIN/$EXEC $SRC
fi

# Get .c, .cpp and .cu files from src directory
SOURCES=$(ls -R $SRC/|grep '\.c$\|\.cpp$\|\.cu$')

# Build string with all sources, concat with '.o'
SRCS=""
for x in $SOURCES; do
    SRCS="$SRCS$x.o "
done

# Call make, passing sources argument and executable name
printf "Building...\n"
cd $SRC
SRC=$SRCS EXEC=$EXEC make
cd ../

# Move back all object to $OBJ directory,
# so they don't pollute the sources directory
for fname in "$SRC/*.o"; do
	mv $fname $OBJ
done

# Move back executable to bin directory
mv $SRC/$EXEC $BIN/

#!/bin/bash
srcdir=$1
objdir=$2

files=$(find $srcdir -name "*.c" -or -name "*.cpp")

echo "Making dependencies..."

for f in $files;
do
    # Get filename without ".c" or ".cpp"
    name=$(basename $f | cut -d"." -f 1)
    # Get tie directory name, and change srcdir by objdir (in first position)
    dir_name=`dirname $f`/
    file_dir_truncated=`echo $dir_name | cut -d"/" -f 2-`
    file_dir="$objdir/$file_dir_truncated"

    # Now we can generate the buggy dependancies with gcc
    g++ -std=c++0x -MM $f > "$file_dir$name.d"
    # And correct them
    safe_file_dir=$(printf "%s\n" "$file_dir" | sed 's/[\&/]/\\&/g')
    sed -i -e "s/^\(.*:\)/$safe_file_dir\1/g" $file_dir$name.d
done

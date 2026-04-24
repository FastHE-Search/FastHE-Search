#!/bin/bash

#  Copyright (c) 2025 Sam Martin, Nirajan Koirala, Helena Berens, Micah Brody, Taeho Jung
#
#  Licensed under the MIT License (the "License"); you may not use this file
#  except in compliance with the License.
#
#  You may obtain a copy of the License in the LICENSE file at the project
#  root or at
#
#  https://mit-license.org/
#
#  SPDX-License-Identifier: MIT

# run this script from build/ using
# ../tools/dataset.sh

# ---------- constant variables ----------
DIM=512

# ---------- shell functions ----------
usage() {
    printf "generate_data.sh [FILENAME] [SIZE]\n\n"
    printf "Parameters:\n"
    printf "\tFILENAME\tFilename create dataset within\n"
    printf "\tSIZE    \tInteger number of backend vectors\n"
    return
}

write_query_vec () {
    for ((i=0; i < $DIM; i++)); do
        printf "1 " >> $FILEPATH
    done
    printf "\n" >> $FILEPATH
    return
}

write_random_vec () {
    for ((i=0; i < $DIM; i++)); do
        printf "%d " $(( $RANDOM % 199 - 99)) >> $FILEPATH
    done
    printf "\n" >> $FILEPATH
    return
}

write_matching_vec () {
    for ((i=0; i < $DIM; i++)); do
        printf "%d " $(( $RANDOM % 3 + 1)) >> $FILEPATH
    done
    printf "\n" >> $FILEPATH
    return
}

# ---------- main execution ----------

# check for necessary cmd-line args
if [[ $# -gt 1 ]]; then
    FILEPATH="../test/$1"
    SIZE="$2"
else
    usage
    exit 1
fi

# check that the vector-number param is an integer
if [[ !("$SIZE" =~ ^[0-9]+$) ]]; then
    usage
    exit 1
fi

# write number of vectors and query vector to filepath
printf "$SIZE\n" > $FILEPATH
$(write_query_vec)

# write database vectors to filepath
# matches at indices i=2 and i=n-1
# todo: implement user-input for determining which vectors match
for (( i=0; i<$SIZE; i++ )); do

    if (( i == 0 )); then
        $(write_matching_vec)
    else
        $(write_random_vec)
    fi

done

exit 0
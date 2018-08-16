#!/bin/bash

EXPECTED_ARGS=2
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` <username-MPIIMD> <password-MPIIMD> [parallelDownloads=8]"
  exit $E_BADARGS
fi
usernameMD=$1
passwordMD=$2
parallelDownloads=$3

if [ $# -lt 3 ]
then
  parallelDownloads=8
fi

########## download annotations
# Training set
wget http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/lsmdc2016/LSMDC16_annos_training_FIB.csv --user=$usernameMD --password=$passwordMD
# Validation set
wget http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/lsmdc2016/LSMDC16_annos_val_FIB.csv --user=$usernameMD --password=$passwordMD
# Public_Test set
wget http://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/lsmdc2016/LSMDC16_annos_test_FIB.csv --user=$usernameMD --password=$passwordMD

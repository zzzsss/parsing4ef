#!/bin/bash

cp $ZZ/parsing/ef/t .
LD_LIBRARY_PATH=${ZZ}/libs/dynet/cbuild/dynet/:$LD_LIBRARY_PATH ./t $*

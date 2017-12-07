#!/bin/bash

LD_LIBRARY_PATH=${ZZ}/libs/dynet/cbuild/dynet/:$LD_LIBRARY_PATH gdb $ZZ/parsing/ef/t $*

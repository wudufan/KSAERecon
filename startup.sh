#!/bin/bash
#start jupyter notebook with path set

export LD_LIBRARY_PATH=/usr/local/cuda/lib64;$LD_LIBRARY_PATH
jupyter notebook --script

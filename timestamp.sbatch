#!/bin/bash

#SBATCH 

ml python/3.9.0 py-pytorch/2.0.0_py39
pip3 install datasets soundfile librosa transformers
sbatch timestamp.sbatch

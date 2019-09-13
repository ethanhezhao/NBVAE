#!/bin/bash

# NBVAE_b with [600-200]
# training
python3 run_nbvae.py -d ML-10M -b 1 -m 3 -s ./demo_save -t 1 -a 6
# testing
python3 run_nbvae.py -d ML-10M -b 1 -m 3 -s ./demo_save -t 0 -a 6

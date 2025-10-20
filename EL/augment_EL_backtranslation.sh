#!/bin/bash
# run_stratified_crf.sh
# This script activates the virtual environment, installs dependencies,
# and runs the CRF NER pipeline with stratified split.

# Exit immediately if a command exits with a non-zero status
set -e

# Activate virtual environment
source ./myenv/bin/activate

# Run the CRF NER script with stratified split
python ./augment_EL_backtranslation.py

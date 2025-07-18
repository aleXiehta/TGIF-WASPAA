#!/bin/bash

# Description:
# This script generates audio mixtures using the DynamicMixing Python module.
# It sets up the required parameters and runs the Python script with those parameters.

# Usage:
#   chmod +x generate_mixture.sh
#   ./generate_mixture.sh

# Paths to datasets (update these paths to your actual dataset locations)
CLEAN_DATASET="preproc/metadata/clean_train_dns.txt"
BG_NOISE_DATASET="preproc/metadata/noise_train_dns.txt"    # Set to None if not used
BB_NOISE_DATASET="preproc/metadata/clean_train_dns.txt"    # Set to None if not used
RIR_DATASET="preproc/metadata/rir_trcv.txt"              # Set to None if not used

# Output directory (update to your desired output directory)
SAVED_DIR="TGIF-Dataset/tr" # replace with "TGIF-Dataset/cv" for validation set

# Parameters (you can adjust these as needed)
SNR_RANGE="-5,25"
SIR_RANGE="-5,25"
MAX_BG_NOISE_TO_MIX=1
MAX_BB_NOISE_TO_MIX=4
FIX_NUM_SPEAKERS="false"
REVERB_PROPORTION=0.8
TARGET_LEVEL=-25
TARGET_LEVEL_FLOATING_VALUE=10
ALLOWED_OVERLAPPED_BG_NOISE="true"
SUB_SAMPLE_DURATION=10
TOTAL_HOURS=50
SILENCE_LENGTH=0.2

# Python script to run (ensure this is the correct path)
PYTHON_SCRIPT="preproc/generate.py"

# Run the Python script with the specified arguments
python "$PYTHON_SCRIPT" \
    --clean_dataset "$CLEAN_DATASET" \
    --bg_noise_dataset "$BG_NOISE_DATASET" \
    --bb_noise_dataset "$BB_NOISE_DATASET" \
    --rir_dataset "$RIR_DATASET" \
    --snr_range="$SNR_RANGE" \
    --sir_range="$SIR_RANGE" \
    --max_bg_noise_to_mix $MAX_BG_NOISE_TO_MIX \
    --max_bb_noise_to_mix $MAX_BB_NOISE_TO_MIX \
    --fix_num_speakers $FIX_NUM_SPEAKERS \
    --reverb_proportion $REVERB_PROPORTION \
    --target_level $TARGET_LEVEL \
    --target_level_floating_value $TARGET_LEVEL_FLOATING_VALUE \
    --allowed_overlapped_bg_noise $ALLOWED_OVERLAPPED_BG_NOISE \
    --sub_sample_duration $SUB_SAMPLE_DURATION \
    --total_hours $TOTAL_HOURS \
    --silence_length $SILENCE_LENGTH \
    --saved_dir "$SAVED_DIR"

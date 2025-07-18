#!/bin/bash

# Description:
# This script generates audio mixtures using the DynamicMixing Python module.
# It sets up the required parameters and runs the Python script with those parameters.

# Usage:
#   chmod +x generate_mixture.sh
#   ./generate_mixture.sh

# Paths to datasets (update these paths to your actual dataset locations)
BG_NOISE_DATASET="preproc/metadata/noise_test_family.txt"    # Set to None if not used

# Parameters (you can adjust these as needed)
SNR_RANGE="-15,15"
SIR_RANGE="-15,15"
MAX_BG_NOISE_TO_MIX=1
FIX_NUM_SPEAKERS="false"
REVERB_PROPORTION=0.8
TARGET_LEVEL=-25
TARGET_LEVEL_FLOATING_VALUE=10
ALLOWED_OVERLAPPED_BG_NOISE="true"
SUB_SAMPLE_DURATION=10
TOTAL_HOURS_TR=1
TOTAL_HOURS_CV=0.5
TOTAL_HOURS_TT=1
SILENCE_LENGTH=0.2

# Python script to run (ensure this is the correct path)
PYTHON_SCRIPT="preproc/generate.py"

# Loop through family_0 to family_19
for i in {0..19}
do
    CLEAN_DATASET="preproc/metadata/simple_families/family_${i}.txt"
    RIR_DATASET="preproc/metadata/simple_families/rirs.txt"              # Set to None if not used
    MAX_BB_NOISE_TO_MIX=$((i/4))
    echo "family_${i} max_bb_noise_to_mix: $MAX_BB_NOISE_TO_MIX"
    # Define directories for training, validation, and test sets
    SAVED_DIR_TR="TGIF-Dataset/tt/family_${i}/tr"
    SAVED_DIR_CV="TGIF-Dataset/tt/family_${i}/cv"
    SAVED_DIR_TT="TGIF-Dataset/tt/family_${i}/tt"

    # Generate training set (1 hour)
    python "$PYTHON_SCRIPT" \
        --clean_dataset "$CLEAN_DATASET" \
        --bg_noise_dataset "$BG_NOISE_DATASET" \
        --bb_noise_dataset "$CLEAN_DATASET" \
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
        --total_hours $TOTAL_HOURS_TR \
        --silence_length $SILENCE_LENGTH \
        --saved_dir "$SAVED_DIR_TR"

    # Generate validation set (0.5 hours)
    python "$PYTHON_SCRIPT" \
        --clean_dataset "$CLEAN_DATASET" \
        --bg_noise_dataset "$BG_NOISE_DATASET" \
        --bb_noise_dataset "$CLEAN_DATASET" \
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
        --total_hours $TOTAL_HOURS_CV \
        --silence_length $SILENCE_LENGTH \
        --saved_dir "$SAVED_DIR_CV"

    # Generate test set (1 hour)
    python "$PYTHON_SCRIPT" \
        --clean_dataset "$CLEAN_DATASET" \
        --bg_noise_dataset "$BG_NOISE_DATASET" \
        --bb_noise_dataset "$CLEAN_DATASET" \
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
        --total_hours $TOTAL_HOURS_TT \
        --silence_length $SILENCE_LENGTH \
        --saved_dir "$SAVED_DIR_TT"
done

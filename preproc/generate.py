import os
import argparse
from tqdm import tqdm
import pandas as pd
import multiprocessing
from distutils.util import strtobool
from DynamicMixing import DynamicMixing
import random  # Import the random module for shuffling
import json
import glob  # For reading metadata files


def process_file(args_tuple):
    file_id, clean_path, args_dict = args_tuple
    saved_dir = args_dict['saved_dir']
    # Create a new mixer instance inside the function
    mixer = DynamicMixing(**args_dict)
    metadata = mixer.generate(file_id, clean_path, save_to_dir=True)
    # Remove 'clean' and 'noisy' keys if they exist to reduce data size
    metadata.pop('clean', None)
    metadata.pop('noisy', None)
    # Write metadata to disk
    metadata_filename = os.path.join(saved_dir, f"metadata_{file_id}.json")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f)
    # Return None to avoid collecting data in the main process
    return None

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='GENERATE MIXTURE')
    args.add_argument('--clean_dataset', type=str, required=True,
                      help='a text file containing list of clean speech audio paths')
    args.add_argument('--bg_noise_dataset', type=str, default=None,
                      help='Default is None')
    args.add_argument('--bb_noise_dataset', type=str, default=None,
                      help='Default is None')
    args.add_argument('--rir_dataset', type=str, default=None,
                      help='Default is None')
    args.add_argument('--snr_range', type=lambda x: [int(item) for item in x.split(',')], default="-5,25",
                      help='Background noise level. Default is [-5, 25].')
    args.add_argument('--sir_range', type=lambda x: [int(item) for item in x.split(',')], default="-5,25",
                      help='Bubble noise level. Default is [-5, 25]')
    args.add_argument('--max_bg_noise_to_mix', type=int, default=3,
                      help='Default is 3')
    args.add_argument('--max_bb_noise_to_mix', type=int, default=3,
                      help='Default is 3')
    args.add_argument('--fix_num_speakers', type=lambda x: bool(strtobool(x)), default="false",
                      help='Default is false')
    args.add_argument('--reverb_proportion', type=float, default=0.5,
                      help='Default is 0.5')
    args.add_argument('--target_level', type=int, default=-25,
                      help='Default is -25')
    args.add_argument('--target_level_floating_value', type=int, default=10,
                      help='Default is 10')
    args.add_argument('--allowed_overlapped_bg_noise', type=lambda x: bool(strtobool(x)), default="true",
                      help='Default is true')
    args.add_argument('--sub_sample_duration', type=float, default=10,
                      help='Default is 10')
    args.add_argument('--total_hours', type=float, default=1000,
                      help='Default is 1000')
    args.add_argument('--silence_length', type=float, default=0.2,
                      help='Default is 0.2')
    args.add_argument('--saved_dir', type=str, required=True)

    args = args.parse_args()
    clean_ds = args.clean_dataset

    # Convert arguments to a dictionary
    args_dict = vars(args)

    # Read the list of clean speech audio paths
    with open(clean_ds, "r") as f:
        clean_paths = [line.strip() for line in f]

    # Compute total_seconds and number of files to process
    total_seconds = args.total_hours * 3600
    num_files_to_process = int(total_seconds / args.sub_sample_duration)

    print(f"Total seconds to process: {total_seconds}")
    print(f"Sub-sample duration: {args.sub_sample_duration}")
    print(f"Number of files to process: {num_files_to_process}")

    selected_clean_paths = []

    num_available_paths = len(clean_paths)

    if num_files_to_process <= num_available_paths:
        random.shuffle(clean_paths)
        selected_clean_paths = clean_paths[:num_files_to_process]
    else:
        num_full_repeats = num_files_to_process // num_available_paths
        remainder = num_files_to_process % num_available_paths

        for _ in range(num_full_repeats):
            random.shuffle(clean_paths)
            selected_clean_paths.extend(clean_paths)

        if remainder > 0:
            random.shuffle(clean_paths)
            selected_clean_paths.extend(clean_paths[:remainder])

    # Shuffle the final list to ensure overall randomness
    random.shuffle(selected_clean_paths)

    # Prepare the list of tasks
    tasks = [(file_id, clean_path, args_dict) for file_id, clean_path in enumerate(selected_clean_paths)]

    # Initialize the multiprocessing pool with the desired number of processes
    with multiprocessing.Pool(processes=64) as pool:
        # Use imap_unordered for better performance and tqdm for progress bar
        # Do not collect the return values to avoid high memory usage
        list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)))

    # After the pool is closed and joined, read all metadata files and combine
    metadata_files = glob.glob(os.path.join(args.saved_dir, 'metadata_*.json'))
    metadata_list = []
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            metadata_list.append(metadata)

    mixture_metadata = pd.DataFrame(metadata_list)
    mixture_metadata.to_csv(os.path.join(args.saved_dir, "metadata.csv"), index=False)

    # Optionally, delete individual metadata files if no longer needed
    for metadata_file in metadata_files:
        os.remove(metadata_file)
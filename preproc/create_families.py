import os
from glob import glob
import random

# Define family size
family_size = 5

# Directory containing the wave files
base_dir = "/path/to/clean/corpus"

# Get all wave files and shuffle
wave_files = glob(os.path.join(base_dir, '**/*.wav'), recursive=True)
random.shuffle(wave_files)
num_speakers = len(wave_files)

# Assign families with a fixed size of 5
families = []
i = 0

while i < num_speakers:
    if i + family_size <= num_speakers:
        families.append(wave_files[i:i + family_size])
        i += family_size
    else:
        remaining_speakers = wave_files[i:]
        while len(remaining_speakers) < family_size:
            remaining_speakers.append(random.choice(wave_files[:i]))
        families.append(remaining_speakers)
        break

# Create output directory for family files
base_output_dir = "preproc/metadata/families"
os.makedirs(base_output_dir, exist_ok=True)

# Generate a text file for each family
for i, family in enumerate(families):
    family_file = os.path.join(base_output_dir, f'family_{i}.txt')
    with open(family_file, 'w') as f:
        for file in family:
            f.write(file + '\n')

print(f"Generated {len(families)} family files in {base_output_dir}")

def save_noise_test_file():
    noise_dir = "/path/to/demand"
    noise_files = glob(os.path.join(noise_dir, '**/*.wav'), recursive=True)
    output_file = "preproc/metadata/noise_test.dns.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for file in noise_files:
            f.write(file + '\n')

save_noise_test_file()

print(f"Generated noise test file in preproc/metadata/noise_test_dns.txt")

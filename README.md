# TGIF-WASPAA
Official Implementation of "TGIF: Talker Group-Informed Familiarization of Target Speaker Extraction," WASPAA, 2025
<img src="assets/overview.png" alt="TGIF Architecture" width="400"/> 

# Usage
Please download the pDNS subset of [DNS Challenge 2024](https://github.com/microsoft/DNS-Challenge).
## Data Generation
Run ```generate_dataset.sh``` for training/validation sets.
Run ```generate_test_dataset.sh``` for TGIF test set.

## Training & Adaptation
For teacher models' training:

TD-Speakerbeam: ```python train_tdsb.py --config config/config_tdsb.yaml```

SpEx+: ```python train_spex_plus.py --config config/config_tdsb.yaml```

See more configs ```config``` for student models.

For adaptation, you need to first generate the teacher's outputs, and replace the training script with ```adapt_tdsb.py```.
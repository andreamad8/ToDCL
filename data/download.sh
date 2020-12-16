#!/bin/bash
# SGD dataset
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git


#!/bin/bash
# TM
git clone https://github.com/google-research-datasets/Taskmaster.git

# MWOZ
pip install absl-py
git clone https://github.com/budzianowski/multiwoz.git
cd multiwoz/data
unzip MultiWOZ_2.1
cd MultiWOZ_2.2
python convert_to_multiwoz_format.py --multiwoz21_data_dir=../MultiWOZ_2.1 --output_file=data.json
cd ../../../../

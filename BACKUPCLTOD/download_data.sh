#!/bin/bash
# SGD dataset
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git

# # MetalWoz
# wget https://download.microsoft.com/download/E/B/8/EB84CB1A-D57D-455F-B905-3ABDE80404E5/metalwoz-v1.zip
# unzip metalwoz-v1.zip
# rm metalwoz-v1.zip

# SMMID
git lfs install
git clone https://github.com/facebookresearch/simmc.git
#!/bin/bash
# TM
git clone https://github.com/google-research-datasets/Taskmaster.git

# MWOZ
pip install absl-py
git clone https://github.com/budzianowski/multiwoz.git
cd multiwoz/data
unzip MultiWOZ_2.2
unzip MultiWOZ_2.1
cd MultiWOZ_2.2
python convert_to_multiwoz_format.py --multiwoz21_data_dir=../MultiWOZ_2.1 --output_file=data.json
cd ../../../

# DATAFLOW
# wget https://smresearchstorage.blob.core.windows.net/smcalflow-public/smcalflow.full.data.tgz
# tar -xvzf smcalflow.full.data.tgz
# rm smcalflow.full.data.tgz
# mkdir flow
# mv train.dataflow_dialogues.jsonl flow
# mv valid.dataflow_dialogues.jsonl flow

# Other install
pip install pytorch_lightning
pip install transformers
pip install tabulate
# pip install jsonlines

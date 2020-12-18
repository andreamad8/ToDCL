#!/bin/bash

## SCORE SGD
# python scorer.py --model_checkpoint runs_INTENT/SGD --task_type INTENT
# python scorer.py --model_checkpoint runs_NLG/SGD --task_type NLG
# python scorer.py --model_checkpoint runs_DST/SGD --task_type DST
# python scorer.py --model_checkpoint runs_E2E/SGD --task_type E2E

### SGD+MWOZ+TM19+TM20
python scorer.py --model_checkpoint runs_INTENT/BEST --task_type INTENT --best
python scorer.py --model_checkpoint runs_NLG/BEST --task_type NLG --best
python scorer.py --model_checkpoint runs_DST/BEST --task_type DST --best
python scorer.py --model_checkpoint runs_E2E/BEST --task_type E2E --best

### SGD+MWOZ+TM19+TM20
# python scorer.py --model_checkpoint runs_INTENT/SGD,TM19,TM20,MWOZ --task_type INTENT --all
# python scorer.py --model_checkpoint runs_NLG/SGD,TM19,TM20,MWOZ --task_type NLG --all
# python scorer.py --model_checkpoint runs_DST/SGD,TM19,TM20,MWOZ --task_type DST --all
# python scorer.py --model_checkpoint runs_E2E/SGD,TM19,TM20,MWOZ --task_type E2E --all

### SGD+MWOZ+TM19+TM20
python scorer.py --model_checkpoint runs_INTENT/BESTALL --task_type INTENT --all --best
python scorer.py --model_checkpoint runs_NLG/BESTALL --task_type NLG --all --best
python scorer.py --model_checkpoint runs_DST/BESTALL --task_type DST --all --best
python scorer.py --model_checkpoint runs_E2E/BESTALL --task_type E2E --all --best
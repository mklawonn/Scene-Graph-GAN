#!/usr/bin/env bash
python dataset_creation/grab_data.py; python dataset_creation/arrange_data.py; python preprocessing/recreateXuEtAlSplit.py; python train.py --vg_batches ./data/xu_et_al_batches/ --GPU 1 --batch_size 256 --dataset_relations_only True &; python train.py --vg_batches ./data/xu_et_al_batches/ --GPU 2 --batch_size 256 --dataset_relations_only True --use_language True
    

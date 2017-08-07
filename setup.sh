#!/usr/bin/env bash
python dataset_creation/grab_data.py
python dataset_creation/arrange_data.py
python preprocessing/recreateXuEtAlSplit.py

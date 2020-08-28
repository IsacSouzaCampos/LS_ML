#!/bin/bash

sh clean_aigs.sh
python split_pla_multi.py
python make_c50_files.py
python train_trees.py

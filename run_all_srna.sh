#!/bin/bash
cd "$(dirname "$0")"
python3 length_and_unique_srnas.py samples.csv --folder aligned_tables --group HEK_KUNV --region 10399 11015 > log_HEK_KUNV.txt 2>&1
python3 length_and_unique_srnas.py samples.csv --folder aligned_tables --group A549_KUNV --region 10399 11015 > log_A549_KUNV.txt 2>&1
python3 length_and_unique_srnas.py samples.csv --folder aligned_tables --group HEK_JEV --region 10395 10964 > log_HEK_JEV.txt 2>&1
python3 length_and_unique_srnas.py samples.csv --folder aligned_tables --group HEK_NY99 --region 10399 11029 > log_HEK_NY99.txt 2>&1
echo done > run_all_done.txt

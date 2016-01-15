#!/bin/bash
#K-Nearest Neighbours

#python file: knn_submission.py
#all these input file should be in current directory
#input file: 	train.csv --complete training data
#		devices --list of devices
#		test-means.csv -- mean of test sequences
#steps to run:

python knn_submission.py > sample_submission.csv  

for i in {1..90024}; do echo $i; done > nos

paste -d "," nos sample_submission.csv > final_submission

rm nos sample_submission.csv

echo "QuestionId,IsTrue" > file_to_submit
cat final_submission >> file_to_submit

mv file_to_submit final_submission_knn.csv

rm file_to_submit final_submission

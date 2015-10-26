#!/bin/bash

ls -hal aclImdb/train/pos | awk '{print "aclImdb/train/pos/"$9",1"}' | grep txt > train_tmp.csv
ls -hal aclImdb/train/neg | awk '{print "aclImdb/train/neg/"$9",0"}' | grep txt >> train_tmp.csv
shuf train_tmp.csv > train.csv
sed -i -e '1ifilename,target\' train.csv
rm train_tmp.csv

ls -hal aclImdb/test/pos | awk '{print "aclImdb/test/pos/"$9",1"}' | grep txt > test_tmp.csv
ls -hal aclImdb/test/neg | awk '{print "aclImdb/test/neg/"$9",0"}' | grep txt >> test_tmp.csv
shuf test_tmp.csv > test.csv
sed -i -e '1ifilename,target\' test.csv
rm test_tmp.csv


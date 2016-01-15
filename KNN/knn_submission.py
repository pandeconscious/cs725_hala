#!/usr/bin/env python

#k-nearest neighbor classification

import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cPickle

#declarations
k=12000 #k in the k-nearest neighbors
train_set = 'train.csv' #the training data file name 
classifier_dump_pickle = 'nn_classifier_complete_k10000_weighted_distance.obj' #the name of the object file that stores the classifier

trainfile = open(train_set, 'r')

datareader = csv.reader(trainfile)

X_list = [] #2-D list for storing features
Y_list = [] #1-D lisit for storing target labels 

#line_read =0

for row in datareader:
	if row[0].isdigit(): #ignore headers
		Y_list.append(int(row[4]))
		X_list.append(map(float, row[0:4]))#using X,Y,Z comp of Acceleration(Time not considered)
		#line_read += 1
		#print("Line reading:", line_read) 

trainfile.close()

Y_array = np.array(Y_list) #convert lists into numpy arrays
X_array = np.array(X_list) #convert lists into numpy arrays

del Y_list
del X_list

nn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance') #n_neighbors is the value of k
nn_classifier.fit(X_array, Y_array)

file_classifier = open(classifier_dump_pickle, 'w')
#cPickle.dump(nn_classifier, file_classifier) #not dumping the model..too time consuming and is not loaded afterwards so, of no use.

questionsfile = open('testMeans-devices.csv', 'r')
questionsreader = csv.reader(questionsfile)

device_list_crude = open('devices').readlines()
device_list = []
for item in device_list_crude:
	device_list.append(int(item))

X_list = [] #2-D list for storing features
Y_list = [] #1-D lisit for storing target labels

for row in questionsreader:
		Y_list.append(int(row[3]))
		X_list.append(map(float, row[0:3]))#using X,Y,Z comp of Acceleration(Time not considered)

questionsfile.close()

X_array = np.array(X_list) #convert lists into numpy arrays

del X_list


for i in range(len(X_array)):
	predict_array = nn_classifier.predict_proba(X_array[i])
	index = device_list.index(Y_list[i]) #read index from the device_list matching Y_list
	print predict_array[0][index]

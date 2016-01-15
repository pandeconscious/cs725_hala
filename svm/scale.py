#!/usr/bin/python

#scale the data to [0,1] range..


import sys
import csv
import numpy as np
import cPickle
import sklearn.preprocessing as prep

 

with open("../../data/train.csv",'rb') as csvTrainfile: #reading actual train.csv file..
	csvTrainReader = csv.reader(csvTrainfile,delimiter=',')
	
	train_vector = np.empty((29563983,3))
	class_vector = np.empty(29563983)

	print "Creating the training array"


	#creating training data
	for index,row in enumerate(csvTrainReader):
        	train_vector[index]= [float(row[1]), float(row[2]), float(row[3])]
		class_vector[index] = int(row[4])


	print "Scaling the training array"

	min_max_scaler = prep.MinMaxScaler()
	scaled_train_vector = min_max_scaler.fit_transform(train_vector)
	
		
	print "Dumping the scaling model"

	cPickle.dump(min_max_scaler, open("scale.model","wb"))	
				
	print "Be patient, your results are finally written to output file"

	with open("../../data/train_scaled.csv",'wb') as csvSubmissionfile:
		csvSubmissionWriter = csv.writer(csvSubmissionfile,delimiter=',')

		#write code from here....
		for index,row in enumerate(scaled_train_vector):	
            		csvSubmissionWriter.writerow([row[0],row[1],row[2],int(class_vector[index])])


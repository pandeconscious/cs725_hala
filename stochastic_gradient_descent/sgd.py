#!/usr/bin/python


#Learn the model using stochastic gradient descent of train.csv and use it to predict the probability of representative of test sequence.


import sys
import csv
import numpy as np
import sklearn.linear_model as sklm
import cPickle

 

with open("../../data/questions.csv","rb") as csvQuestionsFile:
	csvQuestionReader = csv.reader(csvQuestionsFile,delimiter=',')
	
	train_vector = np.empty((29563983,3))
	class_vector = np.empty( 29563983 )

	print "Creating the training array"

	with open("../../data/train.csv",'rb') as csvTrainfile: #reading actual train.csv file..
		csvTrainReader = csv.reader(csvTrainfile,delimiter=',')

		#creating training data
		for index,row in enumerate(csvTrainReader):
        		train_vector[index]= [float(row[1]), float(row[2]), float(row[3])]
        		class_vector[index]= int(row[4])

	print "Training the data set using stochastic gradient descent"

	sgd = sklm.SGDClassifier(loss="modified_huber",penalty="l2")
	sgd.fit(train_vector, class_vector)
	
	print "Dumping the stochastic gradient descent model"

	cPickle.dump(sgd, open("sgd_modified_huber.model","wb"))	
				
	print "Creating Test vector"

	test_vector = np.empty((90024,3))

	with open("../../data/test_avg.csv", 'rb') as csvTestfile: #reading test_avg.csv file.. making prediction based on representative..
                csvTestReader = csv.reader(csvTestfile, delimiter=',')
		
		for index,row in enumerate(csvTestReader):
			test_vector[index] = [float(row[0]),float(row[1]),float(row[2])]

	deviceDict = {} #dictionary mapping deviceId --> Index (0..387)
	
	with open("../../data/devices", 'rb') as csvDevicefile: #reading test_avg.csv file.. making prediction based on representative..
                csvDeviceReader = csv.reader(csvDevicefile, delimiter=',')
		
		for index,row in enumerate(csvDeviceReader):
			deviceDict[int(row[0])] = index
		
	
	print "Calculating probability predictions for test data set"
			
	predictions = sgd.predict_proba(test_vector)


	print "Be patient, your results are finally written to output file"

	with open("submission_sgd_modified_huber.csv",'wb') as csvSubmissionfile:
		csvSubmissionWriter = csv.writer(csvSubmissionfile,delimiter=',')
		
		csvSubmissionWriter.writerow(["QuestionId","IsTrue"])

		#write code from here....
		for index,row in enumerate(csvQuestionReader):
			
				
			question_id = row[0]
			seq = int(row[1])
			device_id = int(row[2])
			
			prob = predictions[index][deviceDict[device_id]]

            		csvSubmissionWriter.writerow([question_id,prob])


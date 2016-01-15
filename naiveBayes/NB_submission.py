#!/usr/bin/python


#find the naive bayes estimates of test_avg.csv using parameters trained in train_avg.csv.


import sys
import csv
import math


def gaussian(x,mu,var):
        """ definition of the guassian function"""
        y = 1 / ( math.sqrt(2. * math.pi * var) )  * math.exp(-((x-mu+0.0)**2 / (2.*var)))
        return y


with open("questions.csv","rb") as csvQuestionsFile:
	csvQuestionReader = csv.reader(csvQuestionsFile,delimiter=',')

	with open("test_avg.csv", 'rb') as csvTestfile:
        	csvTestReader = csv.reader(csvTestfile, delimiter=',')

		testDict = {}
		#creating testing data..	
		for row in csvTestReader:
			testDict[int(row[7])] = [float(row[0]),float(row[1]),float(row[2])]

	with open("train_avg.csv",'rb') as csvTrainfile:
		csvTrainReader = csv.reader(csvTrainfile,delimiter=',')

		#creating training data
		trainDict = {}
		for row in csvTrainReader:
			trainDict[int(row[7])] = [float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),int(row[6])]

	
	with open("submission_norm.csv",'wb') as csvSubmissionfile:
		csvSubmissionWriter = csv.writer(csvSubmissionfile,delimiter=',')

		csvSubmissionWriter.writerow(["QuestionId","IsTrue"])
		#write code from here....
		for row in csvQuestionReader:
			
			question_id = row[0]
			seq = int(row[1])
			seqProb = 0.0

			for devices in trainDict:
				device_id = int(devices)
				devData = trainDict[device_id]
				seqData = testDict[seq]
				prob = 1.0 * (devData[6] / 29563983.)	
				prob = prob * gaussian(seqData[0],devData[0],devData[3]) 
				prob = prob * gaussian(seqData[1],devData[1],devData[4])
				prob = prob * gaussian(seqData[2],devData[2],devData[5])
				seqProb = seqProb + prob
				

			device_id = int(row[2])

			devData = trainDict[device_id]
			seqData = testDict[seq]
			
			prob = 1.0 * (devData[6] / 29563983.)	
			prob = prob * gaussian(seqData[0],devData[0],devData[3]) 
			prob = prob * gaussian(seqData[1],devData[1],devData[4])
			prob = prob * gaussian(seqData[2],devData[2],devData[5])

			norm_prob = prob / seqProb
            		csvSubmissionWriter.writerow([question_id,norm_prob])

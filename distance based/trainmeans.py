from numpy import genfromtxt, savetxt, mean, delete
from sys import *

def main(): 
	print 'Begin loading data'
	dataset = genfromtxt(open('data.csv','r'), delimiter=',', dtype='f8')[1:] # read data from csv file
	dataset = delete(dataset, 1, 1) # drop the timestamp column
	print 'Data loading over'
	print 'getting list of devices...'
	devices = dataset[:,3] 
	devices = list(set(devices)) # remove duplicates
	allmeans={}
	print 'calculating means for each device...'
	for device in devices:
		dataOfDevice = filter(lambda l:l[-1]==device , dataset)
		means = mean(dataOfDevice , axis=0)
		allmeans[device]=means;
	print 'done'
	print allmeans[8.0][2]
	print allmeans;	
	outfile = open('means.txt','w')
	outfile.write(str(allmeans))
	outfile.close()
if __name__=="__main__":
	main() 



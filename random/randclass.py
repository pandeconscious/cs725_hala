#sg
#Predict 0/1 totally randomly
from random import *
import math
def randomPy(ipFile):
    RANGE_MAX = 100000
    ipf = open(ipFile, 'r')
    print ipf.readline()
    for line in ipf:
         randomNumber = math.ceil(random() * RANGE_MAX)
         if(randomNumber % 2 == 0):
             print "%s,%d" %(line.split(',')[0], 1)
         else:
             print "%s,%d" %(line.split(',')[0], 0)


if __name__ == '__main__':
    randomPy('../sampleSubmission.csv')

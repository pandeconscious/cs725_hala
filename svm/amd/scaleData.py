#sg
#script to scale the training data between 0 and 1
from sklearn import preprocessing
from numpy import loadtxt
import numpy


def main(fileName):
    
    dataset = loadtxt(open(fileName), delimiter = ' ', dtype = 'f8')
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset_scaled = min_max_scaler.fit_transform(dataset[:,0:4])
    classes = numpy.array(dataset[:,4])
    dataset_scaled = numpy.column_stack([dataset_scaled, classes])
    #output_file = open('data/scaledMean20.txt', 'w')
    #dataset_scaled.tofile(output_file, sep = ' ')
    numpy.savetxt('data/scaledMean20.txt', dataset_scaled, fmt='%.14f,%.14f,%.14f,%.14f,%d', newline = '\n')



if __name__ == '__main__':
    main('data/trainSamples20.txt')


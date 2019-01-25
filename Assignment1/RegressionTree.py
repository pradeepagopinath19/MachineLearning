import numpy as np
import math
def main():
    housingDataFile = open("housing_train.txt", "r")
    lines = housingDataFile.read().split('\n')

    featureValueMatrix = []
    for line in lines:
        values = line.split()
        if len(values) >0:
            #Empty rows present in the dataset towards the end
            featureValueMatrix.append(values)
    #print(featureValueMatrix)
    fvm = np.array(featureValueMatrix)
    print(fvm.shape)
    listOfMSE=[]
    for i in range (fvm.shape[1]):
        for j in range(fvm.shape[0]):
            left,right = split_data(fvm, i,j)
            leftMSE = calculateMeanSquareError(left)
            rightMSE = calculateMeanSquareError(right)
            listOfMSE.append(leftMSE+rightMSE)
            #exit()|
    print(listOfMSE)
    print("Minimum value is",min(listOfMSE))
    print("Maximum value of",max(listOfMSE))
    print("Size is",len(listOfMSE))

    for i in range(len(featureValueMatrix)):
        for j in range(len(featureValueMatrix[i])):
            #print(featureValueMatrix[i][j])
            a=i


def split_data(data, feature, threshold):
    left = []
    right = []

    print("Threshold value is:", data[threshold][feature])
    for i in range(len(data)):
        if float(data[i][feature]) < float(data[threshold][feature]):
            left.append(float(data[i][data.shape[1]-1]))
        else:
            right.append(float(data[i][data.shape[1]-1]))
    return (left, right)

def calculateMeanSquareError(values):
    sum = 0

    if(len(values)==0):
        return 0
    for i in range(len(values)):
        sum+=values[i]

    mean = sum/len(values)
    print('Mean is', mean)
    #print('Value array contents are', values)
    mse =0

    for i in range(len(values)):
        mse+= math.pow(values[i]-mean,2)
    return mse

if __name__ == "__main__":
    main()
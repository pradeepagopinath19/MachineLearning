import numpy as np
import math
def main():
    housingDataFile = open("housing_train.txt", "r")
    lines = housingDataFile.read().split('\n')
    #print(lines)
    featureValueMatrix = []

    for line in lines:
        values = [float(x) for x in line.split()]
        #print(values)
        # Empty rows present in the dataset towards the end
        if len(values) == 0:
            continue

        featureValueMatrix.append(values)
    #print(featureValueMatrix)

    fvm = np.array(featureValueMatrix)
    #print(fvm)
    #print(fvm.shape)

    mse_iteration={}

    minimum_mse = math.inf

    numberOfFeatures = fvm.shape[1]-1
    for i in range (numberOfFeatures):
        for j in range(fvm.shape[0]):
            left,right = split_data(fvm, i,j)
            leftMSE = calculateMeanSquareError(left)
            rightMSE = calculateMeanSquareError(right)
            mse_iteration[i,j] = leftMSE+rightMSE
            if mse_iteration[i,j] < minimum_mse:
                minimum_mse = mse_iteration[i,j]
                min_feature_column = i
                min_threshold =j

    print(mse_iteration)
    print(minimum_mse, min_feature_column, fvm[min_threshold][min_feature_column])
    print("Size is",len(mse_iteration))


def split_data(data, feature, threshold):
    left = []
    right = []

    print("Threshold value is:", data[threshold][feature])
    for i in range(len(data)):
        if float(data[i][feature]) < float(data[threshold][feature]):
            left.append(data[i][data.shape[1]-1])
        else:
            right.append(data[i][data.shape[1]-1])
    return left, right

def calculateMeanSquareError(values):

    mean = np.mean(values)
    #print('Mean is', mean)

    if(math.isnan(mean)):
        return 0

    print('Number of columns:', len(values))
    mse =0

    for i in range(len(values)):
        mse+= math.pow(values[i]-mean,2)
    return mse/len(values)

if __name__ == "__main__":
    main()
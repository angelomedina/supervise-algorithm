import pandas as pd;
import numpy as np;
import random;
import math;
import matplotlib.pyplot as plt;


#print(dataset[0]);

def loadcsv(file):
    dataset = pd.read_csv(file);
    dataset = dataset.values
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]




#print("trainnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn",train)
#print("testttttttttttttttttttttttttttttttttttttt",test)

#seperating as per the class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


#separated = separateByClass(dataset)
#Accessing seperated...
#print("sssssssssssseeeeeeeeeeeeeeeeeppppppppppppppppppppp",separated["s"]);
#print('Separated instances: {0}'.format(separated))


def mean(arr):
    return sum(arr)/float(len(arr))

def stdev(arr):
    avg=mean(arr)
    variance=sum([pow(x-avg,2) for x in arr])/float(len(arr)-1)
    return math.sqrt(variance)

#numbers = [1,2,3,4,5]
#print(numbers)
#print(mean(numbers))
#print(std(numbers))

#calculate mean and std classwise for each feature
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


#dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
#summary = summarizeByClass(dataset)
#print('Summary by class value:',summary);

#defining pdf

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


#summaries = {0:[(33, 2)], 1:[(9, 4)]}
#inputVector = [33, '?']
#probabilities = calculateClassProbabilities(summaries, inputVector)
#print('Probabilities for each class:',probabilities)

#predicting the best label
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#inputVector = [1.1, '?']
#result = predict(summaries, inputVector)
#print('Prediction:',result)

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

#summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#testSet = [[1.1, '?'], [19.1, '?']]
#predictions = getPredictions(summaries, testSet)
#print('Predictions:',predictions)


def sel_color(l):
    if l[-1]==0:
        return 'red'
    if l[-1]==1:
        return 'blue'

def sel_color_2(l):
    if l[0]==0:
        return 'red'
    if l[0]==1:
        return 'blue'


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        plt.scatter(testSet[x][1],testSet[x][2],marker='>', c=sel_color(testSet[x]))
        plt.scatter(testSet[x][1],testSet[x][2],marker='<', c=sel_color_2(predictions))
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

#testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
#predictions = ['a', 'a', 'a']
#accuracy = getAccuracy(testSet, predictions)
#print('Accuracy:',accuracy)

def main():
    filename="example.csv"
    dataset=loadcsv(filename)
    splitRatio = 0.8
    trainingSet, testSet = splitDataset(dataset, splitRatio)
   # print("trinnnnnnnnnnnnnnnnnnnnnnnnnn",trainingSet)
    #print("testtttttttt",testSet)
    summaries = summarizeByClass(trainingSet)
    print(summaries)
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy:',accuracy)
    plt.show()


if __name__ == '__main__':
    main()

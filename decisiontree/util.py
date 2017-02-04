from __future__ import division
import math

def calcEntropy(tableTrain):
    """Function which calculates entropy for the given input dataset.
    """
    classesTrain = []
    for row in tableTrain:
        classesTrain.append(int(row[-1]))

    S = []
    Splus = sum(classesTrain)
    Sminus = len(classesTrain)-sum(classesTrain)
    S = [Splus, Sminus]
    noOfSamples = sum(S)
    p = Splus/(Splus+Sminus)
    q = Sminus/(Splus+Sminus)
    if p == 0.0 or q == 0.0:
      entropy = 0.0
    else:
      entropy = -p * math.log(p,2)- q * math.log(q,2)
    return entropy
    

def calcVarianceImpurity(tableTrain):
    """Function which calculates the Variance Impurity for the given
    input dataset.
    """
    classesTrain = []
    for row in tableTrain:
        classesTrain.append(int(row[-1]))

    S = []
    K1 = sum(classesTrain)
    K0 = len(classesTrain)-sum(classesTrain)
    S = [K1, K0]
    K = sum(S)
    VI_S = (K0*K1)/K**2
    return VI_S

def infoGain(data, heuristic, attr):
    """Function which calculates the information gain for the given 
    input dataset, by selecting appropriate heuristics.
    """
    counts = {}
    
    for row in data:
        if(counts.has_key(row[attr])):
          counts[row[attr]] += 1
        else:
          counts[row[attr]] = 1
    
    subsetEntropy = 0

    for key in counts.keys():
        prob = counts[key]/sum(counts.values())
        subset = [x for x in data if x[attr] == key]
        subsetEntropy += prob * heuristic(subset) ######
    
    gain = heuristic(data) - subsetEntropy
    return gain

def mostCommonValue(examples):
    """Function which finds the most common value for class in the given examples.
    """
    classesTrain = []
    for row in examples:
        classesTrain.append(int(row[-1]))

    S = []
    Splus = sum(classesTrain)
    Sminus = len(classesTrain)-sum(classesTrain)
    S = [Splus, Sminus]
    if Splus>Sminus:
      return 1
    elif Sminus>Splus:
      return 0
    else:
      return 1

def getValues(examples,attr):
    """Function to get the attribute values of the dataset.
    """
    s = set()
    for row in examples:
        s.update([row[attr],])
    return s

def getSubset(examples,attr,val):
    """Function to get the subset of examples where the attribute is val
    """
    return [example for example in examples if example[attr] == val]
    
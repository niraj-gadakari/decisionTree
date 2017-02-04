from __future__ import division
from decisiontree.util import *
import random
import copy

class Node(object):
    """Class to create nodes to build the decision tree
    """
    def __init__(self,name,commonValue):
        self._name = name
        self._commonValue = commonValue
    
    def getName(self):
        return self._name
    
    def getClass(self):
        return self._commonValue
    
    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name
    

class induceDecisionTree:
    """Class to induce decision tree with a selected heuristic
    and with the training examples using ID3 algorithm. Also 
    the class implements randomized pruning method
    """
    entropy = 1
    varianceImpurity = 2
    
    def __init__(self, option):
        self._tree = {}
        self._nodes = []
        self._attributes = []
        
        if option == self.entropy:
            self.heuristic = calcEntropy
        elif option == self.varianceImpurity:
            self.heuristic = calcVarianceImpurity
    
    def fit(self, data, header, classAttribute):
        """Method to fit data into the decision tree.
        """
        attrList = [ x for x in range(len(header))]
        self._attributes = header
        indexes = header.index(classAttribute)
        #print(indexes)
        self._tree = self._id3(data,indexes,attrList,self.heuristic)
    
    def bestAttribute(self, examples, targetAttribute, attributes, heuristic):
        """Method to determine the best attribute to branch on.
        """
        best = (-1e9999,None)
        
        for attr in attributes:
            if attr != targetAttribute:
              gain = infoGain(examples, heuristic, attr)
              best = max(best,(gain,attr))
   
        return best[1]
    
    def treeWalk(self, tree, example, depth = 0):
        """Recursive Method for tree traversal to determine the 
        class labels given the examples.
        """
        if isinstance(tree,dict):
            node = tree.keys()[0]
            ind = self._attributes.index(node.getName())
            try:
                for k,v in tree[node].iteritems():
                    if(k == example[ind]):
                        return self.treeWalk(v,example,depth+1)
            except:
                return tree[node]
        return tree
    
    def treeTraverse(self,tree,depth):
        """Recursive Method to print the tree in the specified format.
        """
        if isinstance(tree,dict):
            node = tree.keys()[0]
            try:
                for k,v in tree[node].iteritems():
                    print '\n' + '|'*depth,
                    print "%s = %s : " %(node, k),
                    self.treeTraverse(v, depth+1)
            except:
                print tree[node],  
        else:
            print tree,
    def statistics(self):
        self.treeTraverse(self._tree,0)
    
    def accuracy(self,testFile,printFlag=1):
        """Method to determine the accuracy of the decision tree on
        the test dataset.
        """
        with open(testFile) as f:
            lines = f.readlines()

        lines = [x.strip() for x in lines] 
        tableTest = [x.split(',') for x in lines]
        header = tableTest.pop(0)
        classAttribute = header[-1]
        Splus = 0
        Sminus = 0
        indexes = self._attributes.index(classAttribute)
       
        for row in tableTest:
            label = self.treeWalk(self._tree,row)
            #print(label)
            if label == row[indexes]:
                Splus +=1
            else:
                Sminus += 1
        
        if printFlag:
            print "Testing decision tree on %s" %(testFile)
            print "Positive: ", Splus
            print "Negative: ", Sminus
            print "Accuracy: ", Splus/(Splus+Sminus)
        
        return (Splus,Sminus)

    
    def postPrune(self, dataset, L,K):
        """Method to prune the decision tree using randomized algorithm.
        """
        Splus,Sminus = self.accuracy(dataset,0)
        DBest = copy.deepcopy(self._tree)
        for i in range(L):
            Ddash = copy.deepcopy(self._tree)
            m = random.randint(1,K)
            for j in range(m):
                n = len(self._nodes)-1
                if n>0:
                    P = random.randint(0,n-1)
                else:
                    P = 0
                node = self._nodes[P].keys()[0]
                leaf = self._nodes[P][node]
                self._nodes[P][node] = node.getClass()
               
                Sp,Sn = self.accuracy(dataset,0)
                
                if (Sp/(Sp+Sn)) < (Splus/(Splus+Sminus)):
                    DBest = copy.deepcopy(self._tree)
                
                self._nodes[P][node] = leaf
        
        self._tree = DBest
       
    
    def _id3(self, examples, targetAttribute, attributes, heuristic):
        """Method that implements the ID3 algorithm to build the tree
        using the training dataset, and specified heuristic.
        """
        examples = examples[:]
        
        labels = [row[targetAttribute] for row in examples]
        commonValue = mostCommonValue(examples)
        
        if labels.count(labels[0]) == len(labels):
            return labels[0]
        elif (len(attributes)-1)<=0 or not examples:
            return commonValue
        else:
            best = self.bestAttribute(examples, targetAttribute, attributes, heuristic)
            root = Node(self._attributes[best],commonValue)
            tree = {root:{}}
            self._nodes.append(tree)
            
            for val in getValues(examples,best):
                remainingAttr = [attr for attr in attributes if attr!=best]
                subsetExamples = getSubset(examples,best,val)
                subtree = self._id3(subsetExamples,targetAttribute,remainingAttr,heuristic)
                tree[root][val] = subtree
        return tree        
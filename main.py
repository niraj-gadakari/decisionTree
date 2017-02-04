from decisiontree.induceDT import induceDecisionTree
import sys

def growTree(trainFile,option):
    """Function to build the decision tree using the training data.
    Use the heuristic option and the training examples to grow the tree.
    """
    with open(trainFile) as f:
        lines = f.readlines()

    lines = [x.strip() for x in lines] 
    tableTrain = [x.split(',') for x in lines]
    header = tableTrain.pop(0)
    classAttribute = header[-1]
   
    tree = induceDecisionTree(option)
    tree.fit(tableTrain,header,classAttribute)
    return tree


def main():
    """ Main function
    """
    if len(sys.argv) != 7:
        print('''
        Error:
        Please execute the code as: 
        python main.py <L> <K> <training_set> <validation_set> <test_set> <to_print>
        ''')
        sys.exit(1)
    
    L = int(sys.argv[1])
    K = int(sys.argv[2])
    trainFile = sys.argv[3]
    validationFile = sys.argv[4]
    testFile = sys.argv[5]
    printFlag = sys.argv[6]
    
    infoGainHeuristic = 1
    varImpurityHeuristic = 2
    
    #Construct the decision tree using the training data and Information Gain Heuristics
    print "\n Statistics of Decision Tree with Information Gain Heuristics:"
    treeInfoGain = growTree(trainFile,infoGainHeuristic)
    treeInfoGain.accuracy(testFile)
    
    #Pruning
    print "\n Statistics of Decision Tree on Pruning (Information Gain Heuristics):"
    treeInfoGain.postPrune(validationFile,L,K)
    treeInfoGain.accuracy(testFile)
    
    #Construct the decision tree using the training data and Variance Impurity Heuristics    
    print "\n Statistics of Decision Tree with Variance Impurity Heuristics:"
    treeVarImp = growTree(trainFile,varImpurityHeuristic)
    treeVarImp.accuracy(testFile)
    
    #Pruning
    print "\n Statistics of Decision Tree on Pruning (Variance Impurity Heuristics):"
    treeVarImp.postPrune(validationFile,L,K)
    treeVarImp.accuracy(testFile)
    
    #Print the decision tree
    if printFlag == "yes":
      print "\n Decision Tree with Information Gain Heuristics: "
      treeInfoGain.statistics()
      print "\n\n Decision Tree with Variance Impurity Heuristics: "
      treeVarImp.statistics()


if __name__ == '__main__':
    main()
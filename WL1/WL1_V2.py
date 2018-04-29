import CoraDataLoader as CoraDataLoader
import math as math
import statistics as st
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt

def drawGraph(labelDict,adjacencyDict):
    edges =[]
    
    for key, value in adjacencyDict.items():
        for node in value:
            edge= (key,node)
            edges.append(edge)          
    
            
    G = nx.Graph()
    G.add_edges_from(edges)

    val_map = {'Neural_Networks': 'r',
               'Rule_Learning': 'g',
               'Reinforcement_Learning': 'b',
               'Neural_Networks': 'y',
               'Rule_Learning': 'c',
               'Reinforcement_Learning': 'm',
               'Neural_Networks': 'y'}
    
    values = [val_map.get(labelDict[node], 0.25) for node in G.nodes()]
    
    nx.draw(G, cmap = plt.get_cmap('jet'), node_color = values)
    plt.show()
    return

def dictFindReplace(dict, findValue,ReplaceValue):
    for key,value in dict.items():
        if(value==findValue):
            dict[key] = ReplaceValue    
    return

def unTranslate(originalDict, optimizedDict):
    optimizedDictClone = optimizedDict.copy()
    for key, value in optimizedDictClone.items():
        if (value == 1):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,1,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 2):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone, 2,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 3):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,3,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 4):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,4,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 5):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,5,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 6):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,6,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 0):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,0,replaceValue)
            break
        
    return optimizedDictClone

def untranslateAndGetAccuracy(originalDict, optimizedDict):
    
    optimizedDictClone = optimizedDict.copy()
    for key, value in optimizedDictClone.items():
        if (value == 1):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,1,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 2):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone, 2,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 3):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,3,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 4):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,4,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 5):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,5,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 6):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,6,replaceValue)
            break
    for key, value in optimizedDictClone.items():
        if (value == 0):
            replaceValue = originalDict[key]
            dictFindReplace(optimizedDictClone,0,replaceValue)
            break
    accuracy =getAccuracy(originalDict,optimizedDictClone)  
    print(Counter(optimizedDictClone.values()))   
    return accuracy

def getAccuracy(originalDict, optimizedDict):
    accuracy = 0.0
    for key,value in originalDict.items():
        if(value == optimizedDict[key]):
            accuracy = accuracy +1
    
    accuracy = accuracy/len(originalDict.keys())
    return accuracy
    

dataLoader = CoraDataLoader.CoraDataLoader(75)

for key, value in dataLoader.nodeClassDictToOptimize.items():
    dataLoader.nodeClassDictToOptimize[key] = 1

accuracy = untranslateAndGetAccuracy(dataLoader.nodeClassDictWOTranslation, dataLoader.nodeClassDictToOptimize)
print("accuracy before starting is " +  str(accuracy))
print(Counter(dataLoader.nodeClassDictToOptimize.values()))

print("actual relative distribution of classes is ")
print(Counter(dataLoader.nodeClassDictWOTranslation.values()))

'''WL algo'''

numberOfEpochs = 2000
iterationCount = 0
accuracyThreshold = 0.90

while(accuracy< accuracyThreshold and iterationCount<numberOfEpochs):
    for key,value in dataLoader.nodeClassDictToOptimize.items():
        
        sum = 0 
        for adjacentNode in dataLoader.adjacenyDict[key]:
            sum = sum +  dataLoader.nodeClassDictToOptimize[adjacentNode]
            
        dataLoader.nodeClassDictToOptimize[key] = sum % 7      
          
    iterationCount = iterationCount + 1
    accuracy = untranslateAndGetAccuracy(dataLoader.nodeClassDictWOTranslation, dataLoader.nodeClassDictToOptimize)
      
    print("accuracy in iteration " + str(iterationCount) + " is " + str(accuracy))
    #print(Counter(dataLoader.nodeClassDictToOptimize.values()))
    
optimizedUntranslated = unTranslate(dataLoader.nodeClassDictWOTranslation, dataLoader.nodeClassDictToOptimize)
drawGraph(dataLoader.nodeClassDictWOTranslation, dataLoader.adjacenyDict)

drawGraph(optimizedUntranslated, dataLoader.adjacenyDict)
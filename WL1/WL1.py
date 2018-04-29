import CoraDataLoader as CoraDataLoader
import math as math
import statistics as st
from collections import Counter

def getAccuracy(originalDict, optimizedDict):
    accuracy = 0.0
    for key,value in originalDict.items():
        if(value == optimizedDict[key]):
            accuracy = accuracy +1
    
    accuracy = accuracy/len(originalDict.keys())
    return accuracy
    

dataLoader = CoraDataLoader.CoraDataLoader(75)
accuracy = getAccuracy(dataLoader.nodeClassDict, dataLoader.nodeClassDictToOptimize)
print("accuracy before WL is")
print(accuracy)
 
 
i=0
for key, value in dataLoader.nodeClassDictToOptimize.items():
    print(key)
    print(value)
    i=i+1
    if(i==5):
        break

'''WL algo'''

numberOfEpochs = 2000
accuracyThreshold = 0.90

iterationCount = 0

while(accuracy<accuracyThreshold and iterationCount<numberOfEpochs):
    for key,value in dataLoader.nodeClassDictToOptimize.items():
        labelList =[]
         
#         print("key is " + str(key))
#         print("adjacent nodes are ")
#         print(dataLoader.adjacenyDict[key])
        #print(dataLoader.adjacenyDict[key]) 
        for adjacentNode in dataLoader.adjacenyDict[key]:       
            labelList.append(dataLoader.nodeClassDictToOptimize[adjacentNode])
       
        #dataLoader.nodeClassDictToOptimize[key] = int(sum / len(dataLoader.adjacenyDict[key]))
        #dataLoader.nodeClassDictToOptimize[key] = sum % 7
        
#         print(labelList)
#         print("assigned " + str(dataLoader.nodeClassDictToOptimize[key]))
        if(len(labelList)>0):
            dataLoader.nodeClassDictToOptimize[key] = max(labelList,key=labelList.count) 
           
#         if(len(dataLoader.adjacenyDict[key]) != 0):
#             newLabel = int(sum / len(dataLoader.adjacenyDict[key]))
#             dataLoader.nodeClassDictToOptimize[key] = newLabel
#             print("assigning value " + str(newLabel))
            
         
           
    iterationCount = iterationCount + 1
    accuracy = getAccuracy(dataLoader.nodeClassDict, dataLoader.nodeClassDictToOptimize)
    print("accuracy in iteration " + str(iterationCount) + " is " + str(accuracy))
        






# i=0
# 
# for key, value in dataLoader.nodeClassDict.items():
#     print(key)
#     print(value)
#     i=i+1
#     if(i==5):
#         break
# 
# i= 0 
# print("------------------------------")    
# for key, value in dataLoader.nodeClassDictToOptimize.items():
#     print(key)
#     print(value)
#     i=i+1
#     if(i==5):
#         break
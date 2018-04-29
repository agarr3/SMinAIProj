'''
Created on 23-Apr-2018

@author: agarr
'''
import csv as csv
import random as random

class CoraDataLoader(object):
    '''
    classdocs
    '''


    def __init__(self,percentageAnonymization):
        '''
        Constructor
        '''
        self.nodeClassDict = {}
        self.nodeClassDictWOTranslation = {}
        self.adjacenyDict = {}
        
        self.classTranslation = {}        
        self.classTranslation["Case_Based"] = 0
        self.classTranslation["Genetic_Algorithms"] = 1
        self.classTranslation["Neural_Networks"] = 2
        self.classTranslation["Probabilistic_Methods"] = 3
        self.classTranslation["Reinforcement_Learning"] = 4
        self.classTranslation["Rule_Learning"] = 5
        self.classTranslation["Theory"] = 6
        
        self.percentageAnonymization = percentageAnonymization
        
        
        
        with open('./data/cora.content') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')            
            for row in readCSV:
                #self.x.append([float(i) for i in row[1:10]])
                if("?" not in row):
                    self.nodeClassDict[int(row[0])] =  self.classTranslation[row[-1]]
                    self.nodeClassDictWOTranslation[int(row[0])] =  row[-1]
                    
        with open('./data/cora.cites') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')            
            for row in readCSV:
                #self.x.append([float(i) for i in row[1:10]])
                if(int(row[0]) in self.adjacenyDict):
                    self.adjacenyDict[int(row[0])].append(int(row[1]))
                else:
                    self.adjacenyDict[int(row[0])] = []
                
                if(int(row[1]) in self.adjacenyDict):
                    self.adjacenyDict[int(row[1])].append(int(row[0]))
                else:
                    self.adjacenyDict[int(row[1])] = []
        
        self.nodeClassDictToOptimize = self.nodeClassDict.copy()
        totalNumberOfKeysToAnonymize = int((self.percentageAnonymization/100) * len(self.nodeClassDictToOptimize.keys()))             
        keysToAnonymize = random.sample(self.nodeClassDictToOptimize.keys(),totalNumberOfKeysToAnonymize)            
        for key,values in self.nodeClassDictToOptimize.items():
            if(key in keysToAnonymize):
                self.nodeClassDictToOptimize[key] = 1
                
                    
                    

# i=0
# dataLoader = CoraDataLoader(75)
#  
# for key, value in dataLoader.nodeClassDictToOptimize.items():
#     print(key)
#     print(value)
#     i=i+1
#     if(i==5):
#         break



                    
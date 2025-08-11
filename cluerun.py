from enum import IntEnum

import subprocess
import sys
from pathlib import Path
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import clueutil

#TODO Fix bug where one line is lost every clue round

#--------------------------- Settings ------------------------------------
class Algorithm(IntEnum):
    DBSCAN = 1
    IPLSHDBSCAN = 2
    KMEANS = 3

class DistanceMetric(IntEnum):
    EUCLIDEAN = 1
    ANGULAR = 2
    DTW = 3

class ParameterOptimizationLevel(IntEnum):
    NONE = 0
    KDISTANCE = 1
    EXPLORATION = 2
#-------------------------------------------------------------------------

#Configurations relating to how to run the clustering subprocess
class ClueConfig:
    
    #Enums
    algorithm = None            
    distanceMetric = None       
    paramOptimization = None
    
    #Numericals
    epsilon = None
    minPts = None
    hyperplanes = None
    hashtables = None
    kclusters = None
    threads = None

    #Booleans
    standardize = None
    useFeatures = None
    
    #Dictionaries for constructing the Java call to CLUECLUST for clustering stage
    callDictALG = {}
    callDictDM = {}
    
    #Refreshes the dictionaries containing the call information, is called prior to returning the full call
    #to ensure correct variable calls
    def refreshCallOpts(self):
        #Algorithm dict
        self.callDictALG[int(Algorithm.DBSCAN)] = [
                                                   "-c", "vanilla", 
                                                   "-e", str(float(self.epsilon)), 
                                                   "-m", str(int(self.minPts))
                                                   ]
        self.callDictALG[int(Algorithm.IPLSHDBSCAN)] = [
                                                        "-c", "lshdbscan", 
                                                        "-e", str(float(self.epsilon)), 
                                                        "-m", str(int(self.minPts)), 
                                                        "-L", str(int(self.hashtables)), 
                                                        "-M", str(int(self.hyperplanes))
                                                        ]
        self.callDictALG[int(Algorithm.KMEANS)] = [
                                                   "-c", "kmeans", 
                                                   "-k", str(int(self.kclusters))
                                                   ]

        #Distance metric dict
        self.callDictDM[int(DistanceMetric.EUCLIDEAN)] = []
        self.callDictDM[int(DistanceMetric.ANGULAR)] = ["-a"]
        self.callDictDM[int(DistanceMetric.DTW)] = [
                                                    "-dtw", 
                                                    "-window", str(int(self.window))
                                                    ]

    def __init__(self, 
                 algorithm=Algorithm.DBSCAN, 
                 distanceMetric=DistanceMetric.EUCLIDEAN,
                 paramOptimization=ParameterOptimizationLevel.NONE,
                 epsilon=1,
                 minPts=20,
                 hyperplanes=5,
                 hashtables=10,
                 kclusters=8,
                 standardize=False,
                 useFeatures=False,
                 threads=4,
                 window=20
                 ):
        self.algorithm = algorithm
        self.distanceMetric = distanceMetric
        self.paramOptimization=paramOptimization
        self.epsilon=epsilon
        self.minPts=minPts
        self.hyperplanes=hyperplanes
        self.hashtables=hashtables
        self.kclusters=kclusters
        self.standardize = standardize
        self.threads = threads
        self.window = window
        self.useFeatures = useFeatures
        self.refreshCallOpts
        
    #Generate config-dependent components of call for regular clustering usage
    def getConfigCallOpts(self):
        self.refreshCallOpts()
        callOpts = []
        callOpts += self.callDictALG[self.algorithm]
        callOpts += self.callDictDM[self.distanceMetric]
        if (self.standardize):
            callOpts.append("-standardize")
        return callOpts

    #Generate config-dependent components of call for parameter optimization usage
    def getOptimizeCallOpts(self):
        self.refreshCallOpts()
        callOpts = ["-c", "optimize"]
        callOpts += self.callDictDM[self.distanceMetric]
        callOpts.append("-o")
        callOpts.append(str(int(self.paramOptimization)))
        if (self.standardize):
            callOpts.append("-standardize")
        return callOpts


#An individual round is defined by files connected to that round and a configuration for the subprocess
class ClueRound:

    _round = None               #Round number in global ordering
    roundName = None            #Name of round
    directory = None            #Base directory of the global run
    roundDirectory = None       #Directory for the round itself
    clueConfig = None           #Config for the round
    
    inputFile = None            #File descriptor to the file containing the input data for the clustering round
    clustersFile = None         #File descriptor to direct the clustering results
    metadataFile = None         #File descriptor to direct the metadata results
    featureSelectionFile = None #File descriptor for the feature selection file
    clusterSelectionFile = None #File descriptor for the cluster selection file
  
    def __init__(self, 
                 roundName, 
                 _round, 
                 directory, 
                 featureSelectionFile, 
                 clusterSelectionFile,
                 clueConfig):
        self.roundName = roundName
        self._round = _round
        self.directory = directory
        self.roundDirectory = self.directory + "/" + roundName + "/"
        self.clueConfig = clueConfig
        
        self.featureSelectionFile = featureSelectionFile
        self.clusterSelectionFile = clusterSelectionFile
   
        self.clustersFile = "clusters" + str(self._round) + ".csv"
        self.metadataFile = "metadata" + str(self._round) + ".csv"
        self.inputFile = self.roundDirectory + "input" + str(self._round) + ".csv"

    #Builds the subprocess call based off the configuration of the round
    def buildCall(self):
        #Base call, always the same regardless of config
        call = ["java", "-jar", "Java-PH/CLUECLUST.jar", "-f", self.inputFile, 
                "-d", self.roundDirectory, "-output", self.clustersFile,
                "-metadata", self.metadataFile, "-t", str(self.clueConfig.threads)]

        #Generate config-dependent components of call
        call += self.clueConfig.getConfigCallOpts()
        return call

    
    #Special case for building a call using the parameter optimizer
    def runParamOptimizer(self):
        #Base call, always the same regardless of config
        call = ["java", "-jar", "Java-PH/CLUECLUST.jar", "-f", self.inputFile,
                "-d", self.roundDirectory, "-c", "optimize", "-o", str(int(self.clueConfig.paramOptimization))]

        #Generate config-dependent components of call
        call += self.clueConfig.getOptimizeCallOpts()

        subprocess.run(call)
        
        print("Reading best parameter optimization values...")
        paramOptFD = self.roundDirectory + "optimal_params.csv"
        paramOptDF = pd.read_csv(paramOptFD, header=0)
        if (len(paramOptDF) > 0):
            bestParam = paramOptDF.iloc[0]
            self.clueConfig.epsilon = float(bestParam["Epsilon"])
            self.clueConfig.minPts = int(bestParam["MinPts"])
            self.clueConfig.hashtables = int(bestParam["HashTables"])
            self.clueConfig.hyperplanes = int(bestParam["Hyperplanes"])
        else:
            print("Could not find any parameter combinations within acceptable quality range, exiting...")
            sys.exit(2)
        
        print("Best values found to be: \n")
        print("Epsilon = " + str(self.clueConfig.epsilon) + "\n")
        print("MinPts = " + str(self.clueConfig.minPts) + "\n")
        print("HashTables = " + str(self.clueConfig.hashtables) + "\n")
        print("Hyperplanes = " + str(self.clueConfig.hyperplanes) + "\n")

    #Runs the round
    def runRound(self):
        print("Running round " + str(self._round) + " by name: " + self.roundName + "...\n")
        print("Target round directory: " + self.roundDirectory + "\n")
        print("Input File: " + self.inputFile + "\n")
        print("Clusters File: " + self.roundDirectory + self.clustersFile + "\n")
        print("Metadata File: " + self.roundDirectory + self.metadataFile + "\n")
        print("Feature Selection File: " + self.featureSelectionFile + "\n")
        print("Cluster Selection File: " + self.clusterSelectionFile + "\n")

        #If param opt is selected we run the parameter optimizer first to set new config parameters
        if (int(self.clueConfig.paramOptimization) > 0):
            print("Running parameter optimizer with level: " + str(int(self.clueConfig.paramOptimization)) + "\n")
            self.runParamOptimizer()
        
        call = self.buildCall()
        compProc = subprocess.run(call)

#A full run of Clue consists of running a list of rounds
class ClueRun:
    rounds = []                     #Rounds to run
    runName = None                  #Name of run
    baseFile = None                 #Base input file, raw data
    baseFeaturesFile = None         #Base file containing features, does not need to exist prior to running, will be created in run directory
    targetRunDirectory = None    #Run directory
    outputDirectory = None
    interactive = None

    def __init__(self, runName, baseFile, targetRunDirectory, outputDirectory="output", interactive=False):
        self.runName = runName
        self.baseFile = baseFile
        self.targetRunDirectory = targetRunDirectory + "/" + self.runName
        self.outputDirectory = outputDirectory

    #For building a round and adding it to the end of the rounds to be ran
    def buildRound(self, roundName, featuresFile, selectionFile, clueConfig):
        self.rounds.append(ClueRound(roundName, 
                                     len(self.rounds) + 1, 
                                     self.targetRunDirectory, 
                                     featuresFile, 
                                     selectionFile,
                                     clueConfig))

    #Main run function
    def run(self):
        Path(self.targetRunDirectory).mkdir(parents=True, exist_ok=True)
        inputDF = pd.read_csv(self.baseFile)

        print("Building base features file...")
        self.baseFeaturesFile = self.targetRunDirectory + "/baseFeatures.csv"
        baseFeaturesDF = clueutil.FeatureExtractor.extractFromDataFrame(inputDF)
        baseFeaturesDF.to_csv(self.baseFeaturesFile, index=False)
        print("inputFeatures file written to " + self.baseFeaturesFile)
       
        #First round is ran without any cluster, metadata, or cluster selection files.
        currentDF = inputDF
        if (self.rounds[0].clueConfig.useFeatures):
            currentDF = baseFeaturesDF
        newInputs = clueutil.InputFilter.filter(currentDF, 
                                                None, #No previous metadata FD
                                                None, #No previous clusters FD
                                                self.rounds[0].
                                                featureSelectionFile, 
                                                None #No cluster selection file for first round
                                       ) 
        Path(self.rounds[0].roundDirectory).mkdir(exist_ok=True)
        newInputs.to_csv(path_or_buf=self.rounds[0].inputFile, header=False, index=False)
        prevRound = self.rounds[0]
        prevRound.runRound()

        #Subsequent rounds
        for i in range(1, len(self.rounds)):
            currRound = self.rounds[i]
            currentDF = inputDF
            if (currRound.clueConfig.useFeatures):
                currentDF = baseFeaturesDF
            prevMetadataFD = prevRound.roundDirectory + prevRound.metadataFile
            prevClustersFD = prevRound.roundDirectory + prevRound.clustersFile
            newInputs = clueutil.InputFilter.filter(currentDF, 
                                                    prevClustersFD, 
                                                    prevMetadataFD, 
                                                    currRound.
                                                    featureSelectionFile, 
                                                    currRound.clusterSelectionFile)
            Path(currRound.roundDirectory).mkdir(exist_ok=True)
            newInputs.to_csv(path_or_buf=currRound.inputFile, header=False, index=False)
            prevRound = currRound
            currRound.runRound()
        
        targetOutputDirectory = self.targetRunDirectory + "/" + self.outputDirectory
        print("Writing output files to directory: " + targetOutputDirectory + "\n")
        Path(targetOutputDirectory).mkdir(exist_ok=True)
        finalRound = self.rounds[len(self.rounds) - 1]
        print("Final round name: " + finalRound.roundName)
        print("Copying final round clusters file...")
        shutil.copy(finalRound.roundDirectory + finalRound.clustersFile, targetOutputDirectory + "/clusters_output.csv")
        print("Copying final round metadata file...")
        shutil.copy(finalRound.roundDirectory + finalRound.metadataFile, targetOutputDirectory + "/metadata_output.csv")
        print("Generating graphs from final round data...")
        clueutil.ClueGraphing.generateGraphs(finalRound, 
                                             self.baseFile, 
                                             self.baseFeaturesFile, 
                                             outputDirectory=targetOutputDirectory, 
                                             directOutput=self.interactive)


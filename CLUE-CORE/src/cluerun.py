from enum import IntEnum

import subprocess
from pathlib import Path
import shutil
import threading

import pandas as pd

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
    kClusters = None
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
                                                   "-k", str(int(self.kClusters))
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
                 kClusters=8,
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
        self.kClusters=kClusters
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
                 directory, 
                 featureSelectionFile, 
                 clusterSelectionFile,
                 clueConfig,
                 ):
        self.setRound(roundName, 
                      directory, 
                      featureSelectionFile, 
                      clusterSelectionFile,
                      clueConfig)
        
    #Change all round settings in the given round
    def setRound(self, 
                 roundName, 
                 directory, 
                 featureSelectionFile, 
                 clusterSelectionFile, 
                 clueConfig):
        self.roundName = roundName
        self.directory = directory
        self.roundDirectory = str(directory) + "/" + roundName + "/"
        self.featureSelectionFile = featureSelectionFile
        self.clusterSelectionFile = clusterSelectionFile
        self.clueConfig = clueConfig

        #TODO Fix inconsistency in managing file descriptors
        self.clustersFile = "clusters.csv"
        self.metadataFile = "metadata.csv"
        self.inputFile = self.roundDirectory + "input.csv"
    
    #Update the round directory to a new directory
    def updateRoundDirectory(self, directory):
        self.directory = directory
        self.roundDirectory = str(directory) + "/" + self.roundName + "/"
        self.inputFile = self.roundDirectory + "input.csv"

    #Builds the subprocess call based off the configuration of the round
    def buildCall(self, CLUECLUST):
        #Base call, always the same regardless of config
        call = ["java", "-jar", CLUECLUST, "-f", self.inputFile, 
                "-d", self.roundDirectory, "-output", self.clustersFile,
                "-metadata", self.metadataFile, "-t", str(self.clueConfig.threads)]

        #Generate config-dependent components of call
        call += self.clueConfig.getConfigCallOpts()
        return call

    
    #Special case for building a call using the parameter optimizer
    def runParamOptimizer(self, CLUECLUST):
        #Base call, always the same regardless of config
        call = ["java", "-jar", CLUECLUST, "-f", self.inputFile,
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
            raise clueutil.ClueException("Could not find any parameter combinations within acceptable quality range")
        
        print("Best values found to be: \n")
        print("Epsilon = " + str(self.clueConfig.epsilon) + "\n")
        print("MinPts = " + str(self.clueConfig.minPts) + "\n")
        print("HashTables = " + str(self.clueConfig.hashtables) + "\n")
        print("Hyperplanes = " + str(self.clueConfig.hyperplanes) + "\n")

    #Runs the round with a specified CLUECLUST jar file
    def runRound(self, CLUECLUST):
        print("Running round: " + self.roundName + "...")
        print("Using CLUECLUST jar file: " + CLUECLUST)
        print("Target round directory: " + self.roundDirectory)
        print("Input File: " + self.inputFile)
        print("Clusters File: " + self.roundDirectory + self.clustersFile)
        print("Metadata File: " + self.roundDirectory + self.metadataFile)
        print("Feature Selection File: " + str(self.featureSelectionFile))
        print("Cluster Selection File: " + str(self.clusterSelectionFile))

        #If param opt is selected we run the parameter optimizer first to set new config parameters
        if (int(self.clueConfig.paramOptimization) > 0):
            print("Running parameter optimizer with level: " + str(int(self.clueConfig.paramOptimization)) + "\n")
            self.runParamOptimizer(CLUECLUST)
        
        call = self.buildCall(CLUECLUST)
        proc = subprocess.run(call, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.STDOUT, 
                              text=True,
                              bufsize=1)
        
        #When running using the UI, the output must be printed to the cosnsole for visibility
        if threading.current_thread() is not threading.main_thread():
            for line in proc.stdout.splitlines():
                print(line + "\n", end="", flush=True)

#A full run of Clue
class ClueRun:
    rounds = []                     #Rounds to run
    runName = None                  #Name of run
    baseFile = None                 #Base input file, raw data
    baseFeaturesFile = None         #Base file containing features, does not need to exist prior to running, will be created in run directory
    targetRunDirectory = None       #Run directory
    outputDirectory = None          #Output directory for the final results, will be created if it does not exist
    interactive = None              #If false, will display graphs directly instead of saving them to a file
    CLUECLUST = None                #Path to the CLUECLUST jar file, must be set before running


    def __init__(self, runName, baseFile, targetRunDirectory, outputDirectory="output", interactive=False, CLUECLUST="CLUECLUST.jar"):
        self.runName = runName
        self.baseFile = baseFile
        self.targetRunDirectory = targetRunDirectory + "/" + self.runName
        self.outputDirectory = outputDirectory
        self.interactive = interactive
        self.CLUECLUST = CLUECLUST

    #For building a round and adding it to the end of the rounds to be ran
    def buildRound(self, roundName, featuresFile, selectionFile, clueConfig):
        self.rounds.append(ClueRound(roundName, 
                                     self.targetRunDirectory, 
                                     featuresFile, 
                                     selectionFile,
                                     clueConfig
                                     ))
    
    def setRound(self, roundName, featureSelectionFile, clusterSelectionFile, clueConfig):
        #Change all round settings in the given round
        for round in self.rounds:
            if (round.roundName == roundName):
                round.setRound(roundName, 
                               self.targetRunDirectory, 
                               featureSelectionFile, 
                               clusterSelectionFile,
                               clueConfig)
                return
            
    #Update the target run directory to a new directory after directory change
    def updateTargetRunDirectory(self, targetRunDirectory):
        self.targetRunDirectory = targetRunDirectory + "/" + self.runName
        for round in self.rounds:
            round.updateRoundDirectory(self.targetRunDirectory)

    #Update the target run directory to a new directory after run name change
    def updateRunName(self, runName):
        self.runName = runName
        self.targetRunDirectory = str(self.targetRunDirectory) + "/" + runName
        for round in self.rounds:
            round.updateRoundDirectory(self.targetRunDirectory)

    #Moves a round up in the listst of rounds to be ran, if it is not already at the top.
    def moveRoundUp(self, roundName):
        for i in range(1, len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                #Swap with the previous round
                if (i > 0):
                    self.rounds[i], self.rounds[i - 1] = self.rounds[i - 1], self.rounds[i]
                    return
                else:
                    raise clueutil.ClueException("Round with name " + roundName + " is already at the top, cannot move up.")
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot move up.")

    #Moves a round down in the list of rounds to be ran, if it is not already at the bottom.
    def moveRoundDown(self, roundName):
        for i in range(len(self.rounds) - 1):
            if (self.rounds[i].roundName == roundName):
                #Swap with the next round
                if (i < len(self.rounds) - 1):
                    self.rounds[i], self.rounds[i + 1] = self.rounds[i + 1], self.rounds[i]
                    return
                else:
                    raise clueutil.ClueException("Round with name " + roundName + " is already at the bottom, cannot move down.")
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot move down.")

    #Get the index of a round by its name
    def getRoundIndex(self, roundName):
        for i in range(len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                return i
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot get index.")

    #Get a round by its name
    def getRound(self, roundName):
        for round in self.rounds:
            if (round.roundName == roundName):
                return round
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot get round.")
    
    #Remove a round by its name
    def removeRound(self, roundName):
        #Remove a round from the run
        for i in range(len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                del self.rounds[i]
                return
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot remove.")

    #Main run function
    def run(self):
        if (len(self.rounds) == 0):
            raise clueutil.ClueException("No rounds to run, exiting...")
        
        Path(self.targetRunDirectory).mkdir(parents=True, exist_ok=True)
        inputDF = pd.read_csv(self.baseFile)

        print("Building base features file...")
        self.baseFeaturesFile = self.targetRunDirectory + "/baseFeatures.csv"
        baseFeaturesDF = clueutil.FeatureExtractor.extractFromDataFrame(inputDF)
        baseFeaturesDF.to_csv(self.baseFeaturesFile, index=False)
        print("baseFeatures file written to " + self.baseFeaturesFile)
       
        #First round is ran without any cluster, metadata, or cluster selection files.
        currentDF = inputDF
        if (self.rounds[0].clueConfig.useFeatures): 
            currentDF = baseFeaturesDF
        newInputs = clueutil.InputFilter.filter(currentDF, 
                                                None, #No previous metadata FD
                                                None, #No previous clusters FD
                                                self.rounds[0].
                                                featureSelectionFile, 
                                                None  #No cluster selection file for first round
                                       ) 
        Path(self.rounds[0].roundDirectory).mkdir(exist_ok=True) #Build round directory
        newInputs.to_csv(path_or_buf=self.rounds[0].inputFile, header=False, index=False)
        prevRound = self.rounds[0]
        prevRound.runRound(self.CLUECLUST)

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
            currRound.runRound(self.CLUECLUST)
        
        targetOutputDirectory = str(self.targetRunDirectory) + "/" + str(self.outputDirectory)
        print("Writing output files to directory: " + targetOutputDirectory + "\n")
        Path(targetOutputDirectory).mkdir(exist_ok=True)
        finalRound = self.rounds[len(self.rounds) - 1]
        print("Final round name: " + finalRound.roundName)
        print("Copying final round clusters file...")
        shutil.copy(finalRound.roundDirectory + finalRound.clustersFile, targetOutputDirectory + "/clusters_output.csv")
        print("Copying final round metadata file...")
        shutil.copy(finalRound.roundDirectory + finalRound.metadataFile, targetOutputDirectory + "/metadata_output.csv")
        #print("Generating graphs from final round data...")
        #clueutil.ClueGraphing.generateGraphs(finalRound, 
        #                                     self.baseFile, 
        #                                     self.baseFeaturesFile, 
        #                                     outputDirectory=targetOutputDirectory, 
        #                                     directOutput=self.interactive) 


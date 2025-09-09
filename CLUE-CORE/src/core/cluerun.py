from enum import IntEnum

import os
import subprocess
from pathlib import Path
import shutil
import time

import pandas as pd

from core import clueutil
from core.clueutil import ClueCancellation, ClueLogger as Logger

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
        Logger.logToFileOnly("refreshCallOpts called")
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
        Logger.logToFileOnly("getConfigCallOpts called")
        self.refreshCallOpts()
        callOpts = []
        callOpts += self.callDictALG[self.algorithm]
        callOpts += self.callDictDM[self.distanceMetric]
        if (self.standardize):
            callOpts.append("-standardize")
        return callOpts

    #Generate config-dependent components of call for parameter optimization usage
    def getOptimizeCallOpts(self):
        Logger.logToFileOnly("getOptimizeCallOpts called")
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
    featureSelectionFile = ""   #File descriptor for the feature selection file
    clusterSelectionFile = ""   #File descriptor for the cluster selection file

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
        Logger.logToFileOnly("setRound called")
        self.roundName = roundName
        self.directory = directory
        self.roundDirectory = str(directory) + "/" + roundName + "/"
        self.featureSelectionFile = featureSelectionFile or ""
        self.clusterSelectionFile = clusterSelectionFile or ""
        self.clueConfig = clueConfig

        #TODO Fix inconsistency in managing file descriptors
        self.clustersFile = "clusters.csv"
        self.metadataFile = "metadata.csv"
        self.inputFile = self.roundDirectory + "input.csv"

    def clearRoundDirectory(self):
        """
            removes the metadata, clusters, and input files from the round directory. Only removes specifically named files,
            does not remove the round directory itself. This is for safety to avoid deleting important files.
        """
        #TODO Add removal of files generated from the parameter optimizer
        Logger.logToFileOnly("clearRoundDirectory called")
        if (os.path.exists(self.roundDirectory)):
            for file in [self.clustersFile, self.metadataFile, "input.csv"]:
                filePath = self.roundDirectory + file
                if (os.path.exists(filePath)):
                    os.remove(filePath)
                    Logger.log("Removed file: " + filePath)
        else:
            Logger.log("Round directory does not exist, nothing to clear: " + self.roundDirectory)
    
    #Update the round directory to a new directory
    def updateDirectory(self, directory):
        Logger.logToFileOnly("updateDirectory called")
        self.directory = directory
        self.roundDirectory = str(directory) + "/" + self.roundName + "/"
        self.inputFile = self.roundDirectory + "input.csv"

    #Builds the subprocess call based off the configuration of the round
    def buildCall(self, CLUECLUST):
        Logger.logToFileOnly("buildCall called")
        #Base call, always the same regardless of config
        call = ["java", "-jar", CLUECLUST, "-f", self.inputFile, 
                "-d", self.roundDirectory, "-output", self.clustersFile,
                "-metadata", self.metadataFile, "-t", str(self.clueConfig.threads)]

        #Generate config-dependent components of call
        call += self.clueConfig.getConfigCallOpts()
        return call

    
    #Special case for building a call using the parameter optimizer
    def runParamOptimizer(self, CLUECLUST):
        Logger.logToFileOnly("runParamOptimizer called")
        #Base call, always the same regardless of config
        call = ["java", "-jar", CLUECLUST, "-f", self.inputFile,
                "-d", self.roundDirectory, "-c", "optimize", "-o", str(int(self.clueConfig.paramOptimization))]

        #Generate config-dependent components of call
        call += self.clueConfig.getOptimizeCallOpts()

        proc = subprocess.Popen(call,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1)
        
        try:
            while proc.poll() is None:
                if ClueCancellation.is_cancellation_requested():
                    Logger.log("Cancelling parameter optimization...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise clueutil.ClueCancelledException("Run Cancelled: The CLUE round was cancelled during parameter optimization.")

                output = proc.stdout.readline()
                if output:
                    Logger.log(output.rstrip())
                else:
                    time.sleep(0.1)

            remainingOutput = proc.stdout.read()
            if remainingOutput:
                for line in remainingOutput.splitlines():
                    Logger.log(line.rstrip())

        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        Logger.log("Reading best parameter optimization values...")
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
        
        Logger.log("Best values found to be: \n")
        Logger.log("Epsilon = " + str(self.clueConfig.epsilon) + "\n")
        Logger.log("MinPts = " + str(self.clueConfig.minPts) + "\n")
        Logger.log("HashTables = " + str(self.clueConfig.hashtables) + "\n")
        Logger.log("Hyperplanes = " + str(self.clueConfig.hyperplanes) + "\n")

    #Runs the round with a specified CLUECLUST jar file
    def runRound(self, CLUECLUST):
        Logger.logToFileOnly("runRound called")
        Logger.log("Running round: " + self.roundName + "...")
        Logger.log("Using CLUECLUST jar file: " + CLUECLUST)
        Logger.log("Target round directory: " + self.roundDirectory)
        Logger.log("Input File: " + self.inputFile)
        Logger.log("Clusters File: " + self.roundDirectory + self.clustersFile)
        Logger.log("Metadata File: " + self.roundDirectory + self.metadataFile)
        Logger.log("Feature Selection File: " + str(self.featureSelectionFile))
        Logger.log("Cluster Selection File: " + str(self.clusterSelectionFile))

        if ClueCancellation.is_cancellation_requested():
            raise clueutil.ClueCancelledException("Run Cancelled: The CLUE round was cancelled before it could start.")

        #If param opt is selected we run the parameter optimizer first to set new config parameters
        if (int(self.clueConfig.paramOptimization) > 0):
            Logger.log("Running parameter optimizer with level: " + str(int(self.clueConfig.paramOptimization)) + "\n")
            self.runParamOptimizer(CLUECLUST)
        
        call = self.buildCall(CLUECLUST)

        proc = subprocess.Popen(call, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1)
        
        try:
            while proc.poll() is None:
                if ClueCancellation.is_cancellation_requested():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise clueutil.ClueCancelledException("Run Cancelled: The CLUE round was cancelled.")

                output = proc.stdout.readline()
                if output:
                    Logger.log(output.rstrip())
                else:
                    time.sleep(0.1)
            remainingOutput = proc.stdout.read()
            if remainingOutput:
                for line in remainingOutput.splitlines():
                    Logger.log(line.rstrip)

        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        Logger.log("Round " + self.roundName + " completed successfully.\n")

#A full run of Clue
class ClueRun:
    rounds = []                     #Rounds to run
    runName = None                  #Name of run
    baseFile = None                 #Base input file, raw data
    baseFeaturesFile = None         #Base file containing features, does not need to exist prior to running, will be created in run directory
    targetRunDirectory = None       #Run directory
    baseDirectory = None            #Base directory
    outputDirectory = None          #Output directory for the final results, will be created if it does not exist
    interactive = None              #If false, will display graphs directly instead of saving them to a file
    CLUECLUST = None                #Path to the CLUECLUST jar file, must be set before running
    __roundPointer = 0               #Pointer to the current round being ran 

    _inputDFCache = None        #Cache for the input dataframe to avoid re-reading from disk multiple times
    _baseFeaturesDFCache = None  #Cache for the base features dataframe to avoid re-reading from disk multiple times
    _cacheValid = False          #Flag to indicate if the cache is valid or needs to be refreshed


    def __init__(self, runName, baseFile, baseDirectory, outputDirectory="output", interactive=False, CLUECLUST="CLUECLUST.jar"):
        self.runName = runName
        self.baseFile = baseFile
        self.baseDirectory = baseDirectory
        self.outputDirectory = outputDirectory
        self.interactive = interactive
        self.CLUECLUST = CLUECLUST

        self.targetRunDirectory = self.baseDirectory + "/" + self.runName

    #For building a round and adding it to the end of the rounds to be ran
    def buildRound(self, roundName, featuresFile, selectionFile, clueConfig):
        Logger.logToFileOnly("buildRound called")
        self.rounds.append(ClueRound(roundName, 
                                     self.targetRunDirectory, 
                                     featuresFile, 
                                     selectionFile,
                                     clueConfig
                                     ))
    
    def setRound(self, roundName, featureSelectionFile, clusterSelectionFile, clueConfig):
        Logger.logToFileOnly("ClueRun.setRound called")
        #Change all round settings in the given round
        for round in self.rounds:
            if (round.roundName == roundName):
                round.setRound(roundName, 
                               self.targetRunDirectory,
                               featureSelectionFile, 
                               clusterSelectionFile,
                               clueConfig)
                return
            
    def getRoundPointer(self):
        Logger.logToFileOnly("getRoundPointer called")
        return self.__roundPointer

    def updateBaseFile(self, baseFile):
        Logger.logToFileOnly("updateBaseFile called")
        if baseFile == self.baseFile:
            return
        self.resetRun()
        self.baseFile = baseFile

    #Update the base directory to a new directory after directory change
    def updateBaseDirectory(self, baseDirectory):
        Logger.logToFileOnly("updateBaseDirectory called")
        if baseDirectory == self.baseDirectory:
            return
        self.resetRun()
        self.baseDirectory = baseDirectory
        self.targetRunDirectory = self.baseDirectory + "/" + self.runName
        for round in self.rounds:
            round.updateDirectory(self.targetRunDirectory)

    #Update the target run directory to a new directory after run name change
    def updateRunName(self, runName):
        Logger.logToFileOnly("updateRunName called")
        self.resetRun()
        self.runName = runName
        self.targetRunDirectory = self.baseDirectory + "/" + self.runName
        for round in self.rounds:
            round.updateDirectory(self.targetRunDirectory)

    #Moves a round up in the listst of rounds to be ran, if it is not already at the top.
    def moveRoundUp(self, roundName):
        Logger.logToFileOnly("moveRoundUp called")
        for i in range(1, len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                #Swap with the previous round
                if (i > 0):
                    self.rounds[i], self.rounds[i - 1] = self.rounds[i - 1], self.rounds[i]
                    self.resetRun()
                    return
                else:
                    raise clueutil.ClueException("Round with name " + roundName + " is already at the top, cannot move up.")
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot move up.")

    #Moves a round down in the list of rounds to be ran, if it is not already at the bottom.
    def moveRoundDown(self, roundName):
        Logger.logToFileOnly("moveRoundDown called")
        for i in range(len(self.rounds) - 1):
            if (self.rounds[i].roundName == roundName):
                #Swap with the next round
                if (i < len(self.rounds) - 1):
                    self.rounds[i], self.rounds[i + 1] = self.rounds[i + 1], self.rounds[i]
                    self.resetRun()
                    return
                else:
                    raise clueutil.ClueException("Round with name " + roundName + " is already at the bottom, cannot move down.")
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot move down.")

    #Get the index of a round by its name
    def getRoundIndex(self, roundName):
        Logger.logToFileOnly("getRoundIndex called")
        for i in range(len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                return i
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot get index.")

    #Get a round by its name
    def getRound(self, roundName):
        Logger.logToFileOnly("getRound called")
        for round in self.rounds:
            if (round.roundName == roundName):
                return round
        return None
        
    #Remove a round by its name
    def removeRound(self, roundName):
        Logger.logToFileOnly("removeRound called")
        #Remove a round from the run
        for i in range(len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                del self.rounds[i]
                self.resetRun()
                return
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot remove.")
    
    def clearRunDirectory(self):
        """
            removes all round files using the clearRoundDirectory method of each round
        """
        Logger.logToFileOnly("clearDirectory called")
        for round in self.rounds:
            round.clearRoundDirectory()
    
    def _loadSharedData(self):
        Logger.logToFileOnly("ClueRun._loadSharedData called")
        if (not self._cacheValid):
            Logger.log("Loading input data from file: " + self.baseFile)
            self._inputDFCache = pd.read_csv(self.baseFile)
            Logger.log("Building base features file...")
            self.baseFeaturesFile = self.targetRunDirectory + "/baseFeatures.csv"
            self._baseFeaturesDFCache = clueutil.FeatureExtractor.extractFromDataFrame(self._inputDFCache)
            self._baseFeaturesDFCache.to_csv(self.baseFeaturesFile, index=False)
            Logger.log("baseFeatures file written to " + self.baseFeaturesFile)

            self._cacheValid = True
    
    def _invalidateCache(self):
        Logger.logToFileOnly("ClueRun._invalidateCache called")
        self._cacheValid = False
        self._inputDFCache = None
        self._baseFeaturesDFCache = None

    def _getInputDataframe(self):
        Logger.logToFileOnly("ClueRun._getInputDataframe called")
        self._loadSharedData()
        return self._inputDFCache
    
    def getFeaturesDataframe(self):
        Logger.logToFileOnly("ClueRun.getFeaturesDataframe called")
        self._loadSharedData()
        return self._baseFeaturesDFCache

    def reset__roundPointer(self):
        Logger.logToFileOnly("reset__roundPointer called")
        self.__roundPointer = 0

    def resetRun(self):
        Logger.logToFileOnly("resetRun called")
        Logger.log("Resetting run to initial state...")
        self.reset__roundPointer()
        self._invalidateCache()
        #self.clearRunDirectory()

    def setupRuns(self):
        Logger.logToFileOnly("setupRuns called")
        Logger.log("Setting up run: " + self.runName + "...")
        self.reset__roundPointer()
        Path(self.targetRunDirectory).mkdir(parents=True, exist_ok=True)
        self._getInputDataframe()
        self.getFeaturesDataframe()
    
    def onStoppedRun(self):
        Logger.logToFileOnly("onStoppedRun called")
        #Unused for now, placeholder for future functionality

    def runNextRound(self):
        Logger.logToFileOnly("runNextRound called")
        if ClueCancellation.is_cancellation_requested():
            raise clueutil.ClueCancelledException("Run Cancelled: The CLUE round was cancelled before it could start.")
        if (not os.path.exists(self.CLUECLUST)):
            raise clueutil.ClueException("CLUECLUST jar file not found at: " + self.CLUECLUST)
        if (len(self.rounds) == 0):
            raise clueutil.ClueException("No rounds to run, exiting...")
        Logger.log("Running round " + str(self.__roundPointer + 1) + " of " + str(len(self.rounds)) + "...")
        try:
            if (self.__roundPointer == 0):
                self.setupRuns()
                currentDF = self._getInputDataframe()
                if (self.rounds[0].clueConfig.useFeatures):
                    currentDF = self.getFeaturesDataframe()
                Logger.log("Filtering input data for round: " + self.rounds[0].roundName + "...")
                newInputs = clueutil.InputFilter.filter(currentDF, 
                                                        None, #No previous metadata FD
                                                        None, #No previous clusters FD
                                                        self.rounds[0].featureSelectionFile, 
                                                        None  #No cluster selection file for first round
                                            )
                Path(self.rounds[0].roundDirectory).mkdir(exist_ok=True) #Build round directory
                newInputs.to_csv(path_or_buf=self.rounds[0].inputFile, header=False, index=False)
                prevRound = self.rounds[0]
                prevRound.runRound(self.CLUECLUST)
                self.__roundPointer += 1
            elif (self.__roundPointer < len(self.rounds)):
                currRound = self.rounds[self.__roundPointer]
                currentDF = self._getInputDataframe()
                if (currRound.clueConfig.useFeatures):
                    currentDF = self.getFeaturesDataframe()
                prevRound = self.rounds[self.__roundPointer - 1]
                prevMetadataFD = prevRound.roundDirectory + prevRound.metadataFile
                prevClustersFD = prevRound.roundDirectory + prevRound.clustersFile
                Logger.log("Filtering input data for round: " + currRound.roundName + "...")
                newInputs = clueutil.InputFilter.filter(currentDF, 
                                                        prevClustersFD, 
                                                        prevMetadataFD, 
                                                        currRound.featureSelectionFile, 
                                                        currRound.clusterSelectionFile)
                Path(currRound.roundDirectory).mkdir(exist_ok=True)
                newInputs.to_csv(path_or_buf=currRound.inputFile, header=False, index=False)
                currRound.runRound(self.CLUECLUST)
                self.__roundPointer += 1
            else:
                raise clueutil.ClueException("All rounds have already been run, cannot run next round.")
        except clueutil.ClueCancelledException as e:
            self.onStoppedRun()
            raise e
        oldRoundPointer = self.__roundPointer
        if (self.__roundPointer == len(self.rounds)):
            outputWriteDirectory = str(self.targetRunDirectory) + "/" + str(self.outputDirectory)
            Logger.log("Writing output files to directory: " + outputWriteDirectory + "\n")
            Path(outputWriteDirectory).mkdir(exist_ok=True)
            finalRound = self.rounds[len(self.rounds) - 1]
            Logger.log("Final round name: " + finalRound.roundName)
            Logger.log("Copying final round clusters file...")
            shutil.copy(finalRound.roundDirectory + finalRound.clustersFile, outputWriteDirectory + "/clusters_output.csv")
            Logger.log("Copying final round metadata file...")
            shutil.copy(finalRound.roundDirectory + finalRound.metadataFile, outputWriteDirectory + "/metadata_output.csv")
            self.resetRun()
        return oldRoundPointer

    #Main run function
    def runFromBeginning(self):
        Logger.logToFileOnly("run called")
        Logger.log("Beginning full run of Clue...")
        Logger.log("Total rounds to run: " + str(len(self.rounds)))
        self.resetRun()
        currentRound = 0
        while (currentRound < len(self.rounds)):
            currentRound = self.runNextRound()



    def runRemainder(self):
        Logger.logToFileOnly("runRemainder called")
        Logger.log("Resuming run of Clue from round " + str(self.__roundPointer + 1) + "...\n")
        currentRound = self.__roundPointer
        while (currentRound < len(self.rounds)):
            currentRound = self.runNextRound()



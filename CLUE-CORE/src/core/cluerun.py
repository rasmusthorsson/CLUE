from enum import IntEnum

import os
import queue
import subprocess
from pathlib import Path
import shutil
import threading
import time
from typing import Dict, List, Optional

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

class ClueConfig:
    """
    Configuration settings for CLUE clustering subprocess.
    
    Attributes:
        algorithm: Clustering algorithm to use (DBSCAN, IPLSHDBSCAN, KMEANS)
        distanceMetric: Distance metric for clustering (EUCLIDEAN, ANGULAR, DTW)
        paramOptimization: Level of parameter optimization (NONE, KDISTANCE, EXPLORATION)
        epsilon: DBSCAN epsilon parameter for neighborhood distance
        minPts: DBSCAN minimum points parameter for core points
        hyperplanes: Number of hyperplanes for LSH-based algorithms
        hashtables: Number of hash tables for LSH-based algorithms
        kClusters: Number of clusters for K-means algorithm
        standardize: Whether to standardize data before clustering
        useFeatures: Whether to use extracted features instead of raw data
        threads: Number of threads for parallel processing
        window: Window size for DTW distance metric
        callDictALG: Dictionary mapping algorithms to command line arguments
        callDictDM: Dictionary mapping distance metrics to command line arguments
    """
    
    def __init__(self, 
                 algorithm: Algorithm = Algorithm.DBSCAN, 
                 distanceMetric: DistanceMetric = DistanceMetric.EUCLIDEAN,
                 paramOptimization: ParameterOptimizationLevel = ParameterOptimizationLevel.NONE,
                 epsilon: float = 1,
                 minPts: int = 20,
                 hyperplanes: int = 5,
                 hashtables: int = 10,
                 kClusters: int = 8,
                 standardize: bool = False,
                 useFeatures: bool = False,
                 threads: int = 4,
                 window: int = 20
                 ):
        self.algorithm: Algorithm = algorithm
        self.distanceMetric: DistanceMetric = distanceMetric
        self.paramOptimization: ParameterOptimizationLevel = paramOptimization
        self.epsilon: float = epsilon
        self.minPts: int = minPts
        self.hyperplanes: int = hyperplanes
        self.hashtables: int = hashtables
        self.kClusters: int = kClusters
        self.standardize: bool = standardize
        self.useFeatures: bool = useFeatures
        self.threads: int = threads
        self.window: int = window
        
        self.callDictALG: Dict[int, List[str]] = {}
        self.callDictDM: Dict[int, List[str]] = {}
        
        self._refreshCallOpts()

    def _refreshCallOpts(self):
        """
            Refreshes the dictionaries containing the call information, is called prior to returning the full call to ensure correct variable calls.
        """
        Logger.logToFileOnly("_refreshCallOpts called")

        # Algorithm dict
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

        # Distance metric dict
        self.callDictDM[int(DistanceMetric.EUCLIDEAN)] = []
        self.callDictDM[int(DistanceMetric.ANGULAR)] = ["-a"]
        self.callDictDM[int(DistanceMetric.DTW)] = [
                                                    "-dtw", 
                                                    "-window", str(int(self.window))
                                                    ]

    def getClusteringCallOpts(self):
        """
            Returns the command line options based on the current configuration settings.
        """
        Logger.logToFileOnly("getClusteringCallOpts called")
        self._refreshCallOpts()
        callOpts = []
        callOpts += self.callDictALG[self.algorithm]
        callOpts += self.callDictDM[self.distanceMetric]
        if (self.standardize):
            callOpts.append("-standardize")
        return callOpts

    def getOptimizeCallOpts(self):
        """
            Returns the command line options specifically for parameter optimization based on the current configuration settings.
        """
        Logger.logToFileOnly("getOptimizeCallOpts called")
        self._refreshCallOpts()
        callOpts = ["-c", "optimize"]
        callOpts += self.callDictDM[self.distanceMetric]
        callOpts.append("-o")
        callOpts.append(str(int(self.paramOptimization)))
        if (self.standardize):
            callOpts.append("-standardize")
        return callOpts

class ClueRound:
    """
    An individual round of CLUE clustering analysis.
    
    Attributes:
        roundName: Name identifier for this round
        directory: Base directory of the global run
        clueConfig: Configuration settings for the clustering subprocess
        featureSelectionFile: Path to feature selection file for this round
        clusterSelectionFile: Path to cluster selection file for this round

    Properties:
        inputFile: Path to input data file for this clustering round
        clustersFile: Filename for clustering results output
        metadataFile: Filename for metadata results output
        roundDirectory: Full path to this round's directory
    """

    def __init__(self, 
                 roundName: str, 
                 directory: str, 
                 featureSelectionFile: str, 
                 clusterSelectionFile: str,
                 clueConfig: ClueConfig,
                 ):
        # Initialize all instance attributes with type hints
        self.roundName: Optional[str] = None
        self.directory: Optional[str] = None
        self.clueConfig: Optional[ClueConfig] = None
        self.featureSelectionFile: str = ""
        self.clusterSelectionFile: str = ""

        self._roundDirectory: Optional[str] = None
        self._inputFile: Optional[str] = None
        self._clustersFile: Optional[str] = None
        self._metadataFile: Optional[str] = None

        # Set initial round settings
        self.setRound(roundName, 
                      directory, 
                      featureSelectionFile, 
                      clusterSelectionFile,
                      clueConfig)
        
    @property
    def inputFile(self) -> str:
        return self._inputFile
    
    @property
    def clustersFile(self) -> str:
        return self._clustersFile
    
    @property
    def metadataFile(self) -> str:
        return self._metadataFile
    
    @property
    def roundDirectory(self) -> str:
        return self._roundDirectory
        
    #Change all round settings in the given round
    def setRound(self, 
                 roundName, 
                 directory, 
                 featureSelectionFile, 
                 clusterSelectionFile, 
                 clueConfig):
        """
            Sets or updates the round settings.
        """
        Logger.logToFileOnly("setRound called")
        self.roundName = roundName
        self.directory = directory
        self._roundDirectory = str(directory) + "/" + roundName + "/"
        self.featureSelectionFile = featureSelectionFile or ""
        self.clusterSelectionFile = clusterSelectionFile or ""
        self.clueConfig = clueConfig

        #TODO Fix inconsistency in managing file descriptors
        # Currently the file descriptors are not fully qualified and are instead joined with the round directory when they
        # are used by the user. This is not ideal and should be fixed but has dependencies in other parts of the code.
        self._clustersFile = "clusters.csv"
        self._metadataFile = "metadata.csv"
        self._inputFile = self.roundDirectory + "input.csv"

    def clearRoundDirectory(self):
        """
            removes the metadata, clusters, and input files from the round directory. Only removes specifically named files,
            does not remove the round directory itself. This is for safety to avoid deleting important files.
        """
        #TODO Add removal of files generated from the parameter optimizer
        Logger.logToFileOnly("clearRoundDirectory called")
        if (os.path.exists(self._roundDirectory)):
            for file in [self._clustersFile, self._metadataFile, "input.csv"]:
                filePath = self._roundDirectory + file
                if (os.path.exists(filePath)):
                    os.remove(filePath)
                    Logger.log("Removed file: " + filePath)
        else:
            Logger.log("Round directory does not exist, nothing to clear: " + self._roundDirectory)
    
    def updateDirectory(self, directory):
        """
            Updates the required round variables when a new directory is set.
        """
        Logger.logToFileOnly("updateDirectory called")
        self.directory = directory
        self._roundDirectory = str(directory) + "/" + self.roundName + "/"
        self._inputFile = self._roundDirectory + "input.csv"

    def _buildCall(self, CLUECLUST):
        """
            Builds the subprocess call based off the configuration of the round.
        """
        Logger.logToFileOnly("_buildCall called")

        #Base call, always the same regardless of config
        call = ["java", "-jar", CLUECLUST, "-f", self._inputFile,
                "-d", self._roundDirectory, "-output", self._clustersFile,
                "-metadata", self._metadataFile, "-t", str(self.clueConfig.threads)]

        #Generate config-dependent components of call
        call += self.clueConfig.getClusteringCallOpts()
        return call

    def _runParamOptimizer(self, CLUECLUST):
        """
            Runs the parameter optimizer and updates the clueConfig with the best parameters found. Uses the CLUECLUST jar file.
        """
        Logger.logToFileOnly("_runParamOptimizer called")

        #Base call, always the same regardless of config
        call = ["java", "-jar", CLUECLUST, "-f", self.inputFile,
                "-d", self.roundDirectory, "-c", "optimize", "-o", str(int(self.clueConfig.paramOptimization))]

        #Generate config-dependent components of call
        call += self.clueConfig.getOptimizeCallOpts()

        try:
            proc = subprocess.Popen(call, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1)
        except FileNotFoundError as e:
            raise clueutil.ClueException("Could not find Java runtime to execute CLUECLUST jar file. Please ensure Java is installed and available in your system PATH.") from e

        outputQueue = queue.Queue() # Instead of direct queueing to for threadsafe logging, only main thread is used for logging now

        def readOutput():
            """Read subprocess output in separate thread."""
            try:
                for line in iter(proc.stdout.readline, ''):
                    if line:
                        outputQueue.put(line.rstrip())
                proc.stdout.close()
            except Exception:
                pass
            finally:
                outputQueue.put(None)  # End of output

        readerThread = threading.Thread(target=readOutput, daemon=True)
        readerThread.start()

        # The subprocess is non-blocking, so we poll it for output and check for cancellation.
        # This also ensures that the output is logged in real-time.
        try:
            while proc.poll() is None:
                if ClueCancellation.isCancellationRequested():
                    Logger.log("Cancelling parameter optimization...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise clueutil.ClueCancelledException("Round cancelled. Round was cancelled during parameter optimization.")
                try:
                    output = outputQueue.get(timeout=0.1)
                    if output is None:
                        break
                    Logger.log(output)
                except queue.Empty:
                    continue
            try:
                while True:
                    output = outputQueue.get_nowait()
                    if output is None:
                        break
                    Logger.log(output.rstrip())
            except queue.Empty:
                pass

        # Ensure the process is terminated if it is still running
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            readerThread.join(timeout=2)

        #Read the optimal parameters from the output file and update the clueConfig
        Logger.log("Reading best parameter optimization values...")
        paramOptFD = self._roundDirectory + "optimal_params.csv"
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

    def runRound(self, CLUECLUST):
        """
            Runs the clustering round as configured, including optional parameter optimization. Uses the CLUECLUST jar file.
        """
        Logger.logToFileOnly("runRound called")
        Logger.log("Running round: " + self.roundName + "...")
        Logger.log("Using CLUECLUST jar file: " + CLUECLUST)
        Logger.log("Target round directory: " + self._roundDirectory)
        Logger.log("Input File: " + self._inputFile)
        Logger.log("Clusters File: " + self._roundDirectory + self._clustersFile)
        Logger.log("Metadata File: " + self._roundDirectory + self._metadataFile)
        Logger.log("Feature Selection File: " + str(self.featureSelectionFile))
        Logger.log("Cluster Selection File: " + str(self.clusterSelectionFile))

        if ClueCancellation.isCancellationRequested():
            raise clueutil.ClueCancelledException("Run Cancelled: The CLUE round was cancelled before it could start.")

        #If param opt is selected we run the parameter optimizer first to set new config parameters
        if (int(self.clueConfig.paramOptimization) > 0):
            Logger.log("Running parameter optimizer with level: " + str(int(self.clueConfig.paramOptimization)) + "\n")
            self._runParamOptimizer(CLUECLUST)
        
        call = self._buildCall(CLUECLUST)
        try:
            proc = subprocess.Popen(call, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1)
        except FileNotFoundError as e:
            raise clueutil.ClueException("Could not find Java runtime to execute CLUECLUST jar file. Please ensure Java is installed and available in your system PATH.") from e

        outputQueue = queue.Queue() # Instead of direct queueing to for threadsafe logging, only main thread is used for logging now

        def readOutput():
            """Read subprocess output in separate thread."""
            try:
                for line in iter(proc.stdout.readline, ''):
                    if line:
                        outputQueue.put(line.rstrip())
                proc.stdout.close()
            except Exception:
                pass
            finally:
                outputQueue.put(None)  # End of output

        readerThread = threading.Thread(target=readOutput, daemon=True)
        readerThread.start()

        # The subprocess is non-blocking, so we poll it for output and check for cancellation.
        # This also ensures that the output is logged in real-time.
        try:
            while proc.poll() is None:
                if ClueCancellation.isCancellationRequested():
                    Logger.log("Cancelling clustering run...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise clueutil.ClueCancelledException("Run Cancelled: The CLUE round was cancelled.")
                try:
                    output = outputQueue.get(timeout=0.1)
                    if output is None:
                        break
                    Logger.log(output)
                except queue.Empty:
                    continue
            try:
                while True:
                    output = outputQueue.get_nowait()
                    if output is None:
                        break
                    Logger.log(output.rstrip())
            except queue.Empty:
                pass

        # Ensure the process is terminated if it is still running
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            readerThread.join(timeout=2)

        Logger.log("Round " + self.roundName + " completed successfully.\n")

class ClueRun:
    """
    A full run of CLUE clustering analysis.
    
    Attributes:
        rounds: List of ClueRound objects to run in sequence
        runName: Name of this run
        baseFile: Path to the base input file containing raw data
        baseDirectory: Base directory for the run
        outputDirectory: Directory name for final results
        interactive: Whether to display graphs directly vs save to file
        CLUECLUST: Path to the CLUECLUST jar file

    Properties:
        targetRunDirectory: Directory where run outputs are stored
        baseFeaturesFile: Path to features file (created during run)
        roundPointer: Index of the current round being run
    """
    
    def __init__(self, runName: str, baseFile: str, baseDirectory: str, 
                 outputDirectory: str = "output", interactive: bool = False, 
                 CLUECLUST: str = "CLUECLUST.jar"):

        # Attributes
        self.rounds: List[ClueRound] = []                    # Rounds to run
        self.runName: str = runName                          # Name of run
        self.baseFile: str = baseFile                        # Base input file, raw data
        self.baseDirectory: str = baseDirectory              # Base directory
        self.outputDirectory: str = outputDirectory          # Output directory for final results
        self.interactive: bool = interactive                 # Display graphs directly vs save to file
        self.CLUECLUST: str = CLUECLUST                      # Path to CLUECLUST jar file

        # Properties
        self._baseFeaturesFile: Optional[str] = None                        # Features file (created in run directory)
        self._targetRunDirectory = self.baseDirectory + "/" + self.runName  # Run directory
        self._roundPointer: int = 0                                         # Pointer to current round being run
        
        # Cache attributes
        self._inputDFCache: Optional[pd.DataFrame] = None           # Input dataframe cache
        self._baseFeaturesDFCache: Optional[pd.DataFrame] = None    # Features dataframe cache
        self._cacheValid: bool = False                              # Cache validity flag

    @property
    def baseFeaturesFile(self) -> Optional[str]:
        return self._baseFeaturesFile
    
    @property
    def targetRunDirectory(self) -> str:
        return self._targetRunDirectory

    def buildRound(self, roundName, featuresFile, selectionFile, clueConfig):
        """
            Builds a new round and appends it to the list of rounds to be ran.
        """
        Logger.logToFileOnly("buildRound called")
        self.rounds.append(ClueRound(roundName, 
                                     self._targetRunDirectory, 
                                     featuresFile, 
                                     selectionFile,
                                     clueConfig
                                     ))
    
    def setRound(self, roundName, featureSelectionFile, clusterSelectionFile, clueConfig):
        """
            Sets or updates the round settings for an existing round by name.
        """
        Logger.logToFileOnly("ClueRun.setRound called")
        for round in self.rounds:
            if (round.roundName == roundName):
                round.setRound(roundName, 
                               self._targetRunDirectory,
                               featureSelectionFile, 
                               clusterSelectionFile,
                               clueConfig)
                return

    def moveRoundUp(self, roundName):
        """
            Moves a round up in the list of rounds to be ran, if it is not already at the top. Resets run if successful.
        """
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

    def moveRoundDown(self, roundName):
        """
            Moves a round down in the list of rounds to be ran, if it is not already at the bottom. Resets run if successful.
        """
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
    
    def removeRound(self, roundName):
        """
            Removes a round from the list of rounds to be ran by its name. Resets run if successful.
        """
        Logger.logToFileOnly("removeRound called")
        for i in range(len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                del self.rounds[i]
                self.resetRun()
                return
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot remove.")

    def updateBaseFile(self, baseFile):
        """
            Updates the baseFile and resets run if it has changed.
        """
        Logger.logToFileOnly("updateBaseFile called")
        if baseFile == self.baseFile:
            return
        self.resetRun()
        self.baseFile = baseFile

    def updateBaseDirectory(self, baseDirectory):
        """
            Updates the related variables when a new baseDirectory is set.
        """
        Logger.logToFileOnly("updateBaseDirectory called")
        if baseDirectory == self.baseDirectory:
            return
        self.resetRun()
        self.baseDirectory = baseDirectory
        self._targetRunDirectory = self.baseDirectory + "/" + self.runName
        for round in self.rounds:
            round.updateDirectory(self._targetRunDirectory)

    def updateRunName(self, runName):
        """
            Updates the related variables when a new runName is set.
        """
        Logger.logToFileOnly("updateRunName called")
        self.resetRun()
        self.runName = runName
        self._targetRunDirectory = self.baseDirectory + "/" + self.runName
        for round in self.rounds:
            round.updateDirectory(self._targetRunDirectory)
    
    def getRoundPointer(self):
        Logger.logToFileOnly("getRoundPointer called")
        return self._roundPointer

    def getRoundIndex(self, roundName):
        """
            Returns the index of a round by its name.
        """
        Logger.logToFileOnly("getRoundIndex called")
        for i in range(len(self.rounds)):
            if (self.rounds[i].roundName == roundName):
                return i
        raise clueutil.ClueException("Round with name " + roundName + " not found, cannot get index.")
    
    def getRoundByIndex(self, index):
        """
            Returns a round by its index in the list of rounds to be ran.
        """
        Logger.logToFileOnly("getRoundByIndex called")
        if (index < 0 or index >= len(self.rounds)):
            raise clueutil.ClueException("Index " + str(index) + " is out of bounds, cannot get round.")
        return self.rounds[index]

    def getRound(self, roundName):
        """
            Returns a round by its name, or None if not found.
        """
        Logger.logToFileOnly("getRound called")
        for round in self.rounds:
            if (round.roundName == roundName):
                return round
        return None
    
    def clearRunDirectory(self):
        """
            removes all round files using the clearRoundDirectory method of each round
        """
        Logger.logToFileOnly("clearDirectory called")
        for round in self.rounds:
            round.clearRoundDirectory()
    
    def _loadSharedData(self):
        """
            Loads the share data variables if the cache is not valid. This includes generating the base features file.
        """
        Logger.logToFileOnly("ClueRun._loadSharedData called")
        if (not self._cacheValid):
            Logger.log("Loading input data from file: " + self.baseFile)
            self._inputDFCache = pd.read_csv(self.baseFile)

            Logger.log("Building base features file...")
            self._baseFeaturesFile = self._targetRunDirectory + "/baseFeatures.csv"
            self._baseFeaturesDFCache = clueutil.FeatureExtractor.extractFromDataFrame(self._inputDFCache)
            self._baseFeaturesDFCache.to_csv(self._baseFeaturesFile, index=False)
            Logger.log("baseFeatures file written to " + self._baseFeaturesFile)

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
    
    def _getFeaturesDataframe(self):
        Logger.logToFileOnly("ClueRun._getFeaturesDataframe called")
        self._loadSharedData()
        return self._baseFeaturesDFCache

    def _resetRoundPointer(self):
        Logger.logToFileOnly("_resetRoundPointer called")
        self._roundPointer = 0

    def _setupRun(self):
        """
            Sets up the run directory and loads the input data and features dataframes.
        """
        Logger.logToFileOnly("_setupRun called")
        Logger.log("Setting up run: " + self.runName + "...")
        self._resetRoundPointer()
        Path(self._targetRunDirectory).mkdir(parents=True, exist_ok=True)
        self._getInputDataframe()
        self._getFeaturesDataframe()

    def resetRun(self):
        """
            Resets the run to its initial state, starting at round 0 and invalidating the cache.
        """
        Logger.logToFileOnly("resetRun called")
        Logger.log("Resetting run to initial state...")
        self._resetRoundPointer()
        self._invalidateCache()
        #self.clearRunDirectory()

    def _onStoppedRun(self):
        Logger.logToFileOnly("onStoppedRun called")
        #Unused for now, placeholder for future functionality

    def _onClueNoiseException(self):
        """
            Reverts the run to the previous round if a ClueOnlyNoiseException is encountered.
        """
        Logger.logToFileOnly("_onClueNoiseException called")
        self._roundPointer -= 1

    def runNextRound(self):
        """
            Runs the next round in the list of rounds to be ran. If all rounds have been run, writes final output files and resets the run.
            Returns the index of the round that was just run, or the total number of rounds if the run completed.
        """
        Logger.logToFileOnly("runNextRound called")
        if ClueCancellation.isCancellationRequested():
            raise clueutil.ClueCancelledException("Run Cancelled: The CLUE round was cancelled before it could start.")
        if (not os.path.exists(self.CLUECLUST)):
            raise clueutil.ClueException("CLUECLUST jar file not found at: " + self.CLUECLUST)
        if (len(self.rounds) == 0):
            raise clueutil.ClueException("No rounds to run, exiting...")
        
        try:
            # The first round runs with no previous metadata or clusters and requires setup of the run
            if (self._roundPointer == 0):
                Logger.log("Running round " + str(self._roundPointer + 1) + " of " + str(len(self.rounds)) + "...")
                self._setupRun()

                currentDF = self._getInputDataframe()
                if (self.rounds[0].clueConfig.useFeatures):
                    currentDF = self._getFeaturesDataframe()

                Logger.log("Filtering input data for round: " + self.rounds[0].roundName + "...")
                newInputs = clueutil.InputFilter.filter(currentDF, 
                                                        None, #No previous metadata FD
                                                        None, #No previous clusters FD
                                                        self.rounds[0].featureSelectionFile, 
                                                        None  #No cluster selection file for first round
                                            )
                newInputs = newInputs.reset_index(drop=True) #Reset index to ensure it is sequential from 0
                Path(self.rounds[0].roundDirectory).mkdir(exist_ok=True) #Build round directory
                newInputs.to_csv(path_or_buf=self.rounds[0].inputFile, header=False, index=False)
                prevRound = self.rounds[0]
                prevRound.runRound(self.CLUECLUST)
                self._roundPointer += 1
            
            # Subsequent rounds use the previous round's metadata and clusters
            elif (self._roundPointer < len(self.rounds)):
                Logger.log("Running round " + str(self._roundPointer + 1) + " of " + str(len(self.rounds)) + "...")
                currRound = self.rounds[self._roundPointer]

                currentDF = self._getInputDataframe()
                if (currRound.clueConfig.useFeatures):
                    currentDF = self._getFeaturesDataframe()
                
                prevRound = self.rounds[self._roundPointer - 1]
                prevMetadataFD = prevRound.roundDirectory + prevRound.metadataFile
                prevClustersFD = prevRound.roundDirectory + prevRound.clustersFile

                Logger.log("Filtering input data for round: " + currRound.roundName + "...")
                newInputs = clueutil.InputFilter.filter(currentDF, 
                                                        prevClustersFD, 
                                                        prevMetadataFD, 
                                                        currRound.featureSelectionFile, 
                                                        currRound.clusterSelectionFile)
                newInputs = newInputs.reset_index(drop=True) #Reset index to ensure it is sequential from 0
                Path(currRound.roundDirectory).mkdir(exist_ok=True)
                newInputs.to_csv(path_or_buf=currRound.inputFile, header=False, index=False)
                currRound.runRound(self.CLUECLUST)
                self._roundPointer += 1
            else:
                raise clueutil.ClueException("All rounds have already been run, please reset rounds to run again.")
            metadataDF = pd.read_csv(self.rounds[self._roundPointer - 1].roundDirectory + "/" + self.rounds[self._roundPointer - 1]._metadataFile, index_col=False)
            if (len(metadataDF) == 0):
                self._onClueNoiseException()
                raise clueutil.ClueOnlyNoiseException("No data points were clustered in round " + str(self.getRoundByIndex(self._roundPointer).roundName) + ", staying at this round.")
            elif (len(metadataDF[metadataDF['ClusterId'] != -1]) == 0):
                self._onClueNoiseException()
                raise clueutil.ClueOnlyNoiseException("No data points were clustered (all noise) in round " + str(self.getRoundByIndex(self._roundPointer).roundName) + ", staying at this round.")
        except clueutil.ClueCancelledException as e:
            self._onStoppedRun()
            raise e
        
        oldRoundPointer = self._roundPointer
        if (self._roundPointer == len(self.rounds)):
            outputWriteDirectory = str(self._targetRunDirectory) + "/" + str(self.outputDirectory)
            Logger.log("Writing output files to directory: " + outputWriteDirectory + "\n")
            Path(outputWriteDirectory).mkdir(exist_ok=True)
            finalRound = self.rounds[len(self.rounds) - 1]
            Logger.log("Final round name: " + finalRound.roundName)
            Logger.log("Copying final round clusters file...")
            shutil.copy(finalRound.roundDirectory + finalRound.clustersFile, outputWriteDirectory + "/clusters_output.csv")
            Logger.log("Copying final round metadata file...")
            shutil.copy(finalRound.roundDirectory + finalRound.metadataFile, outputWriteDirectory + "/metadata_output.csv")
        return oldRoundPointer

    def runFromBeginning(self):
        """
            Starts the run from the beginning, resetting any previous state.
        """
        Logger.logToFileOnly("run called")
        Logger.log("Beginning full run of Clue...")
        Logger.log("Total rounds to run: " + str(len(self.rounds)))
        self.resetRun()
        currentRound = 0
        while (currentRound < len(self.rounds)):
            try:
                currentRound = self.runNextRound()
            except clueutil.ClueOnlyNoiseException as e:
                Logger.log("ClueOnlyNoiseException caught: " + str(e))
                raise clueutil.ClueException("Run stopped due to ClueOnlyNoiseException: " + str(e))

    def runRemainder(self):
        """
            Resumes the run from the current round pointer, without resetting any previous state.
        """
        Logger.logToFileOnly("runRemainder called")
        Logger.log("Resuming run of Clue from round " + str(self._roundPointer + 1) + "...\n")
        currentRound = self._roundPointer
        if (currentRound == len(self.rounds)):
            raise clueutil.ClueException("All rounds have already been run, please reset rounds to run again.")
        while (currentRound < len(self.rounds)):
            try:
                currentRound = self.runNextRound()
            except clueutil.ClueOnlyNoiseException as e:
                Logger.log("ClueOnlyNoiseException caught: " + str(e))
                raise clueutil.ClueException("Run stopped due to ClueOnlyNoiseException: " + str(e))

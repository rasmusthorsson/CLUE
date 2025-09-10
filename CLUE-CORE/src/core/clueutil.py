import queue
import sys
import random
import threading
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ClueCancellation:
    """
        Static class for handling cancellation requests in CLUE.
    """
    _cancellationEvent = threading.Event()
    
    @staticmethod
    def requestCancellation():
        ClueCancellation._cancellationEvent.set()

    @staticmethod
    def isCancellationRequested():
        return ClueCancellation._cancellationEvent.is_set()

    @staticmethod
    def clearCancellation():
        ClueCancellation._cancellationEvent.clear()

class ClueLogger:
    """
        Static logger class for CLUE, prints all the messages sent and also stores them in a 
        log file if a log file has been specified.
    """
    logFile = None
    logToFile = False
    messageQueue = None
    queueEnabled = False

    @staticmethod
    def enableQueue():
        ClueLogger.queueEnabled = True
        ClueLogger.messageQueue = queue.Queue()

    @staticmethod
    def disableQueue():
        ClueLogger.queueEnabled = False
        ClueLogger.messageQueue = None

    @staticmethod
    def clearQueue():
        if (ClueLogger.messageQueue):
            ClueLogger.messageQueue.queue.clear()

    @staticmethod
    def getQueuedMessages():
        messages = []
        if (ClueLogger.messageQueue):
            while not ClueLogger.messageQueue.empty():
                messages.append(ClueLogger.messageQueue.get())
        return messages

    @staticmethod
    def setLogToFile(logToFile):
        ClueLogger.logToFile = logToFile

    @staticmethod
    def setLogFile(logFilePath):
        if (logFilePath):
            try:
                if (ClueLogger.logFile):
                    ClueLogger.logFile.close()
                ClueLogger.logFile = open(logFilePath, "a")
            except Exception as e:
                print("Error opening log file: " + str(e))
        else:
            if (ClueLogger.logFile):
                ClueLogger.logFile.close()
            ClueLogger.logFile = None

    @staticmethod
    def closeLogFile():
        if (ClueLogger.logFile):
            ClueLogger.logFile.close()
            ClueLogger.logFile = None

    @staticmethod
    def log(*args, **kwargs):
        """
            Logs a message to the console and to the log file if specified.
        """
        if (ClueLogger.queueEnabled and ClueLogger.messageQueue):
            ClueLogger.messageQueue.put(' '.join(map(str, args)))

        print(*args, **kwargs)
        try:
            ClueLogger.logToFileOnly(*args, **kwargs)
        except Exception as e:
            print("Error writing to log file: " + str(e))

    @staticmethod
    def logToFileOnly(*args, **kwargs):
        """
            Logs a message only to the log file if specified.
        """
        if (ClueLogger.logFile and ClueLogger.logToFile):
            try:
                #Print the current time with the log message
                currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("[" + currentTime + "] ", end="", file=ClueLogger.logFile)
                print(*args, file=ClueLogger.logFile, **kwargs)
                ClueLogger.logFile.flush()
            except Exception as e:
                print("Error writing to log file: " + str(e))

#Base class for exceptions in CLUE
class ClueException(Exception):
    pass

class ClueCancelledException(ClueException):
    """
        Exception raised when a CLUE operation is cancelled by user request
    """

    def __init__(self, message="Operation cancelled by user"):
        self.message = message
        super().__init__(self.message)

class FeatureExtractor:
    """
        Static class for extracting features from raw input data.
    """

    def __init__(self):
        pass

#------------ Feature Methods ------------------------------
#For additions of features, define a method for calculating the feature from one
#line of raw data then add the method with a name in __featuresDict below.

    def _peakHour(line):
        """
            Hour with the highest value
        """
        peak = 0
        peakVal = 0
        for i in range(1, len(line)):
            if (line.iloc[i] > peakVal):
                peak = i
                peakVal = line.iloc[i]
        return peak
    
    def _peakHourVal(line):
        """
            Highest value found
        """
        peakVal = 0
        for i in range(1, len(line)):
            if (line.iloc[i] > peakVal):
                peakVal = line.iloc[i]
        return peakVal

    def _baseHour(line):
        """
            Hour with the lowest value
        """
        base = 0
        baseVal = float(sys.maxsize)
        for i in range(1, len(line)):
            if (line.iloc[i] < baseVal):
                base = i
                baseVal = line.iloc[i]
        return base

    def _baseHourVal(line):
        """
            Lowest value found
        """
        baseVal = float(sys.maxsize)
        for i in range(1, len(line)):
            if (line.iloc[i] < baseVal):
                baseVal = line.iloc[i]
        return baseVal

#Add name of feature and method name in __featuresDict.
    __featuresDict = {
            "Peak Hour" : _peakHour,
            "Peak Value" : _peakHourVal,
            "Base Hour" : _baseHour,
            "Base Value" : _baseHourVal
        } 

#-----------------------------------------------------------

    @staticmethod
    def _extract(line):
        """
            Extract features from an individual line, including "ID" feature
        """
        newLine = [line.iloc[0]] #TODO Fix bug where pandas converts to float for ID
        for v in FeatureExtractor.__featuresDict.values():
            newLine.append(v(line))
        return newLine

    @staticmethod
    def headers():
        """
            Returns a list of the features available, including "ID" feature
        """
        ClueLogger.logToFileOnly("__headers called")
        headers = ["ID"]
        for k in FeatureExtractor.__featuresDict.keys():
            headers.append(k) 
        return headers

    @staticmethod
    def extractFromDataFrame(dataframe):
        """
            Extract features from a dataframe of raw data
        """
        ClueLogger.logToFileOnly("extractFromDataFrame called")
        outMatrix = np.matrix(FeatureExtractor._extract(dataframe.iloc[0]))
        for i in range(1, len(dataframe)):
            outMatrix = np.vstack((outMatrix, FeatureExtractor._extract(dataframe.iloc[i])))
        headers = FeatureExtractor.headers()
        return pd.DataFrame(data=outMatrix, columns=headers)

class InputFilter:
    """
        Static class for filtering input data based on feature and cluster selections. Also filters based on clusters found and metadata from previous round.
    """

    @staticmethod
    def filterSelection(commands, metadataDF):
        """
            Filters the metadata dataframe based on the commands provided.
            Commands are in the form of a list of lists, where each inner list is a command split by ":".
            Supported commands:
                OVER:<size> - Keep clusters with size over <size>
                UNDER:<size> - Keep clusters with size under <size>
                IN:<id1,id2,...> - Keep only clusters with IDs in the list
                NOTIN:<id1,id2,...> - Remove clusters with IDs in the list
            Example: [["OVER", "10"], ["NOTIN", "1,2,3"]]
            Returns a filtered dataframe containing only the clusters that match the selection criteria.
        """
        ClueLogger.logToFileOnly("filterSelection called")
        filteredDF = metadataDF
        for command in commands:
            if (command[0] == "OVER"): #Over cluster size
                filteredDF = filteredDF.loc[filteredDF["Size"] > int(command[1])]
            if (command[0] == "UNDER"): #Under cluster size
                filteredDF = filteredDF.loc[filteredDF["Size"] < int(command[1])]
            if (command[0] == "IN"): #Direct cluster inclusion
                filteredDF = filteredDF.loc[filteredDF["ClusterId"].isin(map(int, command[1].split(",")))]
            if (command[0] == "NOTIN"): #Direct cluster exclusion
                filteredDF = filteredDF.drop(map(int, command[1].split(',')), errors='ignore')

        if (len(filteredDF) < 1):
            raise ClueException("No clusters remain after filtering out unwanted clusters")
        return filteredDF

    @staticmethod
    def filter(inputDF, clustersFD, metadataFD, featuresFD, selectionFD):
        """
            Filters the input dataframe based on the features file and the clusters and metadata files.
            If no features file is provided or if it is empty, all features are kept.
            If no clusters or metadata file is provided, all clusters are kept. 
            If only one of the files is provided, a warning is logged and no filtering is done.
            If a selection file is provided, it is used to filter the clusters further.
            Returns a filtered dataframe containing only the selected features and clusters.
        """
        ClueLogger.logToFileOnly("filter called")

        #If no feature file or if feature file is empty, accept all features
        noFeatures = False
        if featuresFD:
            try:
                featuresDF = pd.read_csv(featuresFD, header=0)
            except pd.errors.EmptyDataError:
                ClueLogger.log("Features file is empty, accepting all features...")
                noFeatures = True
        else:
            ClueLogger.log("No features file provided, accepting all features...")
            noFeatures = True
        
        selectedColumns = []
        selectedFeatures = None
        if (noFeatures):
            selectedFeatures = inputDF
        #Otherwise filter based on feature selection
        else:
            for f1_i in range(len(inputDF.columns.array)):
                for f2 in featuresDF:
                    if (inputDF.columns[f1_i] == f2):
                        selectedColumns.append(f2)
            selectedFeatures = inputDF[selectedColumns]

        #If no cluster selection file, or if previous cluster or metadata is non-existant, accept all clusters
        nextInput = selectedFeatures
        
        if (clustersFD and metadataFD):
            clustersDF = pd.read_csv(clustersFD, header=None)
            metadataDF = pd.read_csv(metadataFD)
            if (clustersDF.empty or metadataDF.empty):
                #TODO Fix so that noise can be used as an explicit cluster if desired, also if no clusters found, return special exception to allow
                #run to reset to previous round?
                ClueLogger.log("Either cluster or metadata file is empty, meaning no clusters were found in the previous round, stopping run...")
                raise ClueException("Clusters or metadata file is empty, cannot filter input")
            if (selectionFD):
                selectionFile = open(selectionFD)
                commands = []
                for line in selectionFile:
                    command = line.rstrip().split(":")
                    commands.append(command)
                selectionFile.close()
                filteredSelectionDF = InputFilter.filterSelection(commands, metadataDF)
                filteredClustersDF = clustersDF.loc[clustersDF.iloc[:, 1].isin(map(int, filteredSelectionDF["ClusterId"]))]
                nextInput = nextInput.loc[nextInput.iloc[:, 0].isin(filteredClustersDF.iloc[:, 0].array)]
            else:
                nextInput = nextInput.loc[nextInput.iloc[:, 0].isin(clustersDF.iloc[:, 0].array)]
        elif (clustersFD and not metadataFD): #MetadataFD does not exist
            ClueLogger.log("Warning: Metadata file does not exist but clusters file does, skipping filtering...")
        elif (metadataFD and not clustersFD): #ClustersDF does not exist
            ClueLogger.log("Warning: Clusters file does not exist but metadata file does, skipping filtering...")

        return nextInput

class ClueGraphing:
    """
        Static class for generating graphs from CLUE rounds.
    """

    graphs = {}

#---------------------------- Graphing Methods ----------------------------
#Additional graphs can be added here, each graph function takes the following parameters:
#   clueRound - The ClueRound object for the round to generate the graph for
#   baseInputFD - The file path to the base input data file
#   baseFeaturesFD - The file path to the base features data file
#   ax - The matplotlib axis to plot the graph on
#Each graph function should plot the graph on the provided axis and not return anything.
#The graph functions should also log to the ClueLogger as needed.

    def _rawBands(clueRound, baseInputFD, baseFeaturesFD, ax):
        """
            Plots the raw bands of the mean data for each cluster.
        """
        ClueLogger.logToFileOnly("__rawBands called")
        metadataDF = pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile)
        averagesColumns = [col for col in metadataDF.columns if col.startswith('OrigMean')]
        lines = []
        legendLabels = []
        for rowIndex in range(len(metadataDF)):
            legendLabels.append("Cluster " + str(metadataDF["ClusterId"].iloc[rowIndex]))
            line = ax.plot(metadataDF[averagesColumns].iloc[rowIndex])
            lines.append(line)

        xTicks = []
        xLabels = []
        for index in range(len(averagesColumns)):
            if (index % 2 == 0):
                xLabels.append("Timestep " + str(index + 1))
                xTicks.append(index)
        
        ax.set_xticks(ticks=xTicks)
        ax.set_xticklabels(labels=xLabels)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("Measurement Timestep")
        ax.set_ylabel("Consumption")

        ax.legend(legendLabels)

    def _featureProfiles(clueRound, baseInputFD, baseFeaturesFD, ax): #TODO Use feature selection to limit plotted features
        """
            Plots the feature profiles of the mean features for each cluster.
        """
        ClueLogger.logToFileOnly("__featureProfiles called")
        baseFeaturesDF = pd.read_csv(baseFeaturesFD)
        clustersDF = pd.read_csv(clueRound.roundDirectory + clueRound.clustersFile)
        metadataDF = pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile)
        filteredFeaturesDF = InputFilter.filter(baseFeaturesDF, 
                                                clueRound.roundDirectory + clueRound.clustersFile, 
                                                clueRound.roundDirectory + clueRound.metadataFile, 
                                                None, 
                                                None)
        uniqueClusters = metadataDF.iloc[:, 0].unique()
        clusterFeaturesDict = {}
        for cluster in uniqueClusters:
            clusterIDsDF = clustersDF.loc[clustersDF.iloc[:, 1] == cluster].iloc[:, 0]
            clusterFilteredFeaturesDF = filteredFeaturesDF.loc[filteredFeaturesDF["ID"].isin(clusterIDsDF.array)]
            clusterFeaturesDict[cluster] = clusterFilteredFeaturesDF

        clusterFeatureAverages = []
        for clusterFeatures in clusterFeaturesDict.values():
            featureAverages = []
            for column in clusterFeatures.columns[1:len(clusterFeatures.columns)]:
                featureAverages.append(clusterFeatures[column].mean())
            clusterFeatureAverages.append(featureAverages)

        random.seed(42) #Use of random to vary the colors in the heatmap slightly for increased visibility

        headers = FeatureExtractor.headers()
        ClueLogger.logToFileOnly("HEADERS DONE")
        xAxis = headers[1:len(headers)]
        yAxis = list(clusterFeaturesDict.keys())

        normalizedTransposedAverages = []
        #Normalize features in order to generate the heatmap itself, this allows each column to have a fair color scheme
        for average in np.transpose(clusterFeatureAverages):
            offset = random.uniform(0.1, 0.2)
            rng = np.ptp(average)
            if rng == 0:  # Avoid division by zero
                normalizedAverage = np.full_like(average, offset)
            else:
                # Normalize the average values to the range [0, 1] and add an offset
                normalizedAverage = (average - np.min(average)) / rng + offset
            normalizedTransposedAverages.append(normalizedAverage)

        normalizedAverages = np.transpose(normalizedTransposedAverages) #TODO Do not transpose
        
        heatmap = ax.imshow(normalizedAverages, cmap="Oranges")
        plt.colorbar(heatmap, ticks=[])
        ax.set_xticks(ticks=np.arange(len(xAxis)), labels=xAxis)
        ax.tick_params(axis='x', rotation=45)
        ax.set_yticks(ticks=np.arange(len(yAxis)), labels=yAxis)

        ax.set_xlabel("Feature")
        ax.set_ylabel("Cluster")
        
        for column in range(len(clusterFeatureAverages)):
            for row in range(len(clusterFeatureAverages[column])):
                ax.text(row, 
                         column, 
                         round(clusterFeatureAverages[column][row], 2), 
                         ha="center", 
                         va="center", 
                         color="black")


    #Add new plots here
    _graphDict = {
        "Mean Raw Data" : _rawBands,
        "Feature Profiles" : _featureProfiles
        }

#-------------------------------------------------------------------------------

    def __init__(self):
        pass

    @staticmethod
    def generateGraphs(clueRound, baseInputFD, baseFeaturesFD, outputDirectory=None, directOutput=False):
        """
            Generates all graphs for a given CLUE round based on a specific round, uses the files from that round and the base input and features files.
            If an output directory is specified, saves the graphs to that directory as PNG files.
            If directOutput is set to True, displays the graphs directly using matplotlib's interactive mode.
        """
        ClueLogger.logToFileOnly("generateGraphs called")
        if (len(pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile, header=0)) == 0):
            raise ClueException("Metadata file is empty")
        if (not directOutput and not outputDirectory):
            raise ClueException("In order to generate graphs, please select an outputDirectory or set directOutput to True")
        if (directOutput):
            plt.ioff()
        for graphIndex, graphFunction in enumerate(ClueGraphing._graphDict.values()):
            fig, ax = plt.subplots(constrained_layout=True)
            graphFunction(clueRound, baseInputFD, baseFeaturesFD, ax)
            ClueGraphing.graphs[list(ClueGraphing._graphDict.keys())[graphIndex]] = fig
            if (outputDirectory):
                plt.savefig(outputDirectory + "/" + list(ClueGraphing._graphDict.keys())[graphIndex] + ".png")
            else:
                ClueLogger.log("No output directory specified, skipping saving of graph " + list(ClueGraphing._graphDict.keys())[graphIndex])
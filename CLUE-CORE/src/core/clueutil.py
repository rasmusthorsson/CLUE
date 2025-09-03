import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ClueLogger:
    """
        Static logger class for CLUE, prints all the messages sent and also stores them in a 
        log file if a log file has been specified.
    """
    logFile = None

    @staticmethod
    def setLogFile(logFilePath):
        try:
            ClueLogger.logFile = open(logFilePath, "a")
        except Exception as e:
            print("Error opening log file: " + str(e))

    def log(*args, **kwargs):
        print(*args, **kwargs)
        if (ClueLogger.logFile):
            try:
                print(*args, file=ClueLogger.logFile, **kwargs)
                ClueLogger.logFile.flush()
            except Exception as e:
                print("Error writing to log file: " + str(e))

    @staticmethod
    def closeLogFile():
        if (ClueLogger.logFile):
            ClueLogger.logFile.close()
            ClueLogger.logFile = None


#Base class for exceptions in CLUE
class ClueException(Exception):
    pass

#Utility class for extracting features from the raw data
class FeatureExtractor:
    def __init__(self):
        pass

#------------ Feature Methods ------------------------------
#For additions of features, define a method for calculating the feature from one
#line of raw data then add the method with a name in __featuresDict below.

    #Hour with the highest value
    def __peakHour(line):
        peak = 0
        peakVal = 0
        for i in range(1, len(line)):
            if (line.iloc[i] > peakVal):
                peak = i
                peakVal = line.iloc[i]
        return peak
    
    #Highest value found
    def __peakHourVal(line):
        peakVal = 0
        for i in range(1, len(line)):
            if (line.iloc[i] > peakVal):
                peakVal = line.iloc[i]
        return peakVal

    #Hour with the lowest value
    def __baseHour(line):
        base = 0
        baseVal = float(sys.maxsize)
        for i in range(1, len(line)):
            if (line.iloc[i] < baseVal):
                base = i
                baseVal = line.iloc[i]
        return base

    #Lowest value found
    def __baseHourVal(line):
        baseVal = float(sys.maxsize)
        for i in range(1, len(line)):
            if (line.iloc[i] < baseVal):
                baseVal = line.iloc[i]
        return baseVal

#Add name of feature and method name in __featuresDict.
    __featuresDict = {
            "Peak Hour" : __peakHour,
            "Peak Value" : __peakHourVal,
            "Base Hour" : __baseHour,
            "Base Value" : __baseHourVal
        } 

#-----------------------------------------------------------

    #Extract features from and individual line, including "ID" feature
    @staticmethod
    def __extract(line):
        newLine = [line.iloc[0]] #TODO Fix bug where pandas converts to float for ID
        for v in FeatureExtractor.__featuresDict.values():
            newLine.append(v(line))
        return newLine

    #Returns a list of the features available, including "ID" feature
    @staticmethod
    def headers():
        headers = ["ID"]
        for k in FeatureExtractor.__featuresDict.keys():
            headers.append(k) 
        return headers

    #Extract features from a dataframe, returns a new dataframe with headers included
    @staticmethod
    def extractFromDataFrame(dataframe):
        outMatrix = np.matrix(FeatureExtractor.__extract(dataframe.iloc[0]))
        for i in range(1, len(dataframe)):
            outMatrix = np.vstack((outMatrix, FeatureExtractor.__extract(dataframe.iloc[i])))
        headers = FeatureExtractor.headers()
        return pd.DataFrame(data=outMatrix, columns=headers)

#Static class for filtering the input based on feature selection, cluster selection, clusters found from previous round
#and metadata from previous round
class InputFilter:

    #Returns an array of IDs to be part of the next round by parsing the cluster and metadata file in based on
    #the selections in the cluster selection file.
    @staticmethod
    def filterSelection(commands, metadataDF):
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

    #Filters out the non-selected features from the features file and removes IDs not part of the parsed cluster
    #selection.
    @staticmethod
    def filter(inputDF, clustersFD, metadataFD, featuresFD, selectionFD):
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
                #TODO Fix so that noise can be used as an explicit cluster if desired
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

#Constructing graphs from files and rounds
class ClueGraphing:

    graphs = {}

#---------------------------- Graphing Methods ----------------------------

    #Plot bands of the mean data for each cluster
    def __rawBands(clueRound, baseInputFD, baseFeaturesFD, ax):
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

    #Plot feature profiles of the raw feature data for each cluster 
    def __featureProfiles(clueRound, baseInputFD, baseFeaturesFD, ax): #TODO Use feature selection to limit plotted features
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
    __graphDict = {
        "Mean Raw Data" : __rawBands,
        "Feature Profiles" : __featureProfiles
        }

#-------------------------------------------------------------------------------

    def __init__(self):
        pass

    #Generates all plots specified in __graphDict
    @staticmethod
    def generateGraphs(clueRound, baseInputFD, baseFeaturesFD, outputDirectory=None, directOutput=False):
        if (len(pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile, header=0)) == 0):
            raise ClueException("Metadata file is empty")
        if (not directOutput and not outputDirectory):
            raise ClueException("In order to generate graphs, please select an outputDirectory or set directOutput to True")
        if (directOutput):
            plt.ioff()
        for graphIndex, graphFunction in enumerate(ClueGraphing.__graphDict.values()):
            fig, ax = plt.subplots(constrained_layout=True)
            graphFunction(clueRound, baseInputFD, baseFeaturesFD, ax)
            ClueGraphing.graphs[list(ClueGraphing.__graphDict.keys())[graphIndex]] = fig
            if (outputDirectory):
                plt.savefig(outputDirectory + "/" + list(ClueGraphing.__graphDict.keys())[graphIndex] + ".png")
            else:
                ClueLogger.log("No output directory specified, skipping saving of graph " + list(ClueGraphing.__graphDict.keys())[graphIndex])
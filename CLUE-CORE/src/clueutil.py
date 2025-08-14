from pathlib import Path

import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            "peakHour" : __peakHour,
            "peakHourVal" : __peakHourVal,
            "baseHour" : __baseHour,
            "baseHourVal" : __baseHourVal
        } 

#-----------------------------------------------------------

    #Extract features from and individual line, including "ID" feature
    @staticmethod
    def __extract(line):
        newLine = [line["ID"].astype(int)] #TODO Fix bug where pandas converts to float for ID
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
            print("No clusters remain after filtering out unwanted clusters, exiting prematurely..")
            sys.exit(2)
        return filteredDF

    #Filters out the non-selected features from the features file and removes IDs not part of the parsed cluster
    #selection.
    @staticmethod
    def filter(inputDF, clustersFD, metadataFD, featuresFD, selectionFD):
        #If no feature file or if feature file is empty, accept all features
        noFeatures = False 
        if (featuresFD != None):
            try:
                featuresDF = pd.read_csv(featuresFD, header=0)
            except pd.errors.EmptyDataError:
                noFeatures = True
        else:
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
                print("No clusters found, exiting early...")
                exit(2)
            if (selectionFD):
                selectionFile = open(selectionFD)
                commands = []
                for line in selectionFile:
                    command = line.rstrip().split(":")
                    commands.append(command)
                selectionFile.close()
                filteredSelectionDF = InputFilter.filterSelection(commands, metadataDF)
                filteredClustersDF = clustersDF.loc[clustersDF.iloc[:, 1].isin(map(int, filteredSelectionDF["ClusterId"]))]
                nextInput = nextInput.loc[nextInput["ID"].isin(map(int, filteredClustersDF.iloc[:, 0].array))]
            else:
                nextInput = nextInput.loc[nextInput["ID"].isin(map(int, clustersDF.iloc[:, 0].array))]
        elif (clustersFD and not metadataFD): #MetadataFD does not exist
            print("Exception: Metadata file does not exist but clusters file does, skipping filtering...")
        elif (metadataFD and not clustersFD): #ClustersDF does not exist
            print("Exception: Clusters file does not exist but metadata file does, skipping filtering...")

        return nextInput

#Constructing graphs from files and rounds
class ClueGraphing:

    #Plot bands of the mean data for each cluster
    def __rawBands(clueRound, baseInputFD, baseFeaturesFD):
        metadataDF = pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile)
        averagesColumns = [col for col in metadataDF.columns if col.startswith('Mean')]
        lines = []
        for rowIndex in range(len(metadataDF)):
            line = plt.plot(metadataDF[averagesColumns].iloc[rowIndex])
            lines.append(line)

        xTicks = []
        for index in range(len(averagesColumns)):
            if (index % 2 == 0):
                xTicks.append(index)
        plt.xticks(ticks=xTicks, rotation=45)
        plt.xlabel("Measurement Timestep")
        plt.ylabel("Consumption")

        legendLabels = []
        for index in range(len(lines)):
            legendLabels.append("Cluster " + str(index))
        plt.legend(legendLabels)

    #Plot feature profiles of the raw feature data for each cluster 
    def __featureProfiles(clueRound, baseInputFD, baseFeaturesFD): #TODO Use feature selection to limit plotted features
        baseFeaturesDF = pd.read_csv(baseFeaturesFD)
        clustersDF = pd.read_csv(clueRound.roundDirectory + clueRound.clustersFile)
        metadataDF = pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile)
        filteredFeaturesDF = InputFilter.filter(baseFeaturesDF, 
                                                clueRound.roundDirectory + clueRound.clustersFile, 
                                                clueRound.roundDirectory + clueRound.metadataFile, 
                                                None, 
                                                None)
        uniqueClusters = clustersDF.iloc[:, 1].unique()
        clusterFeaturesDict = {}
        for cluster in uniqueClusters:
            clusterIDsDF = clustersDF.loc[clustersDF.iloc[:, 1] == cluster].iloc[:, 0]
            clusterFilteredFeaturesDF = filteredFeaturesDF.loc[filteredFeaturesDF["ID"].isin(map(int, clusterIDsDF.array))]
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
            normalizedAverage = (average - np.min(average))/np.ptp(average) + offset
            normalizedTransposedAverages.append(normalizedAverage)

        normalizedAverages = np.transpose(normalizedTransposedAverages) #TODO Do not transpose
        
        heatmap = plt.imshow(normalizedAverages, cmap="Oranges")
        plt.colorbar(heatmap, ticks=[])
        plt.xticks(ticks=np.arange(len(xAxis)), labels=xAxis, rotation=45)
        plt.yticks(ticks=np.arange(len(yAxis)), labels=yAxis)

        plt.xlabel("Feature")
        plt.ylabel("Cluster")
        
        for column in range(len(clusterFeatureAverages)):
            for row in range(len(clusterFeatureAverages[column])):
                plt.text(row, 
                         column, 
                         round(clusterFeatureAverages[column][row], 2), 
                         ha="center", 
                         va="center", 
                         color="black")

    #Add new plots here
    __graphDict = {
        "rawBands" : __rawBands,
        "featureProfiles" : __featureProfiles
        }

    def __init__(self):
        pass

    #Generates all plots specified in __graphDict
    @staticmethod
    def generateGraphs(clueRound, baseInputFD, baseFeaturesFD, outputDirectory=None, directOutput=False):
        if (not directOutput and not outputDirectory):
            print("In order to generate graphs, please select an outputDirectory or set directOutput to True")
            return
        if (directOutput):
            plt.ioff()
        for graphIndex in range(0, len(ClueGraphing.__graphDict.values())):
            plt.figure(graphIndex + 1)
            list(ClueGraphing.__graphDict.values())[graphIndex](clueRound, baseInputFD, baseFeaturesFD)
        for graphIndex in range(0, len(ClueGraphing.__graphDict.keys())):
            plt.figure(graphIndex + 1)
            if (directOutput):
                plt.show()
            else:
                plt.savefig(outputDirectory + "/" + list(ClueGraphing.__graphDict.keys())[graphIndex] + ".png")

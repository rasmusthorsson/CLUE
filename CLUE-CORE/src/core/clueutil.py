import queue
import sys
import random
import threading
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

class ClueOnlyNoiseException(ClueException):
    """
        Exception raised when all data points are classified as noise
    """

    def __init__(self, message="All data points classified as noise"):
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

    commandsDict = {
            "OVER": {
                "description": "Keep clusters with size over specified value",
                "ifApply": lambda val: val is not None,
                "func": lambda df, val: df.loc[df["Size"] > int(val)] if val is not None else df,
                "operator": lambda df1, df2: df1.loc[df1["ClusterId"].isin(list(set(df1["ClusterId"]) & set(df2["ClusterId"])))],
                "type": int
            },
            "UNDER": {
                "description": "Keep clusters with size under specified value",
                "ifApply": lambda val: val is not None,
                "func": lambda df, val: df.loc[df["Size"] < int(val)] if val is not None else df,
                "operator": lambda df1, df2: df1.loc[df1["ClusterId"].isin(list(set(df1["ClusterId"]) & set(df2["ClusterId"])))],
                "type": int

            },
            "IN": {
                "description": "Keep only clusters with specified IDs",
                "ifApply": lambda val: bool(val),
                "func": lambda df, val: df.loc[df["ClusterId"].isin(map(int, val))] if val else df,
                "operator": lambda df1, df2: df1.loc[df1["ClusterId"].isin(list(set(df1["ClusterId"]) & set(df2["ClusterId"])))],
                "type": list
            },
            "NOTIN": {
                "description": "Remove clusters with specified IDs",
                "ifApply": lambda val: bool(val),
                "func": lambda df, val: df.loc[~df["ClusterId"].isin(map(int, val))] if val else df,
                "operator": lambda df1, df2: df1.loc[df1["ClusterId"].isin(list(set(df1["ClusterId"]) - set(df2["ClusterId"])))],
                "type": list
            }
        }

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

        inclusionUsed = False
        accDF = metadataDF
        for command, values in commands.items():
            if command in InputFilter.commandsDict:
                if not InputFilter.commandsDict[command]["ifApply"](values):
                    continue
                filteredDF = InputFilter.commandsDict[command]["func"](metadataDF, values)
                accDF = InputFilter.commandsDict[command]["operator"](accDF, filteredDF)
                if command == "IN":
                    inclusionUsed = True
                ClueLogger.log("Clusters in accDF after command '{}': {}".format(command, len(accDF)))
            else:
                ClueLogger.log(f"Warning: Unknown command '{command[0]}' ignored in selection filtering.")
        filteredDF = accDF
        # Noise has to be explicitly included, if no inclusion command was used remove noise.
        if not inclusionUsed:
            filteredDF = filteredDF.loc[filteredDF["ClusterId"] != -1]

        if (len(filteredDF) < 1):
            raise ClueException("No clusters remain after filtering out unwanted clusters")
        return filteredDF

    @staticmethod
    def filter(inputDF, clustersFD, metadataFD, features: list, selectionCommands: dict):
        """
            Filters the input dataframe based on the features file and the clusters and metadata files.
            If no features file is provided or if it is empty, all features are kept.
            If no clusters or metadata file is provided, all clusters are kept. 
            If only one of the files is provided, a warning is logged and no filtering is done.
            If a selection file is provided, it is used to filter the clusters further.
            Returns a filtered dataframe containing only the selected features and clusters.
        """
        ClueLogger.logToFileOnly("filter called")
    
        selectedColumns = []
        selectedFeatures = None
        if not features:
            selectedFeatures = inputDF
        #Otherwise filter based on feature selection
        else:
            for f1_i in range(len(inputDF.columns.array)):
                for f2 in features:
                    if (inputDF.columns[f1_i] == f2):
                        selectedColumns.append(f2)
            selectedFeatures = inputDF[selectedColumns]

        #If no cluster selection file, or if previous cluster or metadata is non-existant, accept all clusters
        nextInput = selectedFeatures
        
        if (clustersFD and metadataFD):
            clustersDF = pd.read_csv(clustersFD, header=None)
            metadataDF = pd.read_csv(metadataFD, index_col=False)
            if (clustersDF.empty or metadataDF.empty):
                ClueLogger.log("Either cluster or metadata file is empty, meaning no clusters were found in the previous round, stopping run...")
                raise ClueException("Clusters or metadata file is empty, cannot filter input")
            if selectionCommands:
                filteredSelectionDF = InputFilter.filterSelection(selectionCommands, metadataDF)
                filteredClustersDF = clustersDF.loc[clustersDF.iloc[:, 1].isin(map(int, filteredSelectionDF["ClusterId"]))]
                
                nextInput = nextInput.loc[nextInput["ID"].isin(filteredClustersDF.iloc[:, 0].array)]
            else:
                # Remove noise if no selection file is provided as noise has to be explicitly included
                noNoiseClustersDF = clustersDF.loc[clustersDF.iloc[:, 1] != -1]
                ClueLogger.log("No selection file provided, removing noise by default...")
                nextInput = nextInput.loc[nextInput["ID"].isin(noNoiseClustersDF.iloc[:, 0].array)]
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

    def _calculateMeans(inputDataFrame, clustersDataFrame):
        """
            Calculates the mean of each cluster, excluding the first column (ID).
            Returns four lists, the means, the cluster IDs, the lower quartile and the upper quartile.
        """
        means = []
        clusterIds = []
        lowerQuartile = []
        upperQuartile = []
        for cluster in clustersDataFrame[1].unique():
            clusterData = inputDataFrame[inputDataFrame["ID"].isin(clustersDataFrame[clustersDataFrame[1] == cluster][0])]
            means.append(clusterData.mean(axis=0).tolist())
            clusterIds.append(cluster)
            lowerQuartile.append(clusterData.quantile(0.25, axis=0).tolist())
            upperQuartile.append(clusterData.quantile(0.75, axis=0).tolist())
        return means, clusterIds, lowerQuartile, upperQuartile

    #TODO Separate calculations and plotting for graphs to allow threading of calculations
    @staticmethod
    def _tSNE(clueRound, baseInputFD, baseFeaturesFD, ax):
        """ 
            Plots a t-SNE graph using scikit-learn (much faster and more reliable).
        """
        
        ClueLogger.logToFileOnly("_tSNE called")
        
        # Load and filter data
        filteredInputDF = InputFilter.filter(pd.read_csv(baseInputFD), 
                                            clueRound.roundDirectory + clueRound.clustersFile, 
                                            clueRound.roundDirectory + clueRound.metadataFile, 
                                            None, 
                                            None)
        clustersDF = pd.read_csv(clueRound.roundDirectory + clueRound.clustersFile, header=None)
        
        # Prepare feature matrix (exclude ID column)
        featureMatrix = filteredInputDF.iloc[:, 1:].values.astype(float)
        
        # Handle insufficient data
        if len(featureMatrix) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for t-SNE\n(need at least 2 points)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('t-SNE Visualization - Insufficient Data')
            return
        
        # Get cluster labels
        clusterLabels = []
        for _, row in filteredInputDF.iterrows():
            dataId = row['ID']
            clusterMatch = clustersDF[clustersDF.iloc[:, 0] == dataId]
            if len(clusterMatch) > 0:
                clusterLabels.append(clusterMatch.iloc[0, 1])
            else:
                clusterLabels.append(-1)  # Noise
        
        nSamples = len(featureMatrix)
        ClueLogger.log(f"Running optimized t-SNE with {nSamples} samples...")
        
        # Standardize features for better t-SNE results
        scaler = StandardScaler()
        featureMatrixScaled = scaler.fit_transform(featureMatrix)
        
        # Configure t-SNE with optimal parameters
        perplexity = min(30, max(5, nSamples // 3))
        
        # Use optimized scikit-learn t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            max_iter=1000,
            random_state=42,
            method='barnes_hut' if nSamples > 250 else 'exact',
            n_jobs=1  # Use single core to avoid threading issues
        )
        
        embedding = tsne.fit_transform(featureMatrixScaled)
        
        # Plot results
        uniqueClusters = np.unique(clusterLabels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(uniqueClusters)))
        
        for i, clusterId in enumerate(uniqueClusters):
            mask = np.array(clusterLabels) == clusterId
            pointsInCluster = np.sum(mask)
            
            if clusterId == -1:
                ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                        c='lightgray', marker='x', s=40, alpha=0.6, 
                        label=f'Noise ({pointsInCluster})')
            else:
                ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                        c=[colors[i % len(colors)]], s=50, alpha=0.8, 
                        label=f'Cluster {clusterId} ({pointsInCluster})')
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE Visualization of Clusters')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ClueLogger.log("t-SNE visualization completed successfully")

    def _sizeBars(clueRound, baseInputFD, baseFeaturesFD, ax):
        """
            Plots a bar chart of the cluster sizes.
        """
        ClueLogger.logToFileOnly("_sizeBars called")

        #clustersDF = pd.read_csv(clueRound.roundDirectory + clueRound.clustersFile, header=None)
        metadataDF = pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile, index_col=False)

        if (len(metadataDF) == 0):
            raise ClueException("Metadata file is empty")

        # Create a mapping from cluster ID to color, ensures that colors are consistent across graphs
        clusterColorMap = ClueGraphing._getClusterColors(metadataDF['ClusterId'].unique())
        barColors = [clusterColorMap[cid] for cid in metadataDF['ClusterId']]

        ax.bar(metadataDF['ClusterId'].astype(str), metadataDF['Size'], color=barColors)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Size')
        ax.set_title('Cluster Sizes')
        ax.set_xticks(metadataDF['ClusterId'].astype(str))
        ax.set_xticklabels(metadataDF['ClusterId'].astype(str), rotation=45)
        ax.grid(axis='y', alpha=0.3)

    @staticmethod
    def _rawBands(clueRound, baseInputFD, baseFeaturesFD, ax):
        """
            Plots the raw bands of the mean data for each cluster.
        """
        ClueLogger.logToFileOnly("_rawBands called")

        #Filters noise out by default - TODO Allow noise to be included if desired
        filteredInputDF = InputFilter.filter(pd.read_csv(baseInputFD), 
                                            clueRound.roundDirectory + clueRound.clustersFile, 
                                            clueRound.roundDirectory + clueRound.metadataFile, 
                                            None, 
                                            None)
        #filteredInputDF = pd.read_csv(baseInputFD) ¤ Alternative to allow noise to be included
        clustersDF = pd.read_csv(clueRound.roundDirectory + clueRound.clustersFile, header=None)

        #Calculate means and bounds, do not use metadata as the clustering might have been based on features
        means, clusterIds, lowerBounds, upperBounds = ClueGraphing._calculateMeans(
                                    filteredInputDF, 
                                    clustersDF)

        # Use a consistent color mapping based on sorted cluster IDs
        clusterColorMap = ClueGraphing._getClusterColors(clusterIds)

        # Create the plot lines
        lines = []
        for rowIndex in range(len(means)):
            xValues = range(len(means[rowIndex]) - 1)  # Skip the ID column
            color = clusterColorMap[clusterIds[rowIndex]]
            # Plot the mean line
            meanLine, = ax.plot(means[rowIndex][1:], label=f"Cluster {clusterIds[rowIndex]}", color=color)  # Skip the ID column
            # Plot the upper and lower bounds as a filled area
            ax.fill_between(
                xValues,
                lowerBounds[rowIndex][1:],      # Lower bound (skip ID column)
                upperBounds[rowIndex][1:],      # Upper bound (skip ID column)
                color=color,
                alpha=0.2
            )
            lines.append(meanLine)
            
        #Sort lines by cluster ID for consistent legend ordering
        lines = [line for _, line in sorted(zip(clusterIds, lines), key=lambda pair: pair[0])]

        # Customize the x-axis
        xTicks = []
        xLabels = []
        for index in range(len(means[0]) - 1): #Skip the ID column
            if (index % 2 == 0):
                xLabels.append("Timestep " + str(index + 1))
                xTicks.append(index)
        # Set x-ticks and labels
        ax.set_xticks(ticks=xTicks)
        ax.set_xticklabels(labels=xLabels)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel("Measurement Timestep")
        ax.set_ylabel("Consumption")
        ax.legend(handles=lines, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title("Mean Raw Data with Interquartile Range")

    def _featureProfiles(clueRound, baseInputFD, baseFeaturesFD, ax): #TODO Use feature selection to limit plotted features
        """
            Plots the feature profiles of the mean features for each cluster.
        """
        ClueLogger.logToFileOnly("_featureProfiles called")

        #Features are already calulated, just need to filter based on clusters and metadata and then calculate means
        baseFeaturesDF = pd.read_csv(baseFeaturesFD)
        clustersDF = pd.read_csv(clueRound.roundDirectory + clueRound.clustersFile, header=None)
        metadataDF = pd.read_csv(clueRound.roundDirectory + clueRound.metadataFile, index_col=False)

        # Filter out noise by default - TODO Allow noise to be included if desired
        filteredFeaturesDF = InputFilter.filter(baseFeaturesDF, 
                                                clueRound.roundDirectory + clueRound.clustersFile, 
                                                clueRound.roundDirectory + clueRound.metadataFile, 
                                                None, 
                                                None)
      
        uniqueClusters = metadataDF['ClusterId'].unique()
        clusterFeaturesDict = {}
        for cluster in uniqueClusters:
            if cluster == -1:
                continue #Skip noise
            clusterIDsDF = clustersDF.loc[clustersDF.iloc[:, 1] == cluster].iloc[:, 0]
            clusterFilteredFeaturesDF = filteredFeaturesDF.loc[filteredFeaturesDF["ID"].isin(clusterIDsDF.array)]
            clusterFeaturesDict[cluster] = clusterFilteredFeaturesDF

        #Calculate feature averages for each cluster
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

        normalizedAverages = np.transpose(normalizedTransposedAverages)
        
        #Create the heatmap
        heatmap = ax.imshow(normalizedAverages, cmap="Oranges")
        plt.colorbar(heatmap, ticks=[])
        ax.set_xticks(ticks=np.arange(len(xAxis)), labels=xAxis)
        ax.tick_params(axis='x', rotation=45)
        ax.set_yticks(ticks=np.arange(len(yAxis)), labels=yAxis)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Cluster")
        ax.set_anchor('C')
        ax.set_title("Cluster Feature Profiles")
        
        #Add text annotations to each cell
        for column in range(len(clusterFeatureAverages)):
            for row in range(len(clusterFeatureAverages[column])):
                ax.text(row, 
                         column, 
                         round(clusterFeatureAverages[column][row], 2), 
                         ha="center", 
                         va="center", 
                         color="black")

    #TODO Possibly allow for more fine-grained selection of graphs to generate
    #Add new plots here
    _graphDictFast = {
        "Mean Raw Data" : _rawBands,
        "Feature Profiles" : _featureProfiles,
        "Cluster Sizes" : _sizeBars
        }

    _graphDictSlow = {
        "t-SNE Visualization" : _tSNE
    }

#-------------------------------------------------------------------------------

    def __init__(self):
        pass

    graphColors = []

    @staticmethod
    def _generateClusterColors(n):
        """
            Generates a list of distinct colors for plotting clusters.
            Uses matplotlib's tab10 colormap for up to 10 clusters, and extends with additional colors if needed.
        """
        ClueLogger.logToFileOnly("_generateClusterColors called")
        if ClueGraphing.graphColors is None or len(ClueGraphing.graphColors) < n:
            ClueGraphing.graphColors = plt.cm.get_cmap('tab10', n).colors

    @staticmethod
    def _getClusterColors(clusters):
        """
            Returns a dictionary mapping cluster IDs to colors.
            Ensures that each unique cluster ID is assigned a distinct color.
        """
        ClueLogger.logToFileOnly("_getClusterColors called")
        uniqueClusters = sorted(set(clusters))
        ClueGraphing._generateClusterColors(len(uniqueClusters))
        return {cid: ClueGraphing.graphColors[i] for i, cid in enumerate(uniqueClusters)}

    def clearGraphs():
        """
            Clears all stored graphs.
        """
        ClueLogger.logToFileOnly("clearGraphs called")
        ClueGraphing.graphs = {}

    def getRoundGraphs(roundName):
        """
            Returns the graphs for a specific round.
        """
        ClueLogger.logToFileOnly("getRoundGraphs called")
        if roundName in ClueGraphing.graphs:
            return ClueGraphing.graphs[roundName]
        else:
            return None

    @staticmethod
    def generateGraphs(clueRound, baseInputFD, baseFeaturesFD, outputDirectory=None, directOutput=False, fastOnly=False):
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
            ClueLogger.log("Direct output enabled, displaying graphs interactively...")
        if (fastOnly):
            ClueGraphing._graphDict = ClueGraphing._graphDictFast
        else:
            ClueGraphing._graphDict = {**ClueGraphing._graphDictFast, **ClueGraphing._graphDictSlow}
        roundGraphs = {}
        for graphIndex, graphFunction in enumerate(ClueGraphing._graphDict.values()):
            fig, ax = plt.subplots(constrained_layout=True)
            graphFunction(clueRound, baseInputFD, baseFeaturesFD, ax)
            roundGraphs[list(ClueGraphing._graphDict.keys())[graphIndex]] = fig
            if (outputDirectory and not directOutput):
                plt.savefig(outputDirectory + "/" + list(ClueGraphing._graphDict.keys())[graphIndex] + ".png")
            else:
                ClueLogger.log("No output directory specified, skipping saving of graph " + list(ClueGraphing._graphDict.keys())[graphIndex])
        if len(ClueGraphing.graphs) < 16:
            ClueGraphing.graphs[clueRound.roundName] = roundGraphs
        else: 
            ClueLogger.log("Maximum number of stored graphs reached, not storing graphs for round " + clueRound.roundName)
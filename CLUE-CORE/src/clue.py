#!/usr/bin/python3
import pathlib
import threading

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from core import cluerun, clueutil, clueserializer
from core.clueutil import ClueCancellation, ClueLogger as Logger
from ui import clueuiui
from ui.clueuiui import ClueGuiUI

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt

class ClueGui(ClueGuiUI):

    #Mapping of round names to buttons in the UI
    roundsDict = {}

    def autoUpdateRounds(func):
        """ 
            Decorator to automatically update the rounds list in the UI after modifying rounds.
        """
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.mainwindow.after(0, self.updateActiveRoundButton)
            return result
        return wrapper


    def _errorPopup(self, title, message):
        """
            Show an error popup with the given title and message.
        """
        errorPopup = tk.Toplevel(self.mainwindow)
        errorPopup.title(title)
        errorPopup.grab_set()
        errorPopup.focus_force()
        errorPopup.columnconfigure(0, weight=1)
        errorPopup.rowconfigure(0, weight=1)
        tk.Label(errorPopup, text=message).grid(row=0, column=0, padx=10, pady=10)
        tk.Button(errorPopup, text="OK", command=errorPopup.destroy).grid(row=1, column=0, padx=10, pady=10)
        errorPopup.update_idletasks()
        width = errorPopup.winfo_reqwidth()
        height = errorPopup.winfo_reqheight()
        minWidth = max(300, width)
        minHeight = max(100, height)
        errorPopup.geometry(f"{minWidth}x{minHeight}")
        Logger.log(f"Error Popup: {title} - {message}")

    # TODO Replace this with direct file selection as some will ask for save, some for open
    def _chooseFile(self, entryWidget, popupWindow, type=("CSV files", "*.csv")):
        """
            Open a file dialog to choose a file and set the entry widget's value to the selected file path.
        """
        filePath = tk.filedialog.asksaveasfilename(
            parent=popupWindow,
            title="Select Features File",
            filetypes=[type, ("All files", "*.*")]
        )
        if filePath:
            entryWidget.delete(0, tk.END)
            entryWidget.insert(0, filePath)

    def buildRoundPopup(self, 
                   currentAlgorithm=0, 
                   currentMetric=0, 
                   currentParamOpt=0,
                   currentEpsilon=1.0,
                   currentMinPts=5,
                   currentHyperplanes=5,
                   currentHashtables=5,
                   currentKClusters=5,
                   currentThreads=1,
                   currentStandardize=False,
                   currentUseFeatures=False,
                   currentFeaturesFile="",
                   currentSelectionFile="",
                   currentRoundName="",
                   ):
        """
            Build a popup window to configure a round. This sets up settings for the following:
            --- Round Settings ---
            - Round Name (string)
            - Features File (string)
            - Selection File (string)
            --- ClueConfig Settings ---
            - Algorithm (DBSCAN, IP.LSH.DBSCAN, KMeans)
            - Distance Metric (Euclidean, Angular, DTW)
            - Parameter Optimization Level (None, K-Distance, Grid Search)
            - Epsilon (float)
            - MinPts (int)
            - Hyperplanes (int)
            - Hashtables (int)
            - KClusters (int)
        """

        #Popup window creation   
        popup = tk.Toplevel(self.mainwindow)
        popup.title(f"Configure {currentRoundName}")
        popup.grab_set()  # Makes it modal
        popup.focus_force()

        popup.columnconfigure(0, weight=1)
        popup.columnconfigure(1, weight=1)
        popup.rowconfigure(0, weight=1)

        #-------------- Config Settings --------------

        #Config frame
        roundConfig = tk.LabelFrame(popup, text="Round Configs", padx=10, pady=10)
        roundConfig.grid(row=0, column=0, padx=10, pady=5, stick="nsew")

        #Algorithm selection
        tk.Label(roundConfig, text="Algorithm").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        chosenAlgorithm = ttk.Combobox(
            roundConfig,
            values=["DBSCAN", "IP.LSH.DBSCAN", "KMeans"],
            state="readonly"
        )
        chosenAlgorithm.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        chosenAlgorithm.current(currentAlgorithm)

        #Distance metric selection
        tk.Label(roundConfig, text="Distance Metric").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        chosenMetric = ttk.Combobox(
            roundConfig,
            values=["Euclidean", "Angular", "DTW"],
            state="readonly",
        )
        chosenMetric.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chosenMetric.current(currentMetric)

        #Parameter optimization level selection
        tk.Label(roundConfig, text="Parameter Optimizer").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        paramOptLevel = ttk.Combobox(
            roundConfig,
            values=["None", "K-Distance", "Grid Search"],
            state="readonly",
        )
        paramOptLevel.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        paramOptLevel.current(currentParamOpt)

        # Epsilon, MinPts, Hyperplanes, Hashtables, KClusters, Threads, Standardize, UseFeatures selection
        tk.Label(roundConfig, text="Epsilon").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        epsilonVar = tk.DoubleVar(value=currentEpsilon)
        tk.Entry(roundConfig, textvariable=epsilonVar).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(roundConfig, text="MinPts").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        minPtsVar = tk.IntVar(value=currentMinPts)
        tk.Entry(roundConfig, textvariable=minPtsVar).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(roundConfig, text="Hyperplanes").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        hyperplanesVar = tk.IntVar(value=currentHyperplanes)
        tk.Entry(roundConfig, textvariable=hyperplanesVar).grid(row=5, column=1, padx=5, pady=5) 

        tk.Label(roundConfig, text="Hashtables").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        hashtablesVar = tk.IntVar(value=currentHashtables)
        tk.Entry(roundConfig, textvariable=hashtablesVar).grid(row=6, column=1, padx=5, pady=5)

        tk.Label(roundConfig, text="KClusters").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        kClustersVar = tk.IntVar(value=currentKClusters)
        tk.Entry(roundConfig, textvariable=kClustersVar).grid(row=7, column=1, padx=5, pady=5)

        tk.Label(roundConfig, text="Threads").grid(row=8, column=0, padx=5, pady=5, sticky="e")
        threadsVar = tk.IntVar(value=currentThreads)
        tk.Entry(roundConfig, textvariable=threadsVar).grid(row=8, column=1, padx=5, pady=5)

        tk.Label(roundConfig, text="Standardize").grid(row=9, column=0, padx=5, pady=5, sticky="e")
        standardizeVar = tk.BooleanVar(value=currentStandardize)
        tk.Checkbutton(roundConfig, variable=standardizeVar).grid(row=9, column=1, padx=5, pady=5, sticky="w")

        tk.Label(roundConfig, text="Use Features").grid(row=10, column=0, padx=5, pady=5, sticky="e")
        useFeaturesVar = tk.BooleanVar(value=currentUseFeatures)
        tk.Checkbutton(roundConfig, variable=useFeaturesVar).grid(row=10, column=1, padx=5, pady=5, sticky="w")
        
        #-------------- Round Settings --------------

        # Round settings frame
        roundSettings = tk.LabelFrame(popup, text="Round Settings", padx=10, pady=10)
        roundSettings.grid(row=0, column=1, padx=10, pady=5, stick="nsew")

        # Round name selection
        tk.Label(roundSettings, text="Round Name:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        roundnameVar = tk.StringVar(roundSettings, value=currentRoundName)
        tk.Entry(roundSettings, textvariable=roundnameVar).grid(row=0, column=1, padx=5, pady=5)
    
        # Features file selection
        tk.Label(roundSettings, text="Features File:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        featuresFileEntry = tk.Entry(roundSettings)
        featuresFileEntry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        featuresFileEntry.insert(0, currentFeaturesFile) 
        featuresFileButton = tk.Button(
            roundSettings,
            text="Browse",
            command=lambda: self._chooseFile(featuresFileEntry, popup)
        )
        featuresFileButton.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # Cluster selection file selection
        tk.Label(roundSettings, text="Selection File:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        selectionFileEntry = tk.Entry(roundSettings)
        selectionFileEntry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        selectionFileEntry.insert(0, currentSelectionFile)
        selectionFileButton = tk.Button(
            roundSettings,
            text="Browse",
            command=lambda: self._chooseFile(selectionFileEntry, popup)
        )
        selectionFileButton.grid(row=2, column=2, padx=5, pady=5, sticky="ew")

        # Callback when user clicks "OK", reads values from the UI and creates/updates the round
        def onOk():
            """
                Callback when user clicks "OK", reads values from the UI and creates/updates the round.
            """

            # Grab values from the UI elements
            name = roundnameVar.get() or f"Round {self.roundCount + 1}"
            featureFile = str(featuresFileEntry.get()) or ""
            selectionFile = str(selectionFileEntry.get()) or ""
            algorithm = cluerun.Algorithm(chosenAlgorithm.current() + 1) or cluerun.Algorithm(currentAlgorithm)
            distanceMetric = cluerun.DistanceMetric(chosenMetric.current() + 1) or cluerun.DistanceMetric(currentMetric)
            paramOptimization = cluerun.ParameterOptimizationLevel(paramOptLevel.current())
            epsilon = epsilonVar.get() or currentEpsilon
            minPts = minPtsVar.get() or currentMinPts
            hyperplanes = hyperplanesVar.get() or currentHyperplanes
            hashtables = hashtablesVar.get() or currentHashtables
            kClusters = kClustersVar.get() or currentKClusters
            threads = threadsVar.get() or currentThreads
            standardize = standardizeVar.get() or currentStandardize
            useFeatures = useFeaturesVar.get() or currentUseFeatures

            # If the round does not have a button associated with it, it must be created
            if currentRoundName not in self.roundsDict.keys():
                if self.clueRun.getRound(roundnameVar.get()) is None:
                    # If the round does not already exist, create it
                    self.clueRun.buildRound(
                        name,
                        featureFile,
                        selectionFile,
                        cluerun.ClueConfig(
                            algorithm=algorithm,
                            distanceMetric=distanceMetric,
                            paramOptimization=paramOptimization,
                            epsilon=epsilon,
                            minPts=minPts,
                            hyperplanes=hyperplanes,
                            hashtables=hashtables,
                            kClusters=kClusters,
                            threads=threads,
                            standardize=standardize,
                            useFeatures=useFeatures
                        )
                    )
                    self.roundCount += 1
                    
                    # Define the callback function for the round button click
                    def roundClick():
                        """
                            Callback when the round button is clicked, calls the original function to open the round popup with current settings.
                            Results in different behaviour from the initial creation of the round since the round already exists.
                        """
                        clueRound = self.clueRun.getRound(name)
                        clueConfig = clueRound.clueConfig

                        # Create a new popup window when the round button is clicked to allow for editing
                        self.buildRoundPopup(
                            currentAlgorithm=clueConfig.algorithm.value - 1,
                            currentMetric=clueConfig.distanceMetric.value - 1,
                            currentParamOpt=clueConfig.paramOptimization.value,
                            currentEpsilon=clueConfig.epsilon,
                            currentMinPts=clueConfig.minPts,
                            currentHyperplanes=clueConfig.hyperplanes,
                            currentHashtables=clueConfig.hashtables,
                            currentKClusters=clueConfig.kClusters,
                            currentThreads=clueConfig.threads,
                            currentStandardize=clueConfig.standardize,
                            currentUseFeatures=clueConfig.useFeatures,
                            currentFeaturesFile=clueRound.featureSelectionFile or "",
                            currentSelectionFile=clueRound.clusterSelectionFile or "",
                            currentRoundName=name
                        )

                    # Create the new button in the round section of the UI
                    newRoundButton = tk.Button(
                        self.roundsFrame,
                        text=f"{name}", 
                        command=roundClick
                    )
                    self.roundsDict[name] = newRoundButton
                    _, currentRows = self.roundsFrame.grid_size()
                    newRoundButton.grid(row=currentRows, column=0, sticky="ew", pady=5)
                    self.roundsFrame.grid_columnconfigure(0, weight=1)
                else:
                    # If the name is already in the roundsDict, show an error
                    self._errorPopup("Round Exists", f"Round '{name}' already exists.")

            # If round already exists, update it
            else:
                # If the name is not the same as the current round name, show an error. This is done to avoid complexity of renaming rounds.
                if name != currentRoundName:
                    self._errorPopup("Not Implemented", f"Namechange not implemented yet. Please delete the round and create a new one with the desired name.")
                    return
                                
                # Update the round in the run
                self.clueRun.setRound(
                    name,
                    featureFile,
                    selectionFile,
                    cluerun.ClueConfig(
                        algorithm=algorithm,
                        distanceMetric=distanceMetric,
                        paramOptimization=paramOptimization,
                        epsilon=epsilon,
                        minPts=minPts,
                        hyperplanes=hyperplanes,
                        hashtables=hashtables,
                        kClusters=kClusters,
                        threads=threads,
                        standardize=standardize,
                        useFeatures=useFeatures
                    )
                )

            popup.destroy()

        tk.Button(popup, text="OK", command=onOk).grid(row=2, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=2, column=1, padx=5, pady=10)

    def addRound(self):
        """
            Add a new round to the current run while also adding it to the UI using buildRoundPopup.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        
        #Create a popup dialog to configure the round
        self.buildRoundPopup()

    def addExistingRound(self, _round):
        """
            Add an already existing round to the current run to the UI, this uses the existing settings of the round.
            Does not prompt for new settings.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return

        if self.clueRun.getRound(_round) is None:
            self._errorPopup("Round Not Found", f"Round '{_round}' does not exist.")
            return

        def roundClick():
            #Grab values from dictionary and Logger.log them
            clueRound = self.clueRun.getRound(_round)
            clueConfig = clueRound.clueConfig

            self.buildRoundPopup(
                currentAlgorithm=int(clueConfig.algorithm) - 1,
                currentMetric=int(clueConfig.distanceMetric) - 1,
                currentParamOpt=int(clueConfig.paramOptimization),
                currentEpsilon=clueConfig.epsilon,
                currentMinPts=clueConfig.minPts,
                currentHyperplanes=clueConfig.hyperplanes,
                currentHashtables=clueConfig.hashtables,
                currentKClusters=clueConfig.kClusters,
                currentThreads=clueConfig.threads,
                currentStandardize=clueConfig.standardize,
                currentUseFeatures=clueConfig.useFeatures,
                currentFeaturesFile=clueRound.featureSelectionFile or "",
                currentSelectionFile=clueRound.clusterSelectionFile or "",
                currentRoundName=_round
            )

        # Create a button for the existing round
        self.roundCount += 1

        roundButton = tk.Button(self.roundsFrame, text=_round, command=roundClick)
        _, currentRows = self.roundsFrame.grid_size()
        roundButton.grid(row=currentRows, column=0, sticky="ew", pady=5)
        self.roundsDict[_round] = roundButton

    def deleteRound(self):
        """
            Delete a round from the current run and remove it from the UI.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        
        #Create a popup dialog to select round name deletion and then delete that round
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Delete Round")
        popup.grab_set() 
        popup.focus_force()
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        # Round name selection dropdown
        roundNameLabel = tk.Label(popup, text="Select Round to Delete:")
        roundNameLabel.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        roundNameVar = tk.StringVar(popup)
        roundNameVar.set("Select Round")
        roundNameMenu = ttk.Combobox(
            popup,
            textvariable=roundNameVar,
            state="readonly"
        )
        roundNameMenu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        roundNameMenu['values'] = list(self.roundsDict.keys())
        if not roundNameMenu['values']:
            self._errorPopup("No rounds", "No rounds available to delete.")
            popup.destroy()  # Close the popup if no rounds are available
            return
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get()))

        def on_delete():
            """
                Callback when the delete button is clicked, deletes the selected round from the run and removes it from the UI.
            """
            roundName = roundNameVar.get()
            if roundName in self.roundsDict:
                self.clueRun.removeRound(roundName)     # Remove the run logically
                self.roundsDict[roundName].destroy()    # Remove the button from the UI
                del self.roundsDict[roundName]          # Remove the button from the dictionary
                self.updateActiveRoundButton()          # Update the all round buttons to reflect the change
                Logger.log("Round Deleted", f"Round '{roundName}' has been deleted.")
            else:
                self._errorPopup("Round Not Found", f"Round '{roundName}' does not exist.")
            popup.destroy()

        tk.Button(popup, text="Delete", command=on_delete).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)

    def moveRoundUp(self):
        """
            Create a popup to select a round to move up in the list of rounds to be ran.
            Also moves the button in the UI to reflect the change.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        if not list(self.roundsDict):
            self._errorPopup("No Rounds", "No rounds available to move up.")
            return

        # Popup window for moving the round up
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Move Round Up")
        popup.grab_set() 
        popup.focus_force()
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        # Round name selection dropdown
        roundNameLabel = tk.Label(popup, text="Select Round to Move Up:")
        roundNameLabel.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        roundNameVar = tk.StringVar(popup)
        roundNameVar.set("Select Round")
        roundNameMenu = ttk.Combobox(
            popup,
            textvariable=roundNameVar,
            state="readonly"
        )
        roundNameMenu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        roundNameMenu['values'] = list(self.roundsDict.keys())
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get()))

        def onMoveUp():
            """
                Callback function for moving the selected round up in the list.
            """
            try:

                roundName = roundNameVar.get()
                currentIndex = self.clueRun.getRoundIndex(roundName)
                if currentIndex == -1:
                    self._errorPopup("Round Not Found", f"Round '{roundName}' does not exist.")
                if currentIndex > 0:
                    # Swap the current round with the one above it
                    aboveRoundName = list(self.clueRun.rounds)[currentIndex - 1].roundName
                    self.clueRun.moveRoundUp(roundName) # Logical move
                    self.updateActiveRoundButton()      # Update button styles to reflect the reset caused by logical move

                    # Update the UI
                    self.roundsDict[roundName].grid(row=currentIndex - 1, column=0, sticky="ew", pady=5)
                    self.roundsDict[aboveRoundName].grid(row=currentIndex, column=0, sticky="ew", pady=5)
                    Logger.log("Round Moved Up: ", f"Round '{roundName}' has been moved up.")
                else:
                    self._errorPopup("Already at Top: ", f"Round '{roundName}' is already at the top.")
            except clueutil.ClueException as e:
                self._errorPopup("Error Moving Round", str(e))
            popup.destroy()

        tk.Button(popup, text="Move Up", command=onMoveUp).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)
        
    def moveRoundDown(self):
        """
            Create a popup to select a round to move down in the list of rounds to be ran.
            Also moves the button in the UI to reflect the change.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        if not list(self.roundsDict):
            self._errorPopup("No Rounds", "No rounds available to move down.")
            return

        # Popup window for moving the round down
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Move Round Down")
        popup.grab_set() 
        popup.focus_force()
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1) 

        # Round name selection dropdown
        roundNameLabel = tk.Label(popup, text="Select Round to Move Down:")
        roundNameLabel.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        roundNameVar = tk.StringVar(popup)
        roundNameVar.set("Select Round")
        roundNameMenu = ttk.Combobox(
            popup,
            textvariable=roundNameVar,
            state="readonly"
        )
        roundNameMenu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        roundNameMenu['values'] = list(self.roundsDict.keys())
        if not roundNameMenu['values']:
            self._errorPopup("No Rounds", "No rounds available to move down.")
            popup.destroy()
            return
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get())) 

        def onMoveDown():
            """
                Callback function for moving the selected round down in the list.
            """
            try:
                roundName = roundNameVar.get()
                currentIndex = self.clueRun.getRoundIndex(roundName)
                if currentIndex == -1:
                    self._errorPopup("Round Not Found: ", f"Round '{roundName}' does not exist.")
                    return
                if currentIndex < len(self.clueRun.rounds) - 1:
                    belowRoundName = list(self.clueRun.rounds)[currentIndex + 1].roundName
                    self.clueRun.moveRoundDown(roundName)
                    self.updateActiveRoundButton()
                    
                    # Update the UI
                    self.roundsDict[roundName].grid(row=currentIndex + 1, column=0, sticky="ew", pady=5)
                    self.roundsDict[belowRoundName].grid(row=currentIndex, column=0, sticky="ew", pady=5)
                    Logger.log("Round Moved Down: ", f"Round '{roundName}' has been moved down.")
                else:
                    self._errorPopup("Already at Bottom: ", f"Round '{roundName}' is already at the bottom.")
            except clueutil.ClueException as e:
                self._errorPopup("Error Moving Round", str(e))
            popup.destroy()

        tk.Button(popup, text="Move Down", command=onMoveDown).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)
    
    @autoUpdateRounds
    def clearRounds(self):
        """
            Clear all rounds from the current run and update the UI.
        """
        for _, button in self.roundsDict.items():
            button.destroy()
        self.roundsDict.clear()
        self.roundCount = 0

    def changeRunNameLabel(self, newName):
        """
            Change the label of the run in the UI.
        """
        if self.clueRun is not None:
            self.builder.get_object("run_name_label", self.mainwindow).config(text=newName)
        else:
            self._errorPopup("No Run Defined", "Please create a run first.")

    @autoUpdateRounds
    def newRun(self):
        """
            Create a new run and clear any existing rounds. Name is specified in a popup dialog.
        """
        if self.runThread is not None and self.runThread.is_alive():
            self._errorPopup("Run In Progress", "Cannot create a new run while another run is in progress.")
            return
        
        self.clearRounds()
        
        # New Run name popup dialog
        popup = tk.Toplevel(self.mainwindow)
        popup.title(f"Create New Run")
        popup.grab_set()
        popup.focus_force()

        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        newRunName = tk.Frame(popup, padx=10, pady=10)
        newRunName.grid(row=0, column=0, padx=10, pady=5, stick="nsew")

        #Round name entry
        tk.Label(newRunName, text="Run Name:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        runNameVar = tk.StringVar(newRunName, value="New Run")
        tk.Entry(newRunName, textvariable=runNameVar).grid(row=0, column=1, padx=5, pady=5)

        def onOk():
            """
                Create a new ClueRun instance with the specified parameters.
            """
            newName = runNameVar.get() or "Run1"
            inputFile = self.builder.get_object("input_file_entry", self.mainwindow).get()
            directory = self.builder.get_object("directory_entry", self.mainwindow).get()
            clueclustLocation = self.builder.get_object("clueclust_entry", self.mainwindow).get()
            self.clueRun = cluerun.ClueRun(
                newName,
                str(inputFile),
                str(pathlib.Path(directory)),
                CLUECLUST=str(clueclustLocation)
            )
            self.changeRunNameLabel(newName)
            Logger.log("New Run Created:", f"Run '{newName}' has been created.")
            popup.destroy()

        tk.Button(popup, text="OK", command=onOk).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)
    
    def saveRun(self):
        """
            Save the current CLUE run to an XML file.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        
        filePath = filedialog.asksaveasfilename(
            parent=self.mainwindow,
            title="Save CLUE Run",
            defaultextension=".xml",
            filetypes=[("CLUE XML Run files", "*.xml"), ("All files", "*.*")]
        )
        if filePath:
            try:
                clueserializer.serialize_cluerun(self.clueRun, filePath)
                Logger.log(f"Run Saved: CLUE run saved to {filePath}.")
            except clueutil.ClueException as e:
                self._errorPopup("Error Saving Run", str(e))

    def loadRun(self):
        """
            Load a CLUE run from an XML file.
        """
        filePath = filedialog.askopenfilename(
            parent=self.mainwindow,
            title="Load CLUE Run",
            filetypes=[("CLUE XML Run files", "*.xml"), ("All files", "*.*")]
        )
        if filePath:
            try:
                self.clueRun = clueserializer.deserialize_cluerun(filePath)
                Logger.log(f"Run Loaded: CLUE run loaded from {filePath}.")
            except clueutil.ClueException as e:
                self._errorPopup("Error Loading Run", str(e))
            self.buildExistingRun()

    @autoUpdateRounds
    def buildExistingRun(self):
        """
            Builds all the round buttons and updates all the UI elements for an already existing run.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        self.clearRounds()
        self.changeRunNameLabel(self.clueRun.runName)

        # Update global setting entries
        self.builder.get_object("input_file_entry", self.mainwindow).delete(0, tk.END)
        self.builder.get_object("input_file_entry", self.mainwindow).insert(0, str(self.clueRun.baseFile))
        self.builder.get_object("directory_entry", self.mainwindow).delete(0, tk.END)
        self.builder.get_object("directory_entry", self.mainwindow).insert(0, str(self.clueRun.baseDirectory))
        self.builder.get_object("clueclust_entry", self.mainwindow).delete(0, tk.END)
        self.builder.get_object("clueclust_entry", self.mainwindow).insert(0, str(self.clueRun.CLUECLUST))
        for round in self.clueRun.rounds:
            self.addExistingRound(round.roundName)

    def createPlotTab(self, notebook, fig, name):
        """
            Create a tab in the notebook for displaying plots.
        """
        plotTab = ttk.Frame(notebook)
        notebook.add(plotTab, text=name)

        figCanvas = FigureCanvasTkAgg(fig, master=plotTab)
        figCanvas.draw()
        figCanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add plot to the navigation toolbar
        toolbar = NavigationToolbar2Tk(figCanvas, plotTab)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill="both")

    def buildGraphs(self, round, baseFile, baseFeaturesFile, outputDirectory, directOutput, fastGraphsOnly):
        """
            Build the graphs for the given round and add them to the UI.
        """
        try:
            clueutil.ClueGraphing.generateGraphs(
                round,
                baseFile,
                baseFeaturesFile,
                outputDirectory=outputDirectory,
                directOutput=directOutput,
                fastOnly=fastGraphsOnly
            )
        except clueutil.ClueException as e:
            self._errorPopup("Graph Generation Error", str(e))
            return
        except Exception as e:
            self._errorPopup("Unexpected Error", str(e))
            return
        
        roundName = round.roundName if round is not None else "None"
        self.builder.get_object("round_graph_name", self.mainwindow).config(text=roundName)

        for tab in self.plotNotebook.tabs():
            self.plotNotebook.forget(tab)

        for name, graph in clueutil.ClueGraphing.graphs.items():
            fig = graph
            if fig is not None:
                # Create a new tab in the notebook for each figure
                self.createPlotTab(self.plotNotebook, fig, name)
                plt.close(fig)
            else:
                self._errorPopup("No figure found for graph: ", name)

    def cleanupAfterRun(self, button, finalRound, baseFile, baseFeaturesFile, outputDirectory, directOutput, fastGraphsOnly=True):
        """
            Cleanup the UI after a run has completed, re-enabling buttons and building graphs.
        """
        self.buildGraphs(finalRound, baseFile, baseFeaturesFile, outputDirectory, directOutput, fastGraphsOnly=fastGraphsOnly)
        self.stopRunningIndicator(button)
        self.runThread = None

    def updateActiveRoundButton(self):
        """
            Update the visual state of the round buttons to indicate which round is active, completed, or not yet run.
        """
        if self.clueRun is None:
            return
        
        roundPointer = self.clueRun.getRoundPointer()
        
        for _, (roundName, button) in enumerate(self.roundsDict.items()):
            roundIndex = self.clueRun.getRoundIndex(roundName) 
            if roundIndex == roundPointer:
                # Next round to run
                button.config(
                    text=f"▶ {roundName}",              # Arrow indicator
                    bg="lightblue",                     # Light blue background
                    font=("TkDefaultFont", 9, "bold"),  # Bold font
                    relief="solid",                     # Solid border style
                    borderwidth=2,                      # Thicker border
                    fg="darkblue"                       # Dark blue text
                )
            elif roundIndex == roundPointer - 1:
                # Just completed round
                button.config(
                    text=f"✓ {roundName}",                 # Checkmark indicator
                    bg="lightgreen",                       # Green background
                    font=("TkDefaultFont", 9, "normal"),   # Normal font
                    relief="solid",                        # Normal solid style
                    borderwidth=1,                         # Normal border
                    fg="darkgreen"                         # Dark green text
                )
            elif roundIndex < roundPointer:
                # Completed rounds
                button.config(
                    text=f"✓ {roundName}",                 # Checkmark indicator
                    bg="palegreen",                        # Pale green background
                    font=("TkDefaultFont", 9, "normal"),   # Normal font
                    relief="raised",                       # Normal raised style
                    borderwidth=1,                         # Normal border
                    fg="darkgreen"                         # Dark green text
                )
            else:
                # Future round
                button.config(
                    text=f"  {roundName}",               # Indented (no indicator)
                    bg="grey85",                         # Default background
                    font=("TkDefaultFont", 9, "normal"), # Normal font
                    relief="raised",                     # Normal raised style
                    borderwidth=1,                       # Normal border
                    fg="black"                           # Normal text color
                )

    def runCheck(self):
        """
            Check if a run can be started. Ensures a run is defined, rounds exist, rounds remain to be run, and no other run is in progress.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return False
        if not self.clueRun.rounds:
            self._errorPopup("No Rounds", "No rounds defined in the current run. Please add rounds before running.")
            return False
        if self.clueRun.getRoundPointer() >= len(self.clueRun.rounds):
            self._errorPopup("All Rounds Completed", "All rounds in the current run have already been completed.")
            return False
        if self.runThread and self.runThread.is_alive():
            self._errorPopup("Run in Progress", "A CLUE run is already in progress. Please wait for it to complete before starting a new one.")
            return False
        return True
    
    def runNextRound(self):
        """
            Execute the next round in the current CLUE run. Threaded to keep the UI responsive.
        """
        try:
            Logger.log("Running CLUE: Starting the next round...")
            _round = self.clueRun.runNextRound() # Logically run the next round
            
            # Schedule the cleanup and graph building on the main thread
            if _round < len(self.clueRun.rounds): # If there are more rounds to run after this one
                self.mainwindow.after(0, self.cleanupAfterRun,
                            self.buttonRunNext,
                            self.clueRun.rounds[_round - 1],
                            self.clueRun.baseFile,
                            self.clueRun.baseFeaturesFile,
                            str(self.clueRun.rounds[_round - 1].roundDirectory),
                            True)
            else: # If this was the last round
                self.mainwindow.after(0, lambda: self.cleanupAfterRun(
                            self.buttonRunNext,
                            self.clueRun.rounds[len(self.clueRun.rounds) - 1],
                            self.clueRun.baseFile,
                            self.clueRun.baseFeaturesFile,
                            str(self.clueRun.targetRunDirectory) + "/" + self.clueRun.outputDirectory,
                            True,
                            fastGraphsOnly=False))
            if self.clueRun.getRoundPointer() < len(self.clueRun.rounds):
                Logger.log("Round Complete: The current round has completed successfully. Ready for the next round.")
            else:
                Logger.log("Run Complete: The CLUE run has completed successfully.")
            if ClueCancellation.isCancellationRequested():
                Logger.log("Run Cancelled: The CLUE round was completed before it could be cancelled.")
                ClueCancellation.clearCancellation()
            self.updateActiveRoundButton()
        except clueutil.ClueCancelledException as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunCancelled(self.buttonRunNext, errorMsg))
        except clueutil.ClueException as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunError(self.buttonRunNext, errorMsg))
        except Exception as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunError(self.buttonRunNext, errorMsg))

    def runRest(self):
        """
            Execute the remainder of the current CLUE run. Threaded to keep the UI responsive.
        """
        try:
            Logger.log("Running CLUE: Starting the remainder of the CLUE run...")
            self.clueRun.runRemainder() # Logically run the rest of the rounds

            # Schedule the cleanup and graph building on the main thread
            self.mainwindow.after(0, self.cleanupAfterRun,
                            self.buttonRunRest,
                            self.clueRun.rounds[len(self.clueRun.rounds) - 1],
                            self.clueRun.baseFile,
                            self.clueRun.baseFeaturesFile,
                            str(self.clueRun.targetRunDirectory) + "/" + self.clueRun.outputDirectory,
                            True)
            Logger.log("Run Complete: The CLUE run has completed successfully.")
            if ClueCancellation.isCancellationRequested():
                Logger.log("Run Cancelled: The CLUE run was completed before it could be cancelled.")
                ClueCancellation.clearCancellation()
            self.updateActiveRoundButton()
        except clueutil.ClueCancelledException as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunCancelled(self.buttonRunRest, errorMsg))
        except clueutil.ClueException as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunError(self.buttonRunRest, errorMsg))
        except Exception as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunError(self.buttonRunRest, errorMsg))

    def runClueFromBeginning(self):
        """
            Execute the entire CLUE run from the beginning. Threaded to keep the UI responsive.
        """
        try:
            Logger.log("Running CLUE: Starting the CLUE run...")
            self.clueRun.runFromBeginning() # Logically run the entire run from the beginning

            # Schedule the cleanup and graph building on the main thread
            self.mainwindow.after(0, self.cleanupAfterRun,
                            self.buttonRunClue,
                            self.clueRun.rounds[len(self.clueRun.rounds) - 1],
                            self.clueRun.baseFile,
                            self.clueRun.baseFeaturesFile,
                            str(self.clueRun.targetRunDirectory) + "/" + self.clueRun.outputDirectory,
                            True)
            Logger.log("Run Complete: The CLUE run has completed successfully.")
            if ClueCancellation.isCancellationRequested():
                Logger.log("Run Cancelled: The CLUE run was completed before it could be cancelled.")
                ClueCancellation.clearCancellation()
            self.updateActiveRoundButton()
        except clueutil.ClueCancelledException as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunCancelled(self.buttonRunClue, errorMsg))
        except clueutil.ClueException as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunError(self.buttonRunClue, errorMsg))
        except Exception as e:
            errorMsg = f"{type(e).__name__}: {str(e)}"
            self.mainwindow.after(0, lambda: self.onRunError(self.buttonRunClue, errorMsg))

    def runNextRoundButton(self):
        """
            Method called when the run next round button is clicked. Starts the next round in a separate thread.
        """
        if not self.runCheck():
            return
        
        #Set global run settings before run
        inputFile = self.builder.get_object("input_file_entry", self.mainwindow).get()
        outputDirectory = self.builder.get_object("directory_entry", self.mainwindow).get()
        clueclustLocation = self.builder.get_object("clueclust_entry", self.mainwindow).get()

        if not inputFile or not outputDirectory or not clueclustLocation:
            self._errorPopup("Missing Information", "Please ensure all fields are filled out before running CLUE.")
            return
        else:
            self.clueRun.updateBaseFile(str(inputFile))
            self.clueRun.updateBaseDirectory(str(outputDirectory))
            self.clueRun.CLUECLUST = str(clueclustLocation)

            self.startRunningIndicator(self.buttonRunNext)  # Start the running indicator on the button
            ClueCancellation.clearCancellation()            # Clear any previous cancellation requests

            self.runThread = threading.Thread(target=self.runNextRound)
            self.runThread.start()

    def runRestButton(self):
        """
            Method called when the run rest button is clicked. Starts the remainder of the run in a separate thread.
        """
        if not self.runCheck():
            return
        
        #Set global run settings before run
        inputFile = self.builder.get_object("input_file_entry", self.mainwindow).get()
        outputDirectory = self.builder.get_object("directory_entry", self.mainwindow).get()
        clueclustLocation = self.builder.get_object("clueclust_entry", self.mainwindow).get()

        if not inputFile or not outputDirectory or not clueclustLocation:
            self._errorPopup("Missing Information", "Please ensure all fields are filled out before running CLUE.")
            return
        else:
            self.clueRun.updateBaseFile(str(inputFile))
            self.clueRun.updateBaseDirectory(str(outputDirectory))
            self.clueRun.CLUECLUST = str(clueclustLocation)

            self.startRunningIndicator(self.buttonRunRest)  # Start the running indicator on the button
            ClueCancellation.clearCancellation()            # Clear any previous cancellation requests

            self.runThread = threading.Thread(target=self.runRest)
            self.runThread.start()

    def runClueButton(self):
        """
            Method called when the run clue button is clicked. Starts the entire run from the beginning in a separate thread.
        """ 
        if not self.runCheck():
            return
        
        #Set global run settings before run
        inputFile = self.builder.get_object("input_file_entry", self.mainwindow).get()
        outputDirectory = self.builder.get_object("directory_entry", self.mainwindow).get()
        clueclustLocation = self.builder.get_object("clueclust_entry", self.mainwindow).get()

        if not inputFile or not outputDirectory or not clueclustLocation:
            self._errorPopup("Missing Information", "Please ensure all fields are filled out before running CLUE.")
            return
        else:
            self.clueRun.updateBaseFile(str(inputFile))
            self.clueRun.updateBaseDirectory(str(outputDirectory))
            self.clueRun.CLUECLUST = str(clueclustLocation)

            self.startRunningIndicator(self.buttonRunClue)  # Start the running indicator on the button
            ClueCancellation.clearCancellation()            # Clear any previous cancellation requests

            self.runThread = threading.Thread(target=self.runClueFromBeginning)
            self.runThread.start()

    def startRunningIndicator(self, button):
        """ 
            Start a flashing "Running" indicator on the given button and disable other run buttons. 
        """
        self.runningButton = button
        self.originalText = button.cget("text")
        self.isRunning = True
        self.flashState = 0
        self.flashButton(button)
        for btn in self.blockButtons:
            btn.config(state="disabled")
    
    def stopRunningIndicator(self, button):
        """ 
            Stop the flashing "Running" indicator on the given button and re-enable other run buttons.
        """
        self.isRunning = False
        if self.originalText:
            button.config(text=self.originalText)
        for btn in self.blockButtons:
            btn.config(state="normal")

    def flashButton(self, button):
        """
            Flash the given button name to indicate running state.
        """
        if self.isRunning:
            if self.flashState == 0:
                button.config(text="Running    ")
            elif self.flashState == 1:
                button.config(text="Running.   ")
            elif self.flashState == 2:
                button.config(text="Running..  ")
            elif self.flashState == 3:
                button.config(text="Running... ")
            self.flashState = (self.flashState + 1) % 4
            self.mainwindow.after(250, lambda: self.flashButton(button)) # Update every 250 ms
        else:
            button.config(text=self.originalText, state="normal")

    @autoUpdateRounds
    def resetRunButton(self):
        """
            Reset the current run to the beginning, allowing all rounds to be run again.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        if self.runThread and self.runThread.is_alive():
            self._errorPopup("Run in Progress", "A CLUE run is already in progress. Please wait for it to complete before resetting.")
            return
        self.clueRun.resetRun()
        Logger.log("Run Reset: " + self.clueRun.runName + " has been reset to the beginning.")

    def cancelRunButton(self):
        """
            Cancel the currently running CLUE run.
            This sets a cancellation flag that is checked periodically during the run.
        """
        if self.clueRun is None:
            self._errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        if self.runThread and self.runThread.is_alive():
            ClueCancellation.requestCancellation()
            Logger.log("Run Cancellation Requested: Attempting to cancel the current CLUE run...")
        else:
            self._errorPopup("No Run in Progress", "No CLUE run is currently in progress to cancel.")

    def onRunError(self, button, errorMsg):
        """
            Handle errors that occur during the CLUE run, displaying an error popup and stopping the running indicator.
        """
        self.stopRunningIndicator(button)
        self._errorPopup("Clue Run Error", str(errorMsg))
        self.runThread = None
        self.updateActiveRoundButton()

    def onRunCancelled(self, button, errorMsg):
        """
            Handle cancellation of the CLUE run, displaying a cancellation message and stopping the running indicator.
        """
        self.stopRunningIndicator(button)
        Logger.log("Run " + self.clueRun.runName + " has been cancelled: " + str(errorMsg))
        self.runThread = None
        self.updateActiveRoundButton()

    def selectInputFile(self):
        """
            Select the input file for the CLUE run.
        """
        filePath = filedialog.askopenfilename(
            parent=self.mainwindow,
            title="Select Input File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filePath:
            self.clueRun.baseFile = str(filePath)
            self.builder.get_object("input_file_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("input_file_entry", self.mainwindow).insert(0, filePath)

    def selectOutputDirectory(self):
        """
            Select the output directory for the CLUE run.
        """
        directoryPath = filedialog.askdirectory(
            parent=self.mainwindow,
            title="Select Output Directory"
        )
        if directoryPath:
            self.clueRun.updateBaseDirectory(str(directoryPath))
            self.builder.get_object("directory_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("directory_entry", self.mainwindow).insert(0, str(directoryPath))

    def selectClueclustLocation(self):
        """
            Select the location of the Clueclust jar file.
        """
        filePath = filedialog.askopenfilename(
            parent=self.mainwindow,
            title="Select Clueclust Jar",
            filetypes=[("Jar files", "*.jar"), ("All files", "*.*")]
        )
        if filePath:
            self.clueRun.CLUECLUST = str(filePath)
            self.builder.get_object("clueclust_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("clueclust_entry", self.mainwindow).insert(0, filePath)

    def logSettingsPopup(self):
        """
            Create a popup to adjust logging settings, including toggling file logging and selecting the log file.
        """

        # Create the popup window
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Log Settings")
        popup.grab_set()
        popup.focus_force()
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

        settingsFrame = tk.Frame(popup, padx=10, pady=10)
        settingsFrame.grid(row=0, column=0, padx=10, pady=5, stick="nsew")

        # Log to file checkbox, logs to file if enabled and file is selected
        logToFileVar = tk.BooleanVar(value=Logger.logToFile)
        tk.Checkbutton(settingsFrame, text="Log to File", variable=logToFileVar).grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Log file selection
        tk.Label(settingsFrame, text="Log File:").grid(row=0, column=1, padx=5, pady=5, sticky="e")
        logFileEntry = tk.Entry(settingsFrame)
        logFileEntry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        logFileEntry.insert(0, Logger.logFile.name if Logger.logFile else "")

        # Browse button to select log file
        logFileButton = tk.Button(
            settingsFrame,
            text="Browse",
            command=lambda: self._chooseFile(logFileEntry, popup, type=("Log files", "*.log"))
        )
        logFileButton.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        def onOk():
            """
                Apply the logging settings from the popup.
            """
            Logger.logToFile = logToFileVar.get()
            Logger.setLogFile(str(logFileEntry.get()) or None)
            Logger.log("Settings Updated: Logging settings have been updated.")
            popup.destroy()

        tk.Button(popup, text="OK", command=onOk).grid(row=2, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=2, column=1, padx=5, pady=10)

    def updateQueue(self):
        """
            Periodically check the Logger's message queue and update the text box with new messages.
        """
        messages = Logger.getQueuedMessages()
        for msg in messages:
            if msg.endswith('\n'):
                self.textBox.insert(tk.END, msg)
            else:
                self.textBox.insert(tk.END, msg + "\n")
            self.textBox.see(tk.END)
        self.mainwindow.after(100, self.updateQueue)  # Check the queue every 100 ms

    def __init__(self, master=None):
        """
            Initialize the GUI components.
        """
        super().__init__(master)
        self.runThread = None
        self.clueRun = None
        self.roundCount = 0
        self.builder.add_from_file(clueuiui.PROJECT_UI)
        self.runningButton = None

        self.mainwindow = self.builder.get_object("base_frame", master)
        
        menuCommandAddRound = self.builder.get_object("new_round_button", self.mainwindow)
        menuCommandAddRound.config(command=self.addRound)

        menuCommandDeleteRound = self.builder.get_object("delete_round_button", self.mainwindow)
        menuCommandDeleteRound.config(command=self.deleteRound)

        menuCommandMoveRoundUp = self.builder.get_object("move_round_up_button", self.mainwindow)
        menuCommandMoveRoundUp.config(command=self.moveRoundUp)

        menuCommandMoveRoundDown = self.builder.get_object("move_round_down_button", self.mainwindow)
        menuCommandMoveRoundDown.config(command=self.moveRoundDown)

        self.roundsFrame = self.builder.get_object("rounds_list_frame")

        buttonRunClue = self.builder.get_object("run_button", self.mainwindow)
        buttonRunClue.config(command=self.runClueButton)
        self.buttonRunClue = buttonRunClue

        buttonRunRest = self.builder.get_object("run_rest_button", self.mainwindow)
        buttonRunRest.config(command=self.runRestButton)
        self.buttonRunRest = buttonRunRest

        buttonRunNext = self.builder.get_object("run_next_button", self.mainwindow)
        buttonRunNext.config(command=self.runNextRoundButton)
        self.buttonRunNext = buttonRunNext

        buttonResetRun = self.builder.get_object("reset_run_button", self.mainwindow)
        buttonResetRun.config(command=self.resetRunButton)
        self.buttonResetRun = buttonResetRun

        buttonCancelRun = self.builder.get_object("cancel_run_button", self.mainwindow)
        buttonCancelRun.config(command=self.cancelRunButton)

        # Buttons to block while a run is in progress
        self.blockButtons = [buttonRunClue, buttonRunRest, buttonRunNext, buttonResetRun]

        buttonChangeInput = self.builder.get_object("input_file_button", self.mainwindow)
        buttonChangeInput.config(command=self.selectInputFile)

        buttonChangeDirectory = self.builder.get_object("change_directory_button", self.mainwindow)
        buttonChangeDirectory.config(command=self.selectOutputDirectory)

        buttonChangeClueclust = self.builder.get_object("clueclust_button", self.mainwindow)
        buttonChangeClueclust.config(command=self.selectClueclustLocation)

        self.textBox = self.builder.get_object("output_text", self.mainwindow)

        self.plotNotebook = self.builder.get_object("plots_notebook", self.mainwindow)

        self.builder.connect_callbacks(self)

        Logger.enableQueue()
        self.updateQueue() #Start the queue update loop

if __name__ == "__main__":
    app = ClueGui()
    app.run()

#!/usr/bin/python3
import pathlib
import sys
import threading

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

import cluerun
import clueutil
from clueui import ClueGuiUI
import clueuiui

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

class ClueGui(ClueGuiUI):

    #Mapping of round names to buttons in the UI
    roundsDict = {}
    #Active clue run
    clueRun = None

    """
        Show an error popup with the given title and message.
    """
    def __errorPopup(self, title, message):
        error_popup = tk.Toplevel(self.mainwindow)
        error_popup.title(title)
        error_popup.grab_set()
        error_popup.focus_force()
        error_popup.columnconfigure(0, weight=1)
        error_popup.rowconfigure(0, weight=1)
        error_popup.geometry("500x300")
        tk.Label(error_popup, text=message).grid(row=0, column=0, padx=10, pady=10)
        tk.Button(error_popup, text="OK", command=error_popup.destroy).grid(row=1, column=0, padx=10, pady=10)

    """
        Open a file dialog to choose a file and set the entry widget's value to the selected file path.
    """
    def __chooseFile(entryWidget, popupWindow):
        filePath = tk.filedialog.askopenfilename(
            parent=popupWindow,
            title="Select Features File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filePath:
            entryWidget.delete(0, tk.END)
            entryWidget.insert(0, filePath)

    """
        Build a popup window to configure a round.
    """
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

        #Popup window creation   
        popup = tk.Toplevel(self.mainwindow)
        popup.title(f"Configure {currentRoundName}")
        popup.grab_set()  # Makes it modal
        popup.focus_force()

        popup.columnconfigure(0, weight=1)
        popup.columnconfigure(1, weight=1)
        popup.rowconfigure(0, weight=1)

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

        # Epsilon, MinPts, Hyperplanes, Hashtables, KClusters, Threads, Standardize, Use Features selection for the config
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
        featuresFileEntry.insert(0, currentFeaturesFile)  # <-- Set initial value
        featuresFileButton = tk.Button(
            roundSettings,
            text="Browse",
            command=lambda: self.__chooseFile(featuresFileEntry, popup)
        )
        featuresFileButton.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # Cluster selection file selection
        tk.Label(roundSettings, text="Selection File:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        selectionFileEntry = tk.Entry(roundSettings)
        selectionFileEntry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        selectionFileEntry.insert(0, currentSelectionFile)  # <-- Set initial value
        selectionFileButton = tk.Button(
            roundSettings,
            text="Browse",
            command=lambda: self.__chooseFile(selectionFileEntry, popup)
        )
        selectionFileButton.grid(row=2, column=2, padx=5, pady=5, sticky="ew")

        # Callback when user clicks "OK"
        def on_ok():

            # Grab values from the UI elements
            name = roundnameVar.get() or f"Round {len(self.roundsDict) + 1}"
            featureFile = str(featuresFileEntry.get()) or None
            selectionFile = str(selectionFileEntry.get()) or None
            algorithm = cluerun.Algorithm(chosenAlgorithm.current() + 1) or cluerun.Algorithm(currentAlgorithm)
            distanceMetric = cluerun.DistanceMetric(chosenMetric.current() + 1) or cluerun.DistanceMetric(currentMetric)
            paramOptimization = cluerun.ParameterOptimizationLevel(paramOptLevel.current()) or cluerun.ParameterOptimizationLevel(currentParamOpt)
            epsilon = epsilonVar.get() or currentEpsilon
            minPts = minPtsVar.get() or currentMinPts
            hyperplanes = hyperplanesVar.get() or currentHyperplanes
            hashtables = hashtablesVar.get() or currentHashtables
            kClusters = kClustersVar.get() or currentKClusters
            threads = threadsVar.get() or currentThreads
            standardize = standardizeVar.get() or currentStandardize
            useFeatures = useFeaturesVar.get() or currentUseFeatures

            if currentRoundName not in self.roundsDict.keys():
                #If round by name does not exist already, create it
                if self.clueRun.getRound(roundnameVar.get()) is None:
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
                    
                    # Define the callback function for the round button click
                    def roundClick():
                        #Grab values from dictionary and print them
                        clueRound = self.clueRun.getRound(name)
                        clueConfig = clueRound.clueConfig
                        print(f"Round Name: {clueRound.roundName}")
                        print(f"Algorithm: {clueConfig.algorithm}")
                        print(f"Distance Metric: {clueConfig.distanceMetric}")
                        print(f"Parameter Optimization: {clueConfig.paramOptimization}")
                        print(f"Epsilon: {clueConfig.epsilon}")
                        print(f"MinPts: {clueConfig.minPts}")
                        print(f"Hyperplanes: {clueConfig.hyperplanes}")
                        print(f"Hashtables: {clueConfig.hashtables}")
                        print(f"KClusters: {clueConfig.kClusters}")
                        print(f"Threads: {clueConfig.threads}")
                        print(f"Standardize: {clueConfig.standardize}")
                        print(f"Use Features: {clueConfig.useFeatures}")
                        print(f"Features File: {clueRound.featureSelectionFile}")
                        print(f"Selection File: {clueRound.clusterSelectionFile}")

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

                    # Create the new round button in rounds_frame
                    new_round_button = tk.Button(
                        self.rounds_frame,
                        text=f"{name}", 
                        command=roundClick
                    )
                    self.roundsDict[name] = new_round_button
                    _, current_rows = self.rounds_frame.grid_size()
                    new_round_button.grid(row=current_rows, column=0, sticky="ew", pady=5)
                    self.rounds_frame.grid_columnconfigure(0, weight=1)
                else:
                    # If the name is already in the roundsDict, show an error
                    self.__errorPopup("Round Exists", f"Round '{name}' already exists.")

            # If round already exists, update it
            else:
                #If the name is already in the roundsDict, show an error
                if name != currentRoundName:
                    self.__errorPopup("Not Implemented", f"Namechange not implemented yet. Please delete the round and create a new one with the desired name.")
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

        tk.Button(popup, text="OK", command=on_ok).grid(row=2, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=2, column=1, padx=5, pady=10)

    """
        Add a new round to the current run while also adding it to the UI using the builder.
    """
    def addRound(self):
        if self.clueRun is None:
            self.__errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        
        #Create a popup dialog to configure the round
        self.buildRoundPopup()

    """
        Delete a round from the current run and remove it from the UI.
    """
    def deleteRound(self):
        if self.clueRun is None:
            self.__errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        
        #Create a popup dialog to select round name deletion and then delete that round
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Delete Round")
        popup.grab_set()  # Makes it modal
        popup.focus_force()
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

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
            self.__errorPopup("No rounds", "No rounds available to delete.")
            popup.destroy()  # Close the popup if no rounds are available
            return
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get()))

        #On clicking the delete button, delete the round from the run and remove it from the UI
        def on_delete():
            roundName = roundNameVar.get()
            if roundName in self.roundsDict:
                self.clueRun.removeRound(roundName)
                self.roundsDict[roundName].destroy()
                del self.roundsDict[roundName]
                print("Round Deleted", f"Round '{roundName}' has been deleted.")
            else:
                self.__errorPopup("Round Not Found", f"Round '{roundName}' does not exist.")
            popup.destroy()  # Close the popup

        tk.Button(popup, text="Delete", command=on_delete).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)

    """
        Create a popup to selet a round to move up in the list of rounds to be ran.
        Also moves the button in the UI to reflect the change.
    """
    def moveRoundUp(self):
        if self.clueRun is None:
            self.__errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        if not list(self.roundsDict):
            self.__errorPopup("No Rounds", "No rounds available to move up.")
            return

        popup = tk.Toplevel(self.mainwindow)
        popup.title("Move Round Up")
        popup.grab_set() 
        popup.focus_force()
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1)

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

        # Define the callback function for moving the round up
        def on_move_up():
            try:
                roundName = roundNameVar.get()
                current_index = self.clueRun.getRoundIndex(roundName)
                if current_index > 0:
                    # Swap the current round with the one above it
                    above_round_name = list(self.clueRun.rounds)[current_index - 1].roundName
                    self.clueRun.moveRoundUp(roundName)
                    
                    # Update the UI
                    self.roundsDict[roundName].grid(row=current_index - 1, column=0, sticky="ew", pady=5)
                    self.roundsDict[above_round_name].grid(row=current_index, column=0, sticky="ew", pady=5)
                    print("Round Moved Up: ", f"Round '{roundName}' has been moved up.")
                else:
                    self.__errorPopup("Already at Top: ", f"Round '{roundName}' is already at the top.")
            except clueutil.ClueException as e:
                self.__errorPopup("Error Moving Round", str(e))
            popup.destroy()  # Close the popup
        
        tk.Button(popup, text="Move Up", command=on_move_up).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)
        
    """
        Create a popup to select a round to move down in the list of rounds to be ran.
        Also moves the button in the UI to reflect the change.
    """
    def moveRoundDown(self):
        if self.clueRun is None:
            self.__errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        
        #Move a round down in the list of rounds to be ran, if it is not already at the bottom.
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Move Round Down")
        popup.grab_set() 
        popup.focus_force()
        popup.columnconfigure(0, weight=1)
        popup.rowconfigure(0, weight=1) 

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
            self.__errorPopup("No Rounds", "No rounds available to move down.")
            popup.destroy()
            return
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get())) 

        # Define the callback function for moving the round down
        def on_move_down():
            try:
                roundName = roundNameVar.get()
                current_index = self.clueRun.getRoundIndex(roundName)
                if current_index == -1:
                    self.__errorPopup("Round Not Found: ", f"Round '{roundName}' does not exist.")
                    return
                if current_index < len(self.clueRun.rounds) - 1:
                    below_round_name = list(self.clueRun.rounds)[current_index + 1].roundName
                    self.clueRun.moveRoundDown(roundName)
                    
                    # Update the UI
                    self.roundsDict[roundName].grid(row=current_index + 1, column=0, sticky="ew", pady=5)
                    self.roundsDict[below_round_name].grid(row=current_index, column=0, sticky="ew", pady=5)
                    print("Round Moved Down: ", f"Round '{roundName}' has been moved down.")
                else:
                    self.__errorPopup("Already at Bottom: ", f"Round '{roundName}' is already at the bottom.")
            except clueutil.ClueException as e:
                self.__errorPopup("Error Moving Round", str(e))
            popup.destroy()

        tk.Button(popup, text="Move Down", command=on_move_down).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)
    
    """
        Clear all rounds from the current run and update the UI.
    """
    def clearRounds(self):
        for _, button in self.roundsDict.items():
            button.destroy()
        self.roundsDict.clear()

    """
        Change the label of the run in the UI.
    """
    def changeRunNameLabel(self, newName):
        if self.clueRun is not None:
            self.builder.get_object("run_name_label", self.mainwindow).config(text=newName)
        else:
            self.__errorPopup("No Run Defined", "Please create a run first.")

    """
        Create a new run and clear any existing rounds. Name is specified in a popup dialog.
    """
    def newRun(self):
        self.clearRounds()
        
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

        def on_ok():
            newName = runNameVar.get() or "Run1"
            input_file = self.builder.get_object("input_file_entry", self.mainwindow).get()
            directory = self.builder.get_object("directory_entry", self.mainwindow).get()
            clueclust_location = self.builder.get_object("clueclust_entry", self.mainwindow).get()
            self.clueRun = cluerun.ClueRun(
                newName,
                str(input_file),
                str(pathlib.Path(directory)),
                CLUECLUST=str(clueclust_location)
            )
            self.changeRunNameLabel(newName)
            print("New Run Created:", f"Run '{newName}' has been created.")
            popup.destroy()

        tk.Button(popup, text="OK", command=on_ok).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)

    """
        Create a tab in the notebook for displaying plots.
    """
    def createPlotTab(self, notebook, fig, name):
        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text=name)
        
        fig_canvas = FigureCanvasTkAgg(fig, master=plot_tab) 
        fig_canvas.draw() 
        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True) 

        toolbar = NavigationToolbar2Tk(fig_canvas, plot_tab)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill="both")

    """
        Build the graphs for the given round.
    """
    def buildGraphs(self, round, baseFile, baseFeaturesFile, outputDirectory, directOutput):
        try:
            clueutil.ClueGraphing.generateGraphs(
                round,
                baseFile,
                baseFeaturesFile,
                outputDirectory=outputDirectory,
                directOutput=directOutput
            )
        except clueutil.ClueException as e:
            self.__errorPopup("Graph Generation Error", str(e))

        for tab in self.plotNotebook.tabs():
            self.plotNotebook.forget(tab)

        for name, graph in clueutil.ClueGraphing.graphs.items():
            fig = graph
            if fig is not None:
                # Create a new tab in the notebook for each figure
                self.createPlotTab(self.plotNotebook, fig, name)
            else:
                self.__errorPopup("No figure found for graph: ", name)
    """
        Execute the current CLUE run.
    """
    def runClue(self):
        print("Running CLUE", "Starting the CLUE run...")
        self.clueRun.run()
        self.mainwindow.after(0, self.buildGraphs,
                          self.clueRun.rounds[len(self.clueRun.rounds) - 1],
                          self.clueRun.baseFile,
                          self.clueRun.baseFeaturesFile,
                          str(self.clueRun.targetRunDirectory) + "/" + self.clueRun.outputDirectory,
                          True)
        print("Run Complete", "The CLUE run has completed successfully.")

    """
        Method called when the run button is clicked. Starts the CLUE run in a separate thread.
    """
    def runClueButton(self):
        if self.clueRun is None:
            self.__errorPopup("No run defined", "No run defined, Please create a run first.")
            return
        #Set global run settings before run
        input_file = self.builder.get_object("input_file_entry", self.mainwindow).get()
        output_directory = self.builder.get_object("directory_entry", self.mainwindow).get()
        clueclust_location = self.builder.get_object("clueclust_entry", self.mainwindow).get()
        if not input_file or not output_directory or not clueclust_location:
            self.__errorPopup("Missing Information", "Please ensure all fields are filled out before running CLUE.")
            return
        else:
            self.clueRun.baseFile = str(input_file)
            self.clueRun.updateTargetRunDirectory(str(output_directory))
            self.clueRun.CLUECLUST = str(clueclust_location)
        threading.Thread(target=self.runClue).start()

    """
        Select the input file for the CLUE run.
    """
    def selectInputFile(self):
        filePath = filedialog.askopenfilename(
            parent=self.mainwindow,
            title="Select Input File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filePath:
            self.builder.get_object("input_file_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("input_file_entry", self.mainwindow).insert(0, filePath)

    """
        Select the output directory for the CLUE run.
    """
    def selectOutputDirectory(self):
        directoryPath = filedialog.askdirectory(
            parent=self.mainwindow,
            title="Select Output Directory"
        )
        if directoryPath:
            self.builder.get_object("directory_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("directory_entry", self.mainwindow).insert(0, str(directoryPath))

    """
        Select the location of the clueclust executable.
    """
    def selectClueclustLocation(self):
        filePath = filedialog.askopenfilename(
            parent=self.mainwindow,
            title="Select Clueclust Jar",
            filetypes=[("Jar files", "*.jar"), ("All files", "*.*")]
        )
        if filePath:
            self.builder.get_object("clueclust_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("clueclust_entry", self.mainwindow).insert(0, filePath)

    """
        Helper class to redirect stdout to a text widget in the GUI.
    """
    class TextRedirector:
        def __init__(self, text_widget):
            self.text_widget = text_widget

        def write(self, text):
            self.text_widget.insert(tk.END, text)
            self.text_widget.see(tk.END)  # Auto-scroll

        def flush(self):
            pass  # Needed for Python's stdout

    def __init__(self, master=None):
        super().__init__(master)
        self.clueRun = None
        self.builder.add_from_file(clueuiui.PROJECT_UI)

        self.mainwindow = self.builder.get_object("base_frame", master)
        self.rounds_frame = self.builder.get_object("rounds_list_frame")

        menu_command_add_round = self.builder.get_object("new_round_button", self.mainwindow)
        menu_command_add_round.config(command=self.addRound)

        menu_command_delete_round = self.builder.get_object("delete_round_button", self.mainwindow)
        menu_command_delete_round.config(command=self.deleteRound)

        menu_command_move_round_up = self.builder.get_object("move_round_up_button", self.mainwindow)
        menu_command_move_round_up.config(command=self.moveRoundUp)

        menu_command_move_round_down = self.builder.get_object("move_round_down_button", self.mainwindow)
        menu_command_move_round_down.config(command=self.moveRoundDown)

        button_run_clue = self.builder.get_object("run_button", self.mainwindow)
        button_run_clue.config(command=self.runClueButton)

        button_change_input = self.builder.get_object("input_file_button", self.mainwindow)
        button_change_input.config(command=self.selectInputFile)

        button_change_directory = self.builder.get_object("change_directory_button", self.mainwindow)
        button_change_directory.config(command=self.selectOutputDirectory)

        button_change_clueclust = self.builder.get_object("clueclust_button", self.mainwindow)
        button_change_clueclust.config(command=self.selectClueclustLocation)

        text_box = self.builder.get_object("output_text", self.mainwindow)
        sys.stdout = self.TextRedirector(text_box)

        self.plotNotebook = self.builder.get_object("plots_notebook", self.mainwindow)

        self.builder.connect_callbacks(self)

if __name__ == "__main__":
    app = ClueGui()
    app.run()

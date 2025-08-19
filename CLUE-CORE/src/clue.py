#!/usr/bin/python3
import pathlib
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import pygubu
import cluerun
import clueutil
from clueui import ClueGuiUI
import clueuiui


class ClueGui(ClueGuiUI):

    roundsDict = {}

    def chooseFile(entryWidget, popupWindow):
        filePath = tk.filedialog.askopenfilename(
            parent=popupWindow,
            title="Select Features File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filePath:
            entryWidget.delete(0, tk.END)
            entryWidget.insert(0, filePath)

    def buildPopup(self, 
                   currentAlgorithm=0, 
                   currentMetric=0, 
                   currentParamOpt=0,
                   currentEpsilon=1.0,
                   currentMinPts=5,
                   currentHyperplanes=5,
                   currentHashtables=5,
                   currentKClusters=5,
                   currentThreads=1,
                   currentStandardize=True,
                   currentUseFeatures=True,
                   currentFeaturesFile="",
                   currentSelectionFile="",
                   currentRoundName="New Round",
                   ):
                        
        popup = tk.Toplevel(self.mainwindow)
        popup.title(f"Configure {currentRoundName}")
        popup.grab_set()  # Makes it modal
        popup.focus_force()

        popup.columnconfigure(0, weight=1)
        popup.columnconfigure(1, weight=1)
        popup.rowconfigure(0, weight=1)

        roundConfig = tk.LabelFrame(popup, text="Round Configs", padx=10, pady=10)
        roundConfig.grid(row=0, column=0, padx=10, pady=5, stick="nsew")

        tk.Label(roundConfig, text="Algorithm").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        chosenAlgorithm = ttk.Combobox(
            roundConfig,
            values=["DBSCAN", "IP.LSH.DBSCAN", "KMeans"],
            state="readonly"
        )
        chosenAlgorithm.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        chosenAlgorithm.current(currentAlgorithm)

        tk.Label(roundConfig, text="Distance Metric").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        chosenMetric = ttk.Combobox(
            roundConfig,
            values=["Euclidean", "Angular", "DTW"],
            state="readonly",
        )
        chosenMetric.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        chosenMetric.current(currentMetric)

        tk.Label(roundConfig, text="Parameter Optimizer").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        paramOptLevel = ttk.Combobox(
            roundConfig,
            values=["None", "K-Distance", "Grid Search"],
            state="readonly",
        )
        paramOptLevel.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        paramOptLevel.current(currentParamOpt)

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
        
        roundSettings = tk.LabelFrame(popup, text="Round Settings", padx=10, pady=10)
        roundSettings.grid(row=0, column=1, padx=10, pady=5, stick="nsew")

        tk.Label(roundSettings, text="Round Name:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        roundnameVar = tk.StringVar(roundSettings, value=currentRoundName)
        tk.Entry(roundSettings, textvariable=roundnameVar).grid(row=0, column=1, padx=5, pady=5)
    
        # Add another file selector dialog box to select the features file following the same pattern, with the default value set to currentFeaturesFile
        tk.Label(roundSettings, text="Features File:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        featuresFileEntry = tk.Entry(roundSettings)
        featuresFileEntry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        featuresFileEntry.insert(0, currentFeaturesFile)  # <-- Set initial value
        featuresFileButton = tk.Button(
            roundSettings,
            text="Browse",
            command=lambda: self.chooseFile(featuresFileEntry, popup)
        )
        featuresFileButton.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        #Add another file selection dialog box to select the selection file following the same pattern
        tk.Label(roundSettings, text="Selection File:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        selectionFileEntry = tk.Entry(roundSettings)
        selectionFileEntry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        selectionFileEntry.insert(0, currentSelectionFile)  # <-- Set initial value
        selectionFileButton = tk.Button(
            roundSettings,
            text="Browse",
            command=lambda: self.chooseFile(selectionFileEntry, popup)
        )
        selectionFileButton.grid(row=2, column=2, padx=5, pady=5, sticky="ew")

        # Callback when user clicks "OK"
        def on_ok():   
            if self.clueRun.getRound(roundnameVar.get()) is None:         
                name = roundnameVar.get() or f"Round {self.round_count}"
                featureFile = featuresFileEntry.get() or None
                selectionFile = selectionFileEntry.get() or None

                self.clueRun.buildRound(
                    name,
                    featureFile,
                    selectionFile,
                    cluerun.ClueConfig(
                        algorithm=cluerun.Algorithm(chosenAlgorithm.current() + 1),
                        distanceMetric=cluerun.DistanceMetric(chosenMetric.current() + 1),
                        paramOptimization=cluerun.ParameterOptimizationLevel(paramOptLevel.current()),
                        epsilon=epsilonVar.get(),
                        minPts=minPtsVar.get(),
                        hyperplanes=hyperplanesVar.get(),
                        hashtables=hashtablesVar.get(),
                        kClusters=kClustersVar.get(),
                        threads=threadsVar.get(),
                        standardize=standardizeVar.get(),
                        useFeatures=useFeaturesVar.get()
                    )
                )
                
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
                    #Open popup window with the round name and the values from the dictionary using self.popupWindow
                    self.buildPopup(
                        currentAlgorithm=clueConfig.algorithm.value - 1,
                        currentMetric=clueConfig.distanceMetric.value - 1,
                        currentParamOpt=clueConfig.paramOptimization.value - 1,
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
                self.clueRun.setRound(
                    roundnameVar.get(),
                    featuresFileEntry.get(),
                    selectionFileEntry.get(),
                    cluerun.ClueConfig(
                        algorithm=cluerun.Algorithm(chosenAlgorithm.current() + 1),
                        distanceMetric=cluerun.DistanceMetric(chosenMetric.current() + 1),
                        paramOptimization=cluerun.ParameterOptimizationLevel(paramOptLevel.current() + 1),
                        epsilon=epsilonVar.get(),
                        minPts=minPtsVar.get(),
                        hyperplanes=hyperplanesVar.get(),
                        hashtables=hashtablesVar.get(),
                        kClusters=kClustersVar.get(),
                        threads=threadsVar.get(),
                        standardize=standardizeVar.get(),
                        useFeatures=useFeaturesVar.get()
                    )
                )

            popup.destroy()  # Close the popup

        # OK and Cancel buttons
        tk.Button(popup, text="OK", command=on_ok).grid(row=2, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=2, column=1, padx=5, pady=10)

    def addRound(self):
        """
            Add a new round to the current run while also adding it to the UI using the builder and
            the ID: "rounds_frame". 
        """
        if self.run is None:
            clueutil.show_error("No run defined", "Please create a run first.")
            return
        
        self.buildPopup()

    def deleteRound(self):
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
            print("No Rounds", "No rounds available to delete.")
            popup.destroy()  # Close the popup if no rounds are available
            return
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get()))
        def on_delete():
            roundName = roundNameVar.get()
            if roundName in self.roundsDict:
                self.clueRun.removeRound(roundName)
                self.roundsDict[roundName].destroy()
                del self.roundsDict[roundName]
                print("Round Deleted", f"Round '{roundName}' has been deleted.")
            else:
                print("Round Not Found", f"Round '{roundName}' does not exist.")
            popup.destroy()  # Close the popup

        tk.Button(popup, text="Delete", command=on_delete).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)

    def moveRoundUp(self):
        #Move a round up in the list of rounds to be ran, if it is not already at the top.
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Move Round Up")
        popup.grab_set()  # Makes it modal
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
        if not roundNameMenu['values']:
            print("No Rounds", "No rounds available to move up.")
            popup.destroy()  # Close the popup if no rounds are available
            return
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get()))

        def on_move_up():
            for run in self.clueRun.rounds:
                print(run.roundName)
            roundName = roundNameVar.get()
            current_index = self.clueRun.getRoundIndex(roundName)
            if current_index == -1:
                print("Round Not Found: ", f"Round '{roundName}' does not exist.")
                return
            if current_index > 0:
                # Swap the current round with the one above it
                above_round_name = list(self.clueRun.rounds)[current_index - 1].roundName
                self.clueRun.moveRoundUp(roundName)
                
                # Update the UI
                self.roundsDict[roundName].grid(row=current_index - 1, column=0, sticky="ew", pady=5)
                self.roundsDict[above_round_name].grid(row=current_index, column=0, sticky="ew", pady=5)
                print("Round Moved Up: ", f"Round '{roundName}' has been moved up.")
            else:
                print("Already at Top: ", f"Round '{roundName}' is already at the top.")
            popup.destroy()  # Close the popup
        
        tk.Button(popup, text="Move Up", command=on_move_up).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)
        
    def moveRoundDown(self):
        #Move a round down in the list of rounds to be ran, if it is not already at the bottom.
        popup = tk.Toplevel(self.mainwindow)
        popup.title("Move Round Down")
        popup.grab_set()  # Makes it modal
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
            print("No Rounds", "No rounds available to move down.")
            popup.destroy()  # Close the popup if no rounds are available
            return
        roundNameMenu.current(0)
        roundNameMenu.bind("<<ComboboxSelected>>", lambda e: roundNameVar.set(roundNameMenu.get())) 

        def on_move_down():
            roundName = roundNameVar.get()
            current_index = self.clueRun.getRoundIndex(roundName)
            if current_index == -1:
                print("Round Not Found: ", f"Round '{roundName}' does not exist.")
                return
            if current_index < len(self.clueRun.rounds) - 1:
                # Swap the current round with the one below it
                below_round_name = list(self.clueRun.rounds)[current_index + 1].roundName
                self.clueRun.moveRoundDown(roundName)
                
                # Update the UI
                self.roundsDict[roundName].grid(row=current_index + 1, column=0, sticky="ew", pady=5)
                self.roundsDict[below_round_name].grid(row=current_index, column=0, sticky="ew", pady=5)
                print("Round Moved Down: ", f"Round '{roundName}' has been moved down.")
            else:
                print("Already at Bottom: ", f"Round '{roundName}' is already at the bottom.")
            popup.destroy()  # Close the popup

        tk.Button(popup, text="Move Down", command=on_move_down).grid(row=1, column=0, padx=5, pady=10)
        tk.Button(popup, text="Cancel", command=popup.destroy).grid(row=1, column=1, padx=5, pady=10)
    
    def newRun(self):
        self.clueRun = cluerun.ClueRun("run1", "input.csv", "testfiles")

    def runClue(self):
        """
            Run the CLUE algorithm on the current run.
        """
        if self.clueRun is None:
            print("No run defined", "Please create a run first.")
            return
        
        self.clueRun.run()
        print("Run Complete", "The CLUE run has completed successfully.")

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
            self.clueRun.targetRunDirectory = str(pathlib.Path(directoryPath))
            self.builder.get_object("directory_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("directory_entry", self.mainwindow).insert(0, str(directoryPath))

    def selectClueclustLocation(self):
        """
            Select the location of the clueclust executable.
        """
        filePath = filedialog.askopenfilename(
            parent=self.mainwindow,
            title="Select Clueclust Jar",
            filetypes=[("Jar files", "*.jar"), ("All files", "*.*")]
        )
        if filePath:
            self.clueRun.setCLUECLUST(str(filePath)) #TODO Fix so that this updates when creating rounds after
            self.builder.get_object("clueclust_entry", self.mainwindow).delete(0, tk.END)
            self.builder.get_object("clueclust_entry", self.mainwindow).insert(0, filePath)
        
    def __init__(self, master=None):
        super().__init__(master)
        self.newRun()
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
        button_run_clue.config(command=self.runClue)

        button_change_input = self.builder.get_object("input_file_button", self.mainwindow)
        button_change_input.config(command=self.selectInputFile)

        button_change_directory = self.builder.get_object("change_directory_button", self.mainwindow)
        button_change_directory.config(command=self.selectOutputDirectory)

        button_change_clueclust = self.builder.get_object("clueclust_button", self.mainwindow)
        button_change_clueclust.config(command=self.selectClueclustLocation)

        self.builder.connect_callbacks(self)

if __name__ == "__main__":
    app = ClueGui()
    app.run()

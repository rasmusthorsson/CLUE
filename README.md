# CLUE

Here is a short guide to how this implementation is structured and how to use it.

## CLUE-CLUST

CLUE-CLUST works as a seperate module as well as an integrated one. It offers functionality for clustering individual CSV files and outputs 
an output file containing the IDs of clustered individuals along with their associated cluster, and a metadata file containing information about
the clusters themselves. Furthermore, if the parameter optimizer is used other files associated with it will be generated. For a full understanding of
how CLUE-CLUST works, see the code itself and the options available. Some information is also available in the associated thesis document.

## CLUE-CORE

CLUE-CORE is the core software for easily running CLUE-CLUST in a round-based manner. CLUE-CORE works by constructing a 'run' which contains
a series of 'rounds'. Each round corresponds to one execution of CLUE-CLUST and subsequent rounds are filtered by earlier rounds output:

Raw Data -> Round 1 -> Round 2 -> Round 3

Such that Round 3 only contains data which was clustered in Round 1 and Round 2. Furthermore, each round can be configured to only filter
a selection of clusters. For example if one cluster is the only interesting cluster from Round 1, it is possible to select that round as the
only cluster getting passed through to Round 2. 

The input file to CLUE-CORE should be of CSV format, with an ID in the first column, and the values to be clustered in the subsequent columns.
The CSV file should also contain headers, the first column should be named "ID", the rest of the headers are less strict. An example synthetic 
file can be seen in the 'input.csv' file.

The intended interface is the clue.py UI interface. However, CLUE can be used using a Python notebook for example.

### Options and Settings

Here is a brief overview of options and settings available

#### Cluster Selection

This form of selection is done by writing a selection file and associating it with the round. The available options are as follows:

IN:Cluster1,Cluster2,...
NOTIN:...
OVER:...
UNDER:...

The exact way these operators interact can be found in clueutil.py. 

#### Feature Selection

Given an input file containing headers it is possible to select which 'features' are to be used for clustering each round. This can be specified in
a feature selection file which can then be associated with a round. For example, given a file with 8 dimensions:

ID,DIM1,DIM2,DIM3,DIM4,DIM5,DIM6,DIM7,DIM8
1,2.5,3.5, ...
...

it is possible to structure a features file as such:

ID,DIM1,DIM3,DIM5,DIM7

Which then only considers the above dimensions for the clustering. This is particularly useful when the 'use features' option is selected.

#### Algorithm and Distance Metric

Each round can use a variety of clustering algorithms and distance metrics. The possible options are:

Algorithms:
    DBSCAN
    IP.LSH.DBSCAN
    K-Means

Distance Metric:
    Euclidean
    Angular
    DTW

For a more indepth understanding of when these should be used, see the thesis. In short, DBSCAN is a slow but accurate density-based clustering algorithm,
IP.LSH.DBSCAN is extremely fast, but only approximate. K-Means is not density based and instead requires specifying how many clusters are expected to be found.

For distance metrics, Euclidean and Angular each work better depending on the dataset and it can be difficult to know which one is preferable without testing.
DTW is far slower and should be used primarily with IP.LSH.DBSCAN.

#### Parameter Optimizer

Built into the software is a parameter optimizer which can be used to establish parameters for DBSCAN and IP.LSH.DBSCAN. There are two levels to this parameter
optimizer:

K-Distance was originally introduced in the original DBSCAN paper. It primarily finds epsilon, while MinPts must be selected manually.

Grid-Search is our novel contribution, it uses IP.LSH.DBSCAN as a fast algorithm for evaluating hundreds of combinations and checks clustering metrics
for 'good' results. This tends to yield better results for a slightly longer runtime. This method also finds both epsilon and MinPts. There are some 
situations where the parameter optimizer does not result in any 'good' clusterings. In this case, try a different distance metric. Alternatively you will
need to explore the parameter combinations yourself.

#### Other Round Options

The other possible round options are:

Use Features: Selecting this instead clusters on derived features rather than the raw data itself. The features available can be found in the clueutil.py.

Standardize: This is commonly used in combination with Use Features. It standardizes the values so that the range is change to always be 0 - 1 for each
dimension.

#### Global Run Settings

For a run to be possible, certain things must be defined:

Input File: This must be defined as the raw csv file described earlier sections.

Output Directory: This should be a general output directory, the output will be automatically structured into runs and rounds.

CLUECLUST.jar: This is the Jar file of the CLUE-CLUST module, one of these exists in the 'external' directory. If new changes are made to
the CLUE-CLUST module, a new Jar file can be generated using the command:

mvn package

in the CLUE-CLUST directory (requires maven). A new Jar will then be generated (Two will be generated normally, the one WITH dependencies is the
one to be used.).

### UI Structure

The UI is split into different sections. In addition to the above-mentioned options and settings, the UI provides a round manager for adding, deleting, moving,
and regressing to previous rounds. Moving or deleting rounds will reset the run, adding a round or regressing to a previous round will not. There is also
an output text box where the progress of the run is written to. Finally, there is a plot section where graphs are plotted per round, this is to allow 
instant understanding of the clustering generated by a round.

If a round results in no clusters it is not completed and regressed. 

Runs can be saved and loaded via XML format in the top menu.

#### Graph and Feature Additions

Currently, a selection of output graphs and base features are implemented. They use the same but seperate systems:

Each graph/feature is defined as a python function (in ClueGraphing and FeatureExtractor respectively, both in clueutil.py), these functions
are then associated with a name in the python directory in the same class. After this is done, they should appear as all other graphs/features do.
This is an easy way to add new graphs/features when needed. 

### Running Clue

Clue is primarily run via the UI (clue.py). Below is a list of requiremements to run clue:

Java Runtime Environment

Python 3

Python Libraries:
pandas
matplotlib
numpy
pygubu
scikit-learn

With these installed, just run:

python3 clue.py

### Example 

Unfortunately due to time-constraints, runs are not able to be shared properly. Therefore I will lay out instructions to construct an example run here 
rather than share one directly:

1. Start the CLUE UI
2. In the topleft corner under the menu 'Runs' select 'New Run' and type in example_run as the name
3. Create a 'Demo' directory in the 'CLUE' directory
4. In the UI, set the input file to be the 'input.csv' file in the 'CLUE' directory, change the directory to be the newly created 'Demo' directory,
and set the CLUECLUST.jar file to be the one found in 'CLUE-CORE/src/external/'
5. Create a new round by clicking 'New Round', name this round 'KMeans_1' and select the algorithm to be KMeans.
6. Click Run Next and check the results, multiple files should be created in the 'Demo' directory and you should be able to see graphs in the plot tab.
7. Create a new file in the 'Demo' directory named 'example_run_dbscan_selection.csv', in this new file, write and save: IN:0,4
8. Add another round, name this round 'DBSCAN_1' and select DBSCAN as the algorithm. Also select the distance metric to be Angular and select the
'Selection File' to be the newly created 'example_run_dbscan_selection.csv' file. Finally, also select the parameter optimization level to be 'Grid Search'
9. Observe the results: In many cases the results will be one cluster that is the joined curve of cluster 0 and 4 from the previous round (You can open
the 'Mean Raw Data.png' file from the 'Demo/example_run/KMeans_1/' directory and compare to the clusters)
10. Try changing the 'example_run_dbscan_selection.csv' file to try different cluster combinations, you can rerun the DBSCAN round only be clicking
'Previous Round' once. Note that having ran the Grid Search once, you can now click on the DBSCAN round button and change the parameter optimization level
to be 'None'. This means you do not need to rerun the grid search each time you try a new cluster combination.
11. Try primarily with 2 cluster combinations, if you cannot get the DBSCAN round to correctly identify the two clusters (i.e, it only finds 1 cluster with 
2000 points rather than 2 clusters with 1000 points each), then try to manually lower the epsilon slightly (When I tried I had success with 0.037, but it
will depend on cluster combination). Unfortunately the parameter optimizer is not as accurate as I would have liked, this will likely
be remedied in later versions if possible.

Beyond this you can try adding more rounds if you would like, perhaps if you are finding a lot of noise in the DBSCAN round you can try to map it with another
KMeans round and selecting only the -1 cluster (noise cluster).
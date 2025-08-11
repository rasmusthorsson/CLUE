package com.clueclust.util;

import org.apache.commons.cli.*;

/**
 * Parameters class for handling command-line arguments
 */
public class Parameters {
    private String fileName = "";
    private String baselineFileName = "";
    private String targetOutputDirectory = "";
    private String outputFileName = "";
    private String metadataFileName = "";
    private int numberOfHashTables = 10;
    private int numberOfHyperplanesPerTable = 10;
    private int numberOfThreads = Runtime.getRuntime().availableProcessors();
    private String command = "lshdbscan"; // Default command
    private boolean adaptiveSearch = false;
    private int k = 10; // Default number of clusters for K-Means
    private int maxIterations = 100; // Default max iterations for K-Means
    private boolean standardizeFeatures = false; // Standardization for feature-based clustering

    /**
     * Parse command line arguments
     */
    public static Parameters parseArguments(String[] args) {
        Parameters params = new Parameters();

        Options options = new Options();

        // Add options similar to the C++ program
        options.addOption(Option.builder("a")
                .desc("Use angular distance metric")
                .build());

        options.addOption(Option.builder("dtw")
                .desc("Use Dynamic Time Warping distance metric")
                .build());

        options.addOption(Option.builder("window")
                .desc("Window size for DTW calculation (Sakoe-Chiba band, default: 10)")
                .hasArg()
                .argName("window")
                .build());

        options.addOption(Option.builder("nolb")
                .desc("Disable LB_Keogh lower bound for DTW")
                .build());

        options.addOption(Option.builder("m")
                .desc("Minimum points for core bucket")
                .hasArg()
                .argName("minPts")
                .build());

        options.addOption(Option.builder("e")
                .desc("Epsilon value for neighborhood")
                .hasArg()
                .argName("epsilon")
                .build());

        options.addOption(Option.builder("t")
                .desc("Number of threads")
                .hasArg()
                .argName("threads")
                .build());

        options.addOption(Option.builder("f")
                .desc("Input file name")
                .hasArg()
                .argName("fileName")
                .required()
                .build());

        options.addOption(Option.builder("L")
                .desc("Number of hash tables")
                .hasArg()
                .argName("hashTables")
                .build());

        options.addOption(Option.builder("M")
                .desc("Number of hyperplanes per hash table")
                .hasArg()
                .argName("hyperplanes")
                .build());

        options.addOption(Option.builder("b")
                .desc("Baseline clustering file for comparison")
                .hasArg()
                .argName("baselineFile")
                .build());
        options.addOption(Option.builder("E")
                .desc("Epsilons to use as bases for param-opt (disables k-distance)")
                .hasArg()
                .argName("epsilons")
                .build());

        options.addOption(Option.builder("c")
                .desc("Command: lshdbscan, vanilla, vanillalsh, parameter")
                .hasArg()
                .argName("command")
                .build());
        options.addOption(Option.builder("adaptive")
                .desc("Use adaptive parameter search instead of grid search")
                .build());
        // Add K-Means specific options
        options.addOption(Option.builder("k")
                .desc("Number of clusters for K-Means (default: 10)")
                .hasArg()
                .argName("clusters")
                .build());

        options.addOption(Option.builder("o")
                .desc("Parameter Optimization level (default 2)")
                .hasArg()
                .argName("paramoptlevel")
                .build());
        options.addOption(Option.builder("i")
                .desc("Maximum iterations for K-Means (default: 100)")
                .hasArg()
                .argName("iterations")
                .build());
        options.addOption(Option.builder("standardize")
                .desc("Standardize features before clustering (z-score)")
                .build());
        options.addOption(Option.builder("d")
                .desc("Specify target output directory")
                .hasArg()
                .argName("Directory")
                .build());
        options.addOption(Option.builder("output")
                .desc("Specify output filename")
                .hasArg()
                .argName("outputFile")
                .build());
        options.addOption(Option.builder("metadata")
                .desc("Specify metadata filename")
                .hasArg()
                .argName("metadataFile")
                .build());

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();

        try {
            CommandLine cmd = parser.parse(options, args);

            if (cmd.hasOption("a")) {
                Globals.setMetric(Globals.Metric.ANGULAR);
            }

            // Add this to the parse block in Parameters.java
            if (cmd.hasOption("dtw")) {
                Globals.setMetric(Globals.Metric.DTW);
            }

            if (cmd.hasOption("dtw-window")) {
                Globals.DTW_WINDOW_SIZE = Integer.parseInt(cmd.getOptionValue("dtw-window"));
            }

            if (cmd.hasOption("o")) {
                Globals.PARAM_OPT_LEVEL = Integer.parseInt(cmd.getOptionValue("o"));
            }

            if (cmd.hasOption("no-lb")) {
                Globals.USE_LB_KEOGH = false;
            }
            if (cmd.hasOption("E")) {
                String[] epsilons = cmd.getOptionValue("E").split(",");
                for (String e : epsilons) {
                    Globals.PARAM_OPT_EPSILONS.add(Double.parseDouble(e));
                }
            }

            if (cmd.hasOption("m")) {
                Globals.MIN_PTS = Integer.parseInt(cmd.getOptionValue("m"));
            }

            if (cmd.hasOption("e")) {
                Globals.EPSILON_ORIGINAL = Double.parseDouble(cmd.getOptionValue("e"));
                if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
                    Globals.EPSILON = Globals.EPSILON_ORIGINAL * Globals.EPSILON_ORIGINAL;
                } else {
                    Globals.EPSILON = Globals.EPSILON_ORIGINAL;
                }
            }

            if (cmd.hasOption("t")) {
                params.numberOfThreads = Integer.parseInt(cmd.getOptionValue("t"));
            }

            if (cmd.hasOption("f")) {
                params.fileName = cmd.getOptionValue("f");
            }

            if (cmd.hasOption("L")) {
                params.numberOfHashTables = Integer.parseInt(cmd.getOptionValue("L"));
            }

            if (cmd.hasOption("M")) {
                params.numberOfHyperplanesPerTable = Integer.parseInt(cmd.getOptionValue("M"));
            }

            if (cmd.hasOption("b")) {
                params.baselineFileName = cmd.getOptionValue("b");
            }

            if (cmd.hasOption("c")) {
                params.command = cmd.getOptionValue("c");
            }
            if (cmd.hasOption("adaptive")) {
                params.setAdaptiveSearch(true);
            }
            // Parse K-Means specific parameters
            if (cmd.hasOption("k")) {
                params.k = Integer.parseInt(cmd.getOptionValue("k"));
            }

            if (cmd.hasOption("i")) {
                params.maxIterations = Integer.parseInt(cmd.getOptionValue("i"));
            }

            if (cmd.hasOption("standardize")) {
                params.setStandardizeFeatures(true);
            }

            if (cmd.hasOption("d")) {
                params.setTargetOutputDirectory(cmd.getOptionValue("d"));
            }
            if (cmd.hasOption("output")) {
                params.setOutputFileName(cmd.getOptionValue("output"));
            }
            if (cmd.hasOption("metadata")) {
                params.setMetadataFileName(cmd.getOptionValue("metadata"));
            }


        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("LSHDBSCAN", options);
            System.exit(1);
        }

        return params;
    }

    // Getters and setters
    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public String getTargetOutputDirectory() {
        return targetOutputDirectory;
    }

    public void setTargetOutputDirectory(String targetOutputDirectory) {
        this.targetOutputDirectory = targetOutputDirectory;
    }

    public String getOutputFileName() {
        return outputFileName;
    }

    public void setOutputFileName(String outputFileName) {
        this.outputFileName = outputFileName;
    }

    public String getMetadataFileName() {
        return metadataFileName;
    }

    public void setMetadataFileName(String metadataFileName) {
        this.metadataFileName = metadataFileName;
    }

    public String getBaselineFileName() {
        return baselineFileName;
    }

    public void setBaselineFileName(String baselineFileName) {
        this.baselineFileName = baselineFileName;
    }

    public int getNumberOfHashTables() {
        return numberOfHashTables;
    }

    public void setNumberOfHashTables(int numberOfHashTables) {
        this.numberOfHashTables = numberOfHashTables;
    }

    public int getNumberOfHyperplanesPerTable() {
        return numberOfHyperplanesPerTable;
    }

    public void setNumberOfHyperplanesPerTable(int numberOfHyperplanesPerTable) {
        this.numberOfHyperplanesPerTable = numberOfHyperplanesPerTable;
    }

    public int getNumberOfThreads() {
        return numberOfThreads;
    }

    public void setNumberOfThreads(int numberOfThreads) {
        this.numberOfThreads = numberOfThreads;
    }

    public String getCommand() {
        return command;
    }

    public void setCommand(String command) {
        this.command = command;
    }

    public boolean isAdaptiveSearch() {
        return adaptiveSearch;
    }

    public void setAdaptiveSearch(boolean adaptiveSearch) {
        this.adaptiveSearch = adaptiveSearch;
    }

    public int getK() {
        return k;
    }

    public void setK(int k) {
        this.k = k;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public boolean shouldStandardizeFeatures() {
        return standardizeFeatures;
    }

    public void setStandardizeFeatures(boolean standardizeFeatures) {
        this.standardizeFeatures = standardizeFeatures;
    }
}
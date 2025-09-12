package com.clueclust.runner;

import com.clueclust.core.Dataset;
import com.clueclust.util.KDistancePlotter;
import com.clueclust.util.OptimalParameterFinder;
import com.clueclust.util.Parameters;
import com.clueclust.util.Globals;

import java.io.IOException;

/**
 * Runner for finding optimal DBSCAN parameters without ground truth
 * Based on techniques from "DBSCAN Revisited, Revisited" paper
 */
public class OptimalParameterRunner implements Runner {

    @Override
    public void run(Parameters params) {
        try {
            System.out.println("=================================================");
            System.out.println("DBSCAN Parameter Optimization");
            System.out.println("=================================================");

            // Read dataset
            Dataset dataset = new Dataset();
            dataset.setHasHeader(false);
            dataset.setDelimiter(",");

            long startTime = System.nanoTime();
            dataset.readData(params.getFileName());
            long endTime = System.nanoTime();
            System.out.println("Reading data took: " + ((endTime - startTime) / 1_000_000_000.0) + " seconds");
            System.out.println("Dataset dimensions: " + dataset.getNumberOfDimensions());
            System.out.println("Number of points: " + dataset.getPoints().size());

            // Normalize data if using angular metric
            if (Globals.METRIC == Globals.Metric.ANGULAR) {
                dataset.normalizeData();
            }
            // Standardize data if flag is set
            else if (params.shouldStandardizeFeatures()) {
                dataset.standardizeData();
            }

            // Create the optimal parameter finder
            OptimalParameterFinder parameterFinder = new OptimalParameterFinder(dataset);

            // Calculate k-distance plots for various k values
            System.out.println("\n=== Step 1: Finding base epsilon ===");
            if (Globals.PARAM_OPT_EPSILONS.size() < 1) {
                System.out.println("\nNo base epsilon values supplied, calculating k-distance plots...");
                parameterFinder.calculateKDistancePlots();

                // Save the k-distance plots
                String kDistanceCsvFile = params.getFileName() + "_k_distances.csv";
                String kDistancePngFile = params.getFileName() + "_k_distances.png";

                if (params.getTargetOutputDirectory() != "") {
                    kDistanceCsvFile = params.getTargetOutputDirectory() + "k_distances.csv";
                    kDistancePngFile = params.getTargetOutputDirectory() + "k_distances.png";
                }
                parameterFinder.saveKDistancePlots(kDistanceCsvFile);   
                // Generate visualization of the k-distance plots
                KDistancePlotter.createKDistancePlot(
                        kDistanceCsvFile,
                        kDistancePngFile,
                        "K-Distance Plot for " + params.getFileName());
                System.out.println("K-distance plots saved to: " + kDistanceCsvFile + " and " + kDistancePngFile);
            } else {
                System.out.println("\nBase epsilon supplied, skipping k-distance generation");
                System.out.println("\nBase epsilon values: ");
                for (double eps : Globals.PARAM_OPT_EPSILONS) {
                    System.out.println(Double.toString(eps));
                }
            }

            boolean useAdaptiveSearch = params.isAdaptiveSearch();
            if (Globals.PARAM_OPT_LEVEL > 1) {
                // Run parameter exploration
                System.out.println("\n=== Step 2: Running parameter exploration ===");
                if (useAdaptiveSearch) {
                    System.out.println("Using adaptive search strategy");
                } else {
                    System.out.println("Using exhaustive grid search strategy");
                }
                parameterFinder.runParameterExploration(params.getNumberOfThreads(), useAdaptiveSearch);
                // Save the results
                String paramResultsFD = params.getFileName() + "_parameter_results.csv";
                String optParamsFD = params.getFileName().substring(0, params.getFileName().length() - 4) + "_optimal_params.csv";
                if (params.getTargetOutputDirectory() != "") {
                        paramResultsFD = params.getTargetOutputDirectory() + "parameter_results.csv";
                        optParamsFD = params.getTargetOutputDirectory() + "optimal_params.csv";
                }
                parameterFinder.saveResults(paramResultsFD);
                parameterFinder.saveSortedParametersAsCSV(optParamsFD);
                
                System.out.println("\nParameter optimization completed successfully.");
                System.out.println("Results saved to: " + paramResultsFD);
                System.out.println("Ranked optimal parameters saved to: " + optParamsFD);
            } else {   
                parameterFinder.runParameterKDistanceFinder();

                // Save the results
                String paramResultsFD = params.getFileName() + "_parameter_results.csv";
                String optParamsFD = params.getFileName().substring(0, params.getFileName().length() - 4) + "_optimal_params.csv";
                if (params.getTargetOutputDirectory() != "") {
                        paramResultsFD = params.getTargetOutputDirectory() + "parameter_results.csv";
                        optParamsFD = params.getTargetOutputDirectory() + "optimal_params.csv";
                }
                parameterFinder.saveResults(paramResultsFD);
                parameterFinder.saveSortedParametersAsCSV(optParamsFD);

                System.out.println("\nK-Distance parameter optimization completed successfully.");
                System.out.println("Results saved to: " + paramResultsFD);
                System.out.println("Ranked optimal parameters saved to: " + optParamsFD);
            }

            System.out.println("\nUse the recommended epsilon and minPts values with the following command:");
            System.out.println("java -jar clueclust.jar -c clueclust -f " + params.getFileName() +
                    " -e <recommended-epsilon> -m <recommended-minPts> -L " +
                    params.getNumberOfHashTables() + " -M " +
                    params.getNumberOfHyperplanesPerTable() + " -t " +
                    params.getNumberOfThreads());

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
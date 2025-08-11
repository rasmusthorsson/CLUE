package com.clueclust.runner;

import com.clueclust.algorithm.VanillaDBSCANLSH;
import com.clueclust.core.Dataset;
import com.clueclust.util.Parameters;
import com.clueclust.util.ClusteringAccuracy;
import com.clueclust.util.Globals;

import java.io.File;
import java.io.IOException;

/**
 * Runner for the vanilla DBSCAN with LSH algorithm
 */
public class VanillaDBSCANLSHRunner implements Runner {

    @Override
    public void run(Parameters params) {
        try {
            // Read dataset
            Dataset dataset = new Dataset();
            dataset.setHasHeader(true);
            dataset.setDelimiter(",");
            long startTime = System.nanoTime();
            dataset.readData(params.getFileName());
            long endTime = System.nanoTime();
            System.out.println("Reading data took: " + ((endTime - startTime) / 1_000_000_000.0) + " seconds");

            // Normalize data if using angular metric
            if (Globals.METRIC == Globals.Metric.ANGULAR) {
                dataset.normalizeData();
            }
            // Standardize data if flag is set
            else if (params.shouldStandardizeFeatures()) {
                dataset.standardizeData();
            }

            // Create and run clustering algorithm
            VanillaDBSCANLSH vanillaDBSCANLSH = new VanillaDBSCANLSH(
                    dataset,
                    params.getNumberOfHashTables(),
                    params.getNumberOfHyperplanesPerTable());

            System.out.println("Performing vanilla DBSCAN with LSH clustering...");
            startTime = System.nanoTime();
            vanillaDBSCANLSH.performClustering();
            endTime = System.nanoTime();
            System.out.println("Clustering took: " + ((endTime - startTime) / 1_000_000_000.0) + " seconds");

            // Save clustering labels
            String outputFile = params.getFileName() + "_" +
                    params.getNumberOfHashTables() + "_" +
                    params.getNumberOfHyperplanesPerTable() + "_" +
                    ".idx_vanilla_lsh";
            if (params.getOutputFileName() != "") {
                    if (params.getTargetOutputDirectory() != "") {
                        outputFile = params.getTargetOutputDirectory() + params.getOutputFileName();
                    } else {
                        outputFile = params.getOutputFileName();
                    }
            }

            System.out.println("Saving clustering labels...");

            dataset.printData(outputFile, ',', true);

            // Writing metadata to file
            String metadataFile = params.getFileName() + "_" +
                    params.getNumberOfHashTables() + "_" +
                    params.getNumberOfHyperplanesPerTable() + "_" +
                    ".metadata_vanilla_lsh";
            if (params.getMetadataFileName() != "") {
                    if (params.getTargetOutputDirectory() != "") {
                        metadataFile = params.getTargetOutputDirectory() + params.getMetadataFileName();
                    } else {
                        metadataFile = params.getMetadataFileName();
                    }
            }

            System.out.println("Exporting cluster metadata to " + metadataFile);
            dataset.exportClusterMetadata(metadataFile);

            // Compute accuracy against baseline if provided
            if (!params.getBaselineFileName().isEmpty()) {
                System.out.println("Computing clustering accuracy against baseline: " +
                        params.getBaselineFileName() + " ...");

                computeAccuracy(outputFile, params.getBaselineFileName(), "accuracy.txt");
            }

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Compute clustering accuracy against a baseline
     */
    private void computeAccuracy(String outputFile, String baselineFile, String accuracyFile) {
        try {
            System.out.println("Computing accuracy between: " + outputFile + " and baseline: " + baselineFile);

            // Check if files exist before proceeding
            File clusterFile = new File(outputFile);
            File truthFile = new File(baselineFile);

            if (!clusterFile.exists()) {
                System.err.println("Error: Cluster output file not found: " + outputFile);
                return;
            }

            if (!truthFile.exists()) {
                System.err.println("Error: Baseline file not found: " + baselineFile);
                return;
            }

            // Use the utility class for accuracy computation
            ClusteringAccuracy.computeAccuracy(outputFile, baselineFile, accuracyFile);

            System.out.println("Accuracy results written to: " + accuracyFile);
        } catch (IOException e) {
            System.err.println("Error computing accuracy: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
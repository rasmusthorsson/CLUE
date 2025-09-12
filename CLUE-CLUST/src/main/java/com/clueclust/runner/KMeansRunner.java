
package com.clueclust.runner;

import com.clueclust.algorithm.KMeans;
import com.clueclust.core.Dataset;
import com.clueclust.util.ClusteringAccuracy;
import com.clueclust.util.Globals;
import com.clueclust.util.Parameters;

import java.io.File;
import java.io.IOException;

/**
 * Runner for the K-Means clustering algorithm
 */
public class KMeansRunner implements Runner {

    @Override
    public void run(Parameters params) {
        try {
            // Read dataset
            Dataset dataset = new Dataset();
            dataset.setHasHeader(false);
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
            int k = params.getK(); // Number of clusters
            int maxIterations = params.getMaxIterations(); // Maximum iterations

            System.out.println("Performing K-Means clustering with k=" + k + ", maxIterations=" + maxIterations);
            System.out.println("Dataset dimensions: " + dataset.getNumberOfDimensions());
            System.out.println("Dataset size: " + dataset.getPoints().size() + " points");

            KMeans kMeans = new KMeans(dataset, k, maxIterations);

            startTime = System.nanoTime();
            kMeans.performClustering();
            endTime = System.nanoTime();
            System.out.println("Clustering took: " + ((endTime - startTime) / 1_000_000_000.0) + " seconds");

            // Save clustering labels
            System.out.println("Saving clustering labels...");
            String outputFile = params.getFileName() + "_k" + k + "_kmeans.idx";
            if (params.getOutputFileName() != "") {
                    if (params.getTargetOutputDirectory() != "") {
                        outputFile = params.getTargetOutputDirectory() + params.getOutputFileName();
                    } else {
                        outputFile = params.getOutputFileName();
                    }
            }
            dataset.printData(outputFile, ',', true);

            // Writing metadata to file
            String metadataFile = params.getFileName() + "_k" + k + "_kmeans.metadata";
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

                computeAccuracy(outputFile, params.getBaselineFileName(), "accuracy_kmeans.txt");
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
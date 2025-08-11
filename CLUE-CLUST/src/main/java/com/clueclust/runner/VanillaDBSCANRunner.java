package com.clueclust.runner;

import com.clueclust.algorithm.VanillaDBSCAN;
import com.clueclust.core.Dataset;
import com.clueclust.core.Point;
import com.clueclust.util.Parameters;
import com.clueclust.util.ClusterMetrics;
import com.clueclust.util.ClusteringAccuracy;
import com.clueclust.util.Globals;
import com.clueclust.util.ParameterResult;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Runner for the vanilla DBSCAN algorithm
 */
public class VanillaDBSCANRunner implements Runner {

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
            VanillaDBSCAN vanillaDBSCAN = new VanillaDBSCAN(dataset);

            System.out.println("Performing vanilla DBSCAN clustering...");
            startTime = System.nanoTime();
            vanillaDBSCAN.performClustering();
            endTime = System.nanoTime();
            System.out.println("Clustering took: " + ((endTime - startTime) / 1_000_000_000.0) + " seconds");

            // Save clustering labels
            System.out.println("Saving clustering labels...");
            String outputFile = params.getFileName() + "_" + ".idx_vanilla";
            if (params.getOutputFileName() != "") {
                    if (params.getTargetOutputDirectory() != "") {
                        outputFile = params.getTargetOutputDirectory() + params.getOutputFileName();
                    } else {
                        outputFile = params.getOutputFileName();
                    }
            }

            dataset.printData(outputFile, ',', true);

            // Writing metadata to file
            String metadataFile = params.getFileName() + "_vanilla.metadata";
            if (params.getMetadataFileName() != "") {
                    if (params.getTargetOutputDirectory() != "") {
                        metadataFile = params.getTargetOutputDirectory() + params.getMetadataFileName();
                    } else {
                        metadataFile = params.getMetadataFileName();
                    }
            }
            System.out.println("Exporting cluster metadata to " + metadataFile);
            dataset.exportClusterMetadata(metadataFile);

            System.out.println("Running Param evaluationssss");
            ParameterResult result = evaluateClusteringResult(Globals.EPSILON, Globals.MIN_PTS, 0, 0, dataset);
            System.out.println(result.toStringNoDegen());
            System.out.println("Davies-Bouldin: Lower is better\n" +
                                "Silhuette coefficient: Higher is better\n" +
                                "Calinski-Harabasz: Higher is better\n" +
                                "Dunn index: Higher is better\n" +
                                "DBCV: Higher is better\n");

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
     * Evaluate clustering results for a parameter combination
     */
    public ParameterResult evaluateClusteringResult(
            double epsilon, int minPts, int hashTables, int hyperplanes, Dataset dataset) {

        // Count clusters and noise points
        Map<Integer, Integer> clusterSizes = new HashMap<>();
        int noiseCount = 0;
        int totalPoints = 0;

        for (Point p : dataset.getPoints()) {
            int label = p.getLabel();
            if (label == Point.NOISE) {
                noiseCount++;
            } else {
                clusterSizes.put(label, clusterSizes.getOrDefault(label, 0) + 1);
            }
            totalPoints++;
        }

        int numClusters = clusterSizes.size();

        // Use shared computation for clustering metrics
        ClusterMetrics metrics = new ClusterMetrics(dataset, clusterSizes);
        Map<String, Double> allMetrics = metrics.calculateAllMetrics();

        double daviesBouldinIndex = allMetrics.get("daviesBouldinIndex");
        double silhouetteCoefficient = allMetrics.get("silhouetteCoefficient");
        double calinskiHarabaszIndex = allMetrics.get("calinskiHarabaszIndex");
        double dunnIndex = allMetrics.get("dunnIndex");
        double dbcv = allMetrics.get("dbcv");

        // Check for degenerate results based on red flags from paper
        boolean isDegenerate = false;
        String degenerateReason = "";

        // Red Flag 1: Too much noise (>40%) or too little noise (<1%)
        double noisePercentage = (double) noiseCount / totalPoints * 100;
        if (noisePercentage > 70.0) {
            isDegenerate = true;
            degenerateReason = "Too much noise (" + String.format("%.1f", noisePercentage) + "%)";
        } else if (noisePercentage < 1.0 && numClusters > 1) {
            isDegenerate = true;
            degenerateReason = "Too little noise (" + String.format("%.1f", noisePercentage) + "%)";
        }

        // Red Flag 2: One dominant cluster (>90% of non-noise points)
        int maxClusterSize = clusterSizes.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        double maxClusterPercentage = totalPoints > noiseCount
                ? (double) maxClusterSize / (totalPoints - noiseCount) * 100
                : 0;

        if (maxClusterPercentage > 99.0) {
            isDegenerate = true;
            if (degenerateReason.isEmpty()) {
                degenerateReason = "Dominant cluster (" + String.format("%.1f", maxClusterPercentage) + "%)";
            } else {
                degenerateReason += ", dominant cluster (" + String.format("%.1f", maxClusterPercentage) + "%)";
            }
        }

        return new ParameterResult(
                epsilon, minPts, hashTables, hyperplanes,
                numClusters, noiseCount, totalPoints - noiseCount,
                daviesBouldinIndex, silhouetteCoefficient, calinskiHarabaszIndex, dunnIndex, dbcv,
                isDegenerate, degenerateReason);
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
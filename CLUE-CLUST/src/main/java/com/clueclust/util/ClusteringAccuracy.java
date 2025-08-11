package com.clueclust.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Utility class for computing clustering accuracy against a ground truth
 */
public class ClusteringAccuracy {

    /**
     * Compute accuracy metrics between clustering results and ground truth labels
     * 
     * @param clusterLabelsFile Path to the file with clustering results
     * @param groundTruthFile   Path to the file with ground truth labels
     * @param outputFile        Path to write accuracy results
     * @throws IOException If there's an error reading or writing files
     */
    public static void computeAccuracy(String clusterLabelsFile, String groundTruthFile,
            String outputFile) throws IOException {
        System.out.println("Computing clustering accuracy...");
        char delimiter = ',';
        // Read ground truth labels
        Map<String, String> groundTruthLabels = readGroundTruthLabels(groundTruthFile, delimiter);
        System.out.println("Read " + groundTruthLabels.size() + " ground truth labels");

        // Read clustering labels
        Map<String, Integer> clusterLabels = readClusterLabels(clusterLabelsFile, delimiter);
        System.out.println("Read " + clusterLabels.size() + " cluster labels");

        // Find the best mapping between cluster IDs and ground truth labels
        Map<Integer, String> labelMapping = findBestLabelMapping(clusterLabels, groundTruthLabels);

        // Calculate accuracy metrics
        Map<String, Double> metrics = calculateMetrics(clusterLabels, groundTruthLabels, labelMapping);

        // Write results to output file
        writeAccuracyResults(outputFile, metrics, labelMapping, groundTruthLabels, clusterLabels);

        System.out.println("Accuracy computation complete. Results written to " + outputFile);
        System.out.println("Overall accuracy: " + String.format("%.2f%%", metrics.get("accuracy") * 100));
    }

    /**
     * Read ground truth labels from a CSV file
     */
    private static Map<String, String> readGroundTruthLabels(String fileName, char delimiter) throws IOException {
        Map<String, String> labels = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            // Process all lines, no header skipping
            while ((line = reader.readLine()) != null) {
                processGroundTruthLine(line, labels, delimiter);
            }
        }

        return labels;
    }

    /**
     * Process a single line from the ground truth file
     */
    private static void processGroundTruthLine(String line, Map<String, String> labels, char delimiter) {
        if (line == null || line.trim().isEmpty()) {
            return;
        }

        String[] parts = line.split(String.valueOf(delimiter), 2);
        if (parts.length >= 2) {
            String id = parts[0].trim();
            String label = parts[1].trim();
            labels.put(id, label);
        }
    }

    /**
     * Read clustering labels from a file
     */
    private static Map<String, Integer> readClusterLabels(String fileName, char delimiter) throws IOException {
        Map<String, Integer> labels = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            // Process all lines, no header skipping
            while ((line = reader.readLine()) != null) {
                processClusterLine(line, labels, delimiter, 0);
            }
        }

        return labels;
    }

    /**
     * Process a single line from the cluster labels file
     */
    private static void processClusterLine(String line, Map<String, Integer> labels, char delimiter, int lineNumber) {
        if (line == null || line.trim().isEmpty()) {
            return;
        }

        String[] parts = line.split(String.valueOf(delimiter), 2);
        try {
            if (parts.length >= 2) {
                // File has ID and label columns
                String id = parts[0].trim();
                int label = Integer.parseInt(parts[1].trim());
                labels.put(id, label);
            } else {
                // File only has labels, use line number as ID
                int label = Integer.parseInt(parts[0].trim());
                labels.put(String.valueOf(lineNumber), label);
            }
        } catch (NumberFormatException e) {
            System.err.println("Warning: Non-numeric cluster label found: " + line);
        }
    }

    /**
     * Find the best mapping between cluster IDs and ground truth labels
     * Uses a majority voting approach for each cluster
     */
    private static Map<Integer, String> findBestLabelMapping(
            Map<String, Integer> clusterLabels,
            Map<String, String> groundTruthLabels) {

        Map<Integer, String> mapping = new HashMap<>();
        Map<Integer, Map<String, Integer>> clusterLabelCounts = new HashMap<>();

        mapping.put(-1, "NOISE");

        // Count occurrences of each ground truth label in each cluster
        for (String id : clusterLabels.keySet()) {
            if (!groundTruthLabels.containsKey(id)) {
                continue;
            }

            Integer clusterId = clusterLabels.get(id);
            String truthLabel = groundTruthLabels.get(id);

            // Skip noise points in counting phase
            if (clusterId == -1) {
                continue;
            }

            clusterLabelCounts.putIfAbsent(clusterId, new HashMap<>());
            Map<String, Integer> counts = clusterLabelCounts.get(clusterId);
            counts.put(truthLabel, counts.getOrDefault(truthLabel, 0) + 1);
        }

        // For each cluster, find the most common ground truth label
        for (Integer clusterId : clusterLabelCounts.keySet()) {
            Map<String, Integer> counts = clusterLabelCounts.get(clusterId);
            String bestLabel = null;
            int maxCount = 0;

            for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    bestLabel = entry.getKey();
                }
            }

            if (bestLabel != null) {
                mapping.put(clusterId, bestLabel);
            }
        }

        return mapping;
    }

    /**
     * Calculate accuracy metrics
     * Accuracy: What percentage of all points were correctly assigned to their
     * proper cluster?
     * Precision: How many points assigned to this cluster actually belong there?
     * Recall: How many points that should be in this cluster were actually
     * captured?
     * F1 Score: Harmonic mean of precision and recall
     */
    private static Map<String, Double> calculateMetrics(
            Map<String, Integer> clusterLabels,
            Map<String, String> groundTruthLabels,
            Map<Integer, String> labelMapping) {

        Map<String, Double> metrics = new HashMap<>();

        int correctCount = 0;
        int totalCount = 0;
        int noiseCount = 0;
        int correctNoiseCount = 0;

        // Calculate confusion matrix and counts
        Map<String, Map<String, Integer>> confusionMatrix = new HashMap<>();
        Set<String> allTruthLabels = new HashSet<>();
        allTruthLabels.add("NOISE");

        for (String id : clusterLabels.keySet()) {
            if (!groundTruthLabels.containsKey(id)) {
                continue;
            }

            Integer clusterId = clusterLabels.get(id);
            String truthLabel = groundTruthLabels.get(id);
            String predictedLabel = labelMapping.getOrDefault(clusterId, "unknown");

            allTruthLabels.add(truthLabel);

            // Special handling for noise points
            boolean isNoiseCluster = clusterId == -1;
            boolean isNoiseGroundTruth = "NOISE".equals(truthLabel);

            if (isNoiseCluster) {
                noiseCount++;
                if (isNoiseGroundTruth) {
                    correctNoiseCount++;
                }
            }

            // Update confusion matrix
            confusionMatrix.putIfAbsent(truthLabel, new HashMap<>());
            Map<String, Integer> row = confusionMatrix.get(truthLabel);
            row.put(predictedLabel, row.getOrDefault(predictedLabel, 0) + 1);

            // Update accuracy count
            if (predictedLabel.equals(truthLabel)) {
                correctCount++;
            }
            totalCount++;
        }

        // Calculate overall accuracy
        double accuracy = totalCount > 0 ? (double) correctCount / totalCount : 0.0;
        metrics.put("accuracy", accuracy);

        // Calculate noise precision and recall
        double noiseRecall = noiseCount > 0 ? (double) correctNoiseCount / noiseCount : 0.0;
        metrics.put("noise_recall", noiseRecall);

        // Calculate precision, recall, and F1 score for each label
        for (String truthLabel : allTruthLabels) {
            int tp = confusionMatrix.getOrDefault(truthLabel, Collections.emptyMap())
                    .getOrDefault(truthLabel, 0);

            int fp = 0;
            for (String otherTruthLabel : allTruthLabels) {
                if (!otherTruthLabel.equals(truthLabel)) {
                    fp += confusionMatrix.getOrDefault(otherTruthLabel, Collections.emptyMap())
                            .getOrDefault(truthLabel, 0);
                }
            }

            int fn = 0;
            for (String predictedLabel : allTruthLabels) {
                if (!predictedLabel.equals(truthLabel)) {
                    fn += confusionMatrix.getOrDefault(truthLabel, Collections.emptyMap())
                            .getOrDefault(predictedLabel, 0);
                }
            }

            double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
            double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
            double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

            metrics.put("precision_" + truthLabel, precision);
            metrics.put("recall_" + truthLabel, recall);
            metrics.put("f1_" + truthLabel, f1);
        }

        return metrics;
    }

    /**
     * Write accuracy results to a file
     */
    private static void writeAccuracyResults(
            String outputFile,
            Map<String, Double> metrics,
            Map<Integer, String> labelMapping,
            Map<String, String> groundTruthLabels,
            Map<String, Integer> clusterLabels) throws IOException {

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
            // Write header
            writer.write("Clustering Accuracy Report");
            writer.newLine();
            writer.write("=========================");
            writer.newLine();
            writer.newLine();

            // Write overall accuracy
            writer.write(String.format("Overall Accuracy: %.2f%%", metrics.get("accuracy") * 100));
            writer.newLine();
            writer.newLine();

            // Write cluster to label mapping
            writer.write("Cluster to Ground Truth Label Mapping:");
            writer.newLine();
            writer.write("------------------------------------");
            writer.newLine();

            for (Map.Entry<Integer, String> entry : labelMapping.entrySet()) {
                writer.write(String.format("Cluster %d -> %s", entry.getKey(), entry.getValue()));
                writer.newLine();
            }
            writer.newLine();

            // Write per-label metrics
            writer.write("Per-Label Metrics:");
            writer.newLine();
            writer.write("-----------------");
            writer.newLine();

            Set<String> uniqueLabels = new HashSet<>(labelMapping.values());
            for (String label : uniqueLabels) {
                writer.write(String.format("Label: %s", label));
                writer.newLine();

                Double precision = metrics.get("precision_" + label);
                Double recall = metrics.get("recall_" + label);
                Double f1 = metrics.get("f1_" + label);

                if (precision != null) {
                    writer.write(String.format("  Precision: %.2f%%", precision * 100));
                    writer.newLine();
                }

                if (recall != null) {
                    writer.write(String.format("  Recall: %.2f%%", recall * 100));
                    writer.newLine();
                }

                if (f1 != null) {
                    writer.write(String.format("  F1 Score: %.2f%%", f1 * 100));
                    writer.newLine();
                }

                writer.newLine();

            }

            // Write statistics
            writer.write("Statistics:");
            writer.newLine();
            writer.write("-----------");
            writer.newLine();
            writer.write(String.format("Ground Truth Labels: %d", groundTruthLabels.size()));
            writer.newLine();
            writer.write(String.format("Cluster Labels: %d", clusterLabels.size()));
            writer.newLine();
            writer.write(String.format("Matched Labels: %d", labelMapping.size()));
            writer.newLine();
        }
    }
}
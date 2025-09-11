package com.clueclust.core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.clueclust.util.Globals;

//Dataset class for managing a collection of points for clustering
public class Dataset {
    private int numberOfDimensions = 0;
    private List<Point> points = new ArrayList<>();
    private List<String> columnNames = new ArrayList<>();
    private List<String> rowIds = new ArrayList<>();
    private boolean hasHeader = false;
    private String delimiter = ",";

    private double[] featureMeans;
    private double[] featureStdDevs;
    private boolean isStandardized = false;

    // Store the original data before normalization
    private List<List<Double>> originalFeatures = new ArrayList<>();

    // Get the number of dimensions in this dataset
    public int getNumberOfDimensions() {
        return numberOfDimensions;
    }

    // Get the list of points in this dataset
    public List<Point> getPoints() {
        return points;
    }

    // Get the row IDs
    public List<String> getRowIds() {
        return rowIds;
    }

    // Get the column names
    public List<String> getColumnNames() {
        return columnNames;
    }

    // Set whether the dataset has a header row
    public void setHasHeader(boolean hasHeader) {
        this.hasHeader = hasHeader;
    }

    // Set the delimiter used in the CSV file
    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }

    // Get original features before normalization
    public List<List<Double>> getOriginalFeatures() {
        return originalFeatures;
    }

    // Get original features for a specific point by index
    public List<Double> getOriginalFeatures(int index) {
        if (index >= 0 && index < originalFeatures.size()) {
            return originalFeatures.get(index);
        }
        return null;
    }

    // Read data from a file (auto-detect format based on extension)
    public void readData(String fileName) throws IOException {
        if (fileName.toLowerCase().endsWith(".csv")) {
            readCSVData(fileName);
        } else {
            readPlainData(fileName);
        }
    }

    // Read data from a plain text file (original method)
    private void readPlainData(String fileName) throws IOException {
        System.out.println("Reading input dataset in plain format...");

        long id = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.trim().split("\\s+");
                List<Double> features = new ArrayList<>();

                // Add each value as a feature
                for (String value : values) {
                    try {
                        features.add(Double.parseDouble(value));
                    } catch (NumberFormatException e) {
                        System.err.println("Warning: Non-numeric value '" + value +
                                "' found for row " + id + ". Replacing with 0.0");
                        features.add(0.0);
                    }
                }

                Point point = new Point(features);
                point.setId(id++);
                points.add(point);

                // Store original features
                originalFeatures.add(new ArrayList<>(features));
            }
        }

        validateDataset();
    }

    // Read data from a CSV file
    public void readCSVData(String fileName) throws IOException {
        System.out.println("Reading input dataset in CSV format...");

        points.clear();
        rowIds.clear();
        columnNames.clear();

        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            boolean firstLine = true;
            long idCounter = 0;

            while ((line = reader.readLine()) != null) {
                // Handle header if present
                if (firstLine && hasHeader) {
                    String[] headers = line.split(delimiter);
                    for (int i = 1; i < headers.length; i++) {
                        columnNames.add(headers[i]);
                    }
                    firstLine = false;
                    continue;
                }

                String[] values = line.split(delimiter, -1);

                // Get ID from first column
                String rowId = values[0];
                rowIds.add(rowId);

                // Create point from remaining columns
                List<Double> features = new ArrayList<>();
                for (int i = 1; i < values.length; i++) {
                    String value = values[i].trim();
                    try {
                        features.add(value.isEmpty() ? 0.0 : Double.parseDouble(value));
                    } catch (NumberFormatException e) {
                        System.err.println("Warning: Non-numeric value '" + value + "' found for row " + rowId +
                                ", column " + i + ". Replacing with 0.0");
                        features.add(0.0);
                    }
                }
                Point point = new Point(features);
                point.setId(idCounter++);
                points.add(point);
                
                // Store original features
                originalFeatures.add(new ArrayList<>(features));

                firstLine = false;
            }
        }

        validateDataset();
    }

    // Validate the dataset after loading
    private void validateDataset() throws IOException {
        if (points.isEmpty()) {
            throw new IOException("Error while reading data: Input dataset is empty");
        }

        this.numberOfDimensions = points.get(0).getFeatures().size();

        // Check consistency of dimensions
        for (Point point : points) {
            if (point.getFeatures().size() != numberOfDimensions) {
                throw new IOException("Error while reading data: Inconsistent dimensionality among data points");
            }
        }
    }

    public void relabelData() {
        // First, do standard path compression to flatten the union-find tree
        for (Point point : points) {
            point.unsafeCompress();
        }

        // Then relabel, ensuring only points with core roots get cluster labels
        int nonCoreRoots = 0;
        for (Point point : points) {
            if (point.getLabel() == Point.NON_NOISE) {
                Point root = point.findRoot();
                if (!root.isCore()) {
                    nonCoreRoots++;
                    point.setLabel(Point.NOISE); // Mark as noise if root is not core
                } else {
                    point.reLabel(); // Normal relabeling if root is core
                }
            }
        }

        if (nonCoreRoots > 0) {
            System.out.println("Corrected " + nonCoreRoots + " points with non-core roots to NOISE during relabeling");
        }
    }

    // Reset all points in the dataset
    public void resetData() {
        for (Point point : points) {
            point.reset();
        }
    }

    // Normalize all points in the dataset
    public void normalizeData() {
        System.out.println("Normalizing Data ...");
        this.numberOfDimensions++;

        for (Point point : points) {
            point.normalize(true);
        }
        this.numberOfDimensions--;
    }

    // Remove the mean from all points in the dataset
    public void meanRemoveData() {
        System.out.println("Mean Removing Data ...");

        // Create a mean point with all zeros
        Point mean = new Point();
        for (int i = 0; i < numberOfDimensions; i++) {
            mean.getFeatures().add(0.0);
        }

        // Sum all points
        for (Point p : points) {
            mean.addInPlace(p);
        }

        // Divide by number of points to get mean
        mean.divideInPlace(points.size());

        // Subtract mean from each point
        for (Point p : points) {
            p.subtractInPlace(mean);
        }
    }

    // Print the dataset to a file
    public void printData(String fileName, char delimiter, boolean onlyLabel) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (int i = 0; i < points.size(); i++) {
                Point p = points.get(i);

                if (onlyLabel) {
                    // Always include original ID and label
                    String id = (i < rowIds.size()) ? rowIds.get(i) : String.valueOf(p.getId());
                    writer.write(id + delimiter + p.getLabel());
                    writer.newLine();
                } else {
                    // Full data output
                    if (fileName.toLowerCase().endsWith(".csv") && i < rowIds.size()) {
                        // CSV output with original IDs
                        writer.write(rowIds.get(i) + delimiter);
                        List<Double> features = p.getFeatures();
                        for (int j = 0; j < features.size(); j++) {
                            writer.write(String.format("%.6f", features.get(j)));
                            if (j < features.size() - 1) {
                                writer.write(String.valueOf(delimiter));
                            }
                        }
                        writer.write(String.valueOf(delimiter) + p.getLabel());
                    } else {
                        // Standard output
                        writer.write(p.toString());
                    }
                    writer.newLine();
                }
            }
        }
    }

    public void enforceMinClusterSize() {
        System.out.println("Enforcing minimum cluster size...");

        // Step 1: Count points per cluster
        Map<Integer, Integer> clusterSizes = new HashMap<>();
        for (Point p : points) {
            int label = p.getLabel();
            if (label != Point.NOISE) {
                clusterSizes.put(label, clusterSizes.getOrDefault(label, 0) + 1);
            }
        }

        System.out.println("Found " + clusterSizes.size() + " clusters before enforcement");

        // Step 2: Mark small clusters as noise
        int smallClustersCount = 0;
        int pointsRemarkedAsNoise = 0;

        for (Point p : points) {
            int label = p.getLabel();
            if (label != Point.NOISE) {
                Integer size = clusterSizes.get(label);
                if (size != null && size < Globals.MIN_PTS) {
                    p.setLabel(Point.NOISE);
                    pointsRemarkedAsNoise++;
                    smallClustersCount++;
                }
            }
        }

        System.out.println("Marked " + pointsRemarkedAsNoise + " points from " +
                smallClustersCount + " small clusters as noise");
    }

    public void validateOriginalFeatures() {
        if (originalFeatures.isEmpty()) {
            System.err.println("ERROR: originalFeatures list is empty!");
            return;
        }

        if (originalFeatures.size() != points.size()) {
            System.err.println("ERROR: originalFeatures size (" + originalFeatures.size() +
                    ") doesn't match points size (" + points.size() + ")");
            return;
        }

        // Check a few sample points to verify original features are different from
        // normalized
        if (Globals.METRIC == Globals.Metric.ANGULAR && numberOfDimensions > 0) {
            System.out.println("Validating original vs. normalized features for 5 sample points:");
            for (int i = 0; i < Math.min(5, points.size()); i++) {
                Point p = points.get(i);
                List<Double> origFeatures = originalFeatures.get(i);
                List<Double> currFeatures = p.getFeatures();

                System.out.println("Point " + i + ":");
                System.out.println("  Original features: " + origFeatures);
                System.out.println("  Current features: " + currFeatures);

                boolean different = false;
                for (int j = 0; j < Math.min(origFeatures.size(), currFeatures.size()); j++) {
                    if (Math.abs(origFeatures.get(j) - currFeatures.get(j)) > 0.0001) {
                        different = true;
                        break;
                    }
                }

                if (currFeatures.size() != origFeatures.size()) {
                    different = true;
                }

                System.out.println("  Features are " + (different ? "different" : "identical") +
                        " (expect different for normalized data)");
            }
        } else {
            System.out.println("Validation not necessary - data not normalized");
        }

        System.out.println("Original features tracking is " +
                (originalFeatures.isEmpty() ? "NOT " : "") + "enabled");
    }

    /*
     * Create cluster metadata for all clusters and export to file asda
     * Includes bounding boxes and summary statistics
     */

    public void exportClusterMetadata(String fileName) throws IOException {
        // Collect all points by cluster
        Map<Integer, List<Point>> pointsByCluster = new HashMap<>();
        Map<Integer, ClusterMetadata> clusterMetadata = new HashMap<>();

        for (int i = 0; i < points.size(); i++) {
            Point p = points.get(i);
            int label = p.getLabel();
            if (true) { //Do not skip noise points as we want the option to consider them as a cluster later on
            //if (label != Point.NOISE) { // Skip noise points
                // Add to points by cluster
                pointsByCluster.computeIfAbsent(label, k -> new ArrayList<>()).add(p);

                // Initialize metadata if not already
                if (!clusterMetadata.containsKey(label)) {
                    clusterMetadata.put(label, new ClusterMetadata(label, numberOfDimensions));
                }

                // Add point to metadata (with original features if available)
                ClusterMetadata metadata = clusterMetadata.get(label);
                metadata.addPoint(p);

                // Add original features if available
                if (!originalFeatures.isEmpty()) {
                    metadata.addOriginalFeatures(originalFeatures.get(i));
                }
            }
        }

        // Inside exportClusterMetadata in Dataset.java
        System.err.println("DEBUG: Starting exportClusterMetadata"); // Use System.err for immediate output
        System.err.println("DEBUG: Total points: " + points.size());
        System.err.println("DEBUG: minPts value: " + Globals.MIN_PTS);

        // After collecting pointsByCluster
        System.err.println("DEBUG: Total clusters found: " + pointsByCluster.size());
        for (Map.Entry<Integer, List<Point>> entry : pointsByCluster.entrySet()) {
            int label = entry.getKey();
            List<Point> clusterPoints = entry.getValue();
            System.err.println("DEBUG: Cluster " + label + " has " + clusterPoints.size() + " points");

            if (clusterPoints == null || clusterPoints.isEmpty())
                continue;

            boolean hasCorePoint = false;
            for (Point p : clusterPoints) {
                if (p.isCore()) {
                    hasCorePoint = true;
                    break;
                }
            }

            if (!hasCorePoint) {
                System.err.println("WARNING: Cluster " + label + " has NO core points! Size=" + clusterPoints.size());
                if (clusterPoints.size() < 5) {
                    System.err.println("Points in small cluster without core point:");
                    for (Point p : clusterPoints) {
                        System.err.println("  ID=" + p.getId() + ", label=" + p.getLabel() + ", isCore=" + p.isCore());
                    }
                }
            }

            if (clusterPoints.size() < Globals.MIN_PTS) {
                System.err.println("WARNING: Cluster " + label + " is smaller than minPts (" +
                        Globals.MIN_PTS + ")! Size = " + clusterPoints.size());
            }
        }

        // Write metadata to file
        try (

                BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            // Write header
            writer.write("ClusterId,Size,NumCorePts");
            String[] headers = { ",Min_Dim", ",Max_Dim", ",Sum", ",Mean", ",Std", ",Var" };
            String[] origFeatHeaders = { ",OrigMin_Dim", ",OrigMax_Dim", ",OrigMean" };

            int originalDims = originalFeatures.isEmpty() ? 0 : originalFeatures.get(0).size();

            for (String header : headers) {
                for (int i = 0; i < numberOfDimensions; i++) {
                    writer.write(header + i);
                }
            }
            for (String header : origFeatHeaders) {
                for (int i = 0; i < originalDims; i++) {
                    writer.write(header + i);
                }
            }
            /*
             * for (int i = 0; i < numberOfDimensions; i++) {
             * writer.write(",Min_Dim" + i + ",Max_Dim" + i + ",Sum" + i + ",Mean" + i +
             * ",Std" + i);
             * }
             * // Add header for original dimensions if available
             * int originalDims = originalFeatures.isEmpty() ? 0 :
             * originalFeatures.get(0).size();
             * if (originalDims > 0) {
             * for (int i = 0; i < originalDims; i++) {
             * writer.write(",OrigMin_Dim" + i + ",OrigMax_Dim" + i + ",OrigMean" + i);
             * }
             */

            writer.write(",Diameter,DensityCorePoints");
            writer.newLine();

            // Write data for each cluster
            for (ClusterMetadata metadata : clusterMetadata.values()) {
                writer.write(metadata.getClusterId() + ",");
                writer.write(metadata.getSize() + ",");
                writer.write(metadata.getCorePoints().size() + "");

                // Min and max bounds for each dimension
                List<Double> minBounds = metadata.getMinBounds();
                List<Double> maxBounds = metadata.getMaxBounds();
                List<Double> sums = metadata.getSums();
                List<Double> means = metadata.getMeans();
                List<Double> standardDev = metadata.getStandardDev();
                List<Double> variance = metadata.getVariance();

                for (int i = 0; i < numberOfDimensions; i++) {
                    writer.write("," + String.format("%.6f", minBounds.get(i)));
                }
                for (int i = 0; i < numberOfDimensions; i++) {
                    writer.write("," + String.format("%.6f", maxBounds.get(i)));
                }
                for (int i = 0; i < numberOfDimensions; i++) {
                    writer.write("," + String.format("%.6f", sums.get(i)));
                }
                for (int i = 0; i < numberOfDimensions; i++) {
                    writer.write("," + String.format("%.6f", means.get(i)));
                }
                for (int i = 0; i < numberOfDimensions; i++) {
                    writer.write("," + String.format("%.6f", standardDev.get(i)));
                }
                for (int i = 0; i < numberOfDimensions; i++) {
                    writer.write("," + String.format("%.6f", variance.get(i)));
                }

                // Write original dimensions if available
                if (originalDims > 0) {
                    List<Double> origMinBounds = metadata.getOriginalMinBounds();
                    List<Double> origMaxBounds = metadata.getOriginalMaxBounds();
                    List<Double> origMeans = metadata.getOriginalMeans();

                    for (int i = 0; i < originalDims; i++) {
                        writer.write("," + String.format("%.6f", origMinBounds.get(i)));
                    }
                    for (int i = 0; i < originalDims; i++) {
                        writer.write("," + String.format("%.6f", origMaxBounds.get(i)));
                    }
                    for (int i = 0; i < originalDims; i++) {
                        writer.write("," + String.format("%.6f", origMeans.get(i)));
                    }
                }
                // Diameter and density
                writer.write("," + String.format("%.6f", metadata.getDiameter()));
                double density = (double) metadata.getCorePoints().size() / metadata.getSize();
                writer.write("," + String.format("%.6f", density));

                writer.newLine();

            }
        }
    }

    /**
     * Standardize all points in the dataset (z-score normalization)
     */
    public void standardizeData() {
        if (isStandardized) {
            System.out.println("Data is already standardized.");
            return;
        }

        System.out.println("Standardizing data...");
        int dimensions = getNumberOfDimensions();

        // Initialize arrays for means and standard deviations
        featureMeans = new double[dimensions];
        featureStdDevs = new double[dimensions];

        // Calculate means
        for (Point point : points) {
            List<Double> features = point.getFeatures();
            for (int i = 0; i < dimensions; i++) {
                featureMeans[i] += features.get(i);
            }
        }

        for (int i = 0; i < dimensions; i++) {
            featureMeans[i] /= points.size();
        }

        // Calculate standard deviations
        for (Point point : points) {
            List<Double> features = point.getFeatures();
            for (int i = 0; i < dimensions; i++) {
                double diff = features.get(i) - featureMeans[i];
                featureStdDevs[i] += diff * diff;
            }
        }

        for (int i = 0; i < dimensions; i++) {
            featureStdDevs[i] = Math.sqrt(featureStdDevs[i] / points.size());
            // Avoid division by zero
            if (featureStdDevs[i] < 1e-8) {
                featureStdDevs[i] = 1.0;
                System.out.println("Warning: Feature " + i + " has near-zero standard deviation. Using 1.0 instead.");
            }
        }

        // Apply standardization to each point
        for (Point point : points) {
            List<Double> features = point.getFeatures();

            // Update the point's features
            for (int i = 0; i < dimensions; i++) {
                if (i < features.size()) {
                    // Replace existing value
                    features.set(i, (features.get(i) - featureMeans[i]) / featureStdDevs[i]);
                }
            }
        }

        isStandardized = true;
        System.out.println("Standardization complete.");
    }

    // Print the dataset to a CSV file with customizable options
    public void printCSV(String fileName, boolean includeLabels) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            // Write header if column names exist
            if (!columnNames.isEmpty()) {
                writer.write("ID");
                for (String colName : columnNames) {
                    writer.write(delimiter + colName);
                }
                if (includeLabels) {
                    writer.write(delimiter + "Cluster");
                }
                writer.newLine();
            }

            // Write data rows
            for (int i = 0; i < points.size(); i++) {
                Point p = points.get(i);

                // Write ID
                String id = (i < rowIds.size()) ? rowIds.get(i) : "point_" + p.getId();
                writer.write(id);

                // Write features
                List<Double> features = p.getFeatures();
                for (Double feature : features) {
                    writer.write(delimiter + String.format("%.6f", feature));
                }

                // Write cluster label if requested
                if (includeLabels) {
                    writer.write(delimiter + p.getLabel());
                }

                writer.newLine();
            }
        }
    }
}
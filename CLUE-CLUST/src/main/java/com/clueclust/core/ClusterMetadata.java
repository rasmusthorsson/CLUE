package com.clueclust.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Class to store additional metadata about clusters
 */
public class ClusterMetadata {
    private int clusterId;
    private int size;
    private List<Point> corePoints = new ArrayList<>();
    private List<Double> minBounds;
    private List<Double> maxBounds;
    private List<Double> means;
    private List<Double> sums;
    private List<Double> standardDev;
    private List<Double> variance;

    // Added: Track original (un-normalized) data bounds
    private List<Double> originalMinBounds;
    private List<Double> originalMaxBounds;
    private List<Double> originalSums;
    private List<Double> originalMeans;
    private int originalDimCount = 0;

    /**
     * Constructor
     */
    public ClusterMetadata(int clusterId, int dimensions) {
        this.clusterId = clusterId;
        this.size = 0;


        System.out.println("Number of dimensions: " + dimensions);
        // Initialize min and max bounds
        this.minBounds = new ArrayList<>(dimensions);
        this.maxBounds = new ArrayList<>(dimensions);
        this.sums = new ArrayList<>(dimensions);
        this.means = new ArrayList<>(dimensions);
        this.standardDev = new ArrayList<>(dimensions);
        this.variance = new ArrayList<>(dimensions);

        for (int i = 0; i < dimensions; i++) {
            minBounds.add(Double.POSITIVE_INFINITY);
            maxBounds.add(Double.NEGATIVE_INFINITY);
            means.add(0.0);
            standardDev.add(0.0);
            sums.add(0.0);
            variance.add(0.0);
        }

        // initialize the original bound trackers when we first get original data
        this.originalMinBounds = new ArrayList<>();
        this.originalMaxBounds = new ArrayList<>();
        this.originalSums = new ArrayList<>();
        this.originalMeans = new ArrayList<>();
    }

    /**
     * Initialize original feature tracking when first original features are added
     */
    private void initializeOriginalTracking(int dimensionCount) {
        if (originalDimCount == 0) {
            originalDimCount = dimensionCount;

            for (int i = 0; i < dimensionCount; i++) {
                originalMinBounds.add(Double.POSITIVE_INFINITY);
                originalMaxBounds.add(Double.NEGATIVE_INFINITY);
                originalSums.add(0.0);
                originalMeans.add(0.0);
            }
        }
    }

    /**
     * Add a point to this cluster and update bounds
     */
    public void addPoint(Point point) {
        size++;

        // Update bounds
        List<Double> features = point.getFeatures();
        for (int i = 0; i < features.size() && i < minBounds.size(); i++) {
            double value = features.get(i);

            if (value < minBounds.get(i)) {
                minBounds.set(i, value);
            }

            if (value > maxBounds.get(i)) {
                maxBounds.set(i, value);
            }
            // Sums
            sums.set(i, sums.get(i) + value);

            // Mean
            double oldMeans = means.get(i);
            means.set(i, means.get(i) + ((value - means.get(i)) / size));

            // Standard deviation
            double oldVariance = Math.pow(standardDev.get(i), 2);
            double S = oldVariance * (size - 1);
            double newS = S + (value - oldMeans) * (value - means.get(i));
            double newStandardDev = Math.sqrt(newS / size);
            standardDev.set(i, newStandardDev);

            variance.set(i, Math.pow(newStandardDev, 2));
        }

        // Add to core points if this is a core point
        if (point.isCore()) {
            corePoints.add(point);
        }
    }

    /**
     * Add original features for a point to track un-normalized bounds
     */
    public void addOriginalFeatures(List<Double> originalFeatures) {
        if (originalFeatures == null || originalFeatures.isEmpty()) {
            return;
        }

        // Initialize tracking if this is the first point
        if (originalDimCount == 0) {
            initializeOriginalTracking(originalFeatures.size());
        }

        // Update original bounds
        for (int i = 0; i < originalFeatures.size() && i < originalDimCount; i++) {
            double value = originalFeatures.get(i);

            if (value < originalMinBounds.get(i)) {
                originalMinBounds.set(i, value);
            }

            if (value > originalMaxBounds.get(i)) {
                originalMaxBounds.set(i, value);
            }

            // Update sums
            originalSums.set(i, originalSums.get(i) + value);

            // Update means (incremental calculation)
            originalMeans.set(i, originalMeans.get(i) + (value - originalMeans.get(i)) / size);
        }
    }

    /**
     * Get cluster ID
     */
    public int getClusterId() {
        return clusterId;
    }

    /**
     * Get cluster size
     */
    public int getSize() {
        return size;
    }

    /**
     * Get minimum bounds
     */
    public List<Double> getMinBounds() {
        return minBounds;
    }

    /**
     * Get maximum bounds
     */
    public List<Double> getMaxBounds() {
        return maxBounds;
    }

    public List<Double> getSums() {
        return sums;
    }

    public List<Double> getMeans() {
        return means;
    }

    public List<Double> getStandardDev() {
        return standardDev;
    }

    public List<Double> getVariance() {
        return variance;
    }

    /**
     * Get core points
     */
    public List<Point> getCorePoints() {
        return corePoints;
    }

    /**
     * Get original minimum bounds
     */
    public List<Double> getOriginalMinBounds() {
        return originalMinBounds;
    }

    /**
     * Get original maximum bounds
     */
    public List<Double> getOriginalMaxBounds() {
        return originalMaxBounds;
    }

    /**
     * Get original sums
     */
    public List<Double> getOriginalSums() {
        return originalSums;
    }

    /**
     * Get original means
     */
    public List<Double> getOriginalMeans() {
        return originalMeans;
    }

    /**
     * Get center point (midpoint of bounding box)
     */
    public List<Double> getCenter() {
        List<Double> center = new ArrayList<>(minBounds.size());

        for (int i = 0; i < minBounds.size(); i++) {
            center.add((minBounds.get(i) + maxBounds.get(i)) / 2.0);
        }

        return center;
    }

    /**
     * Get center point based on original features
     */
    public List<Double> getOriginalCenter() {
        if (originalDimCount == 0) {
            return getCenter(); // Fall back to normalized center
        }

        List<Double> center = new ArrayList<>(originalDimCount);
        for (int i = 0; i < originalDimCount; i++) {
            center.add((originalMinBounds.get(i) + originalMaxBounds.get(i)) / 2.0);
        }

        return center;
    }

    /**
     * Get the diameter of this cluster (max distance across bounding box)
     */
    public double getDiameter() {
        double sumSquared = 0.0;

        for (int i = 0; i < minBounds.size(); i++) {
            double range = maxBounds.get(i) - minBounds.get(i);
            sumSquared += range * range;
        }

        return Math.sqrt(sumSquared);
    }

    /**
     * Get the diameter of this cluster based on original features
     */
    public double getOriginalDiameter() {
        if (originalDimCount == 0) {
            return getDiameter(); // Fall back to normalized diameter
        }

        double sumSquared = 0.0;
        for (int i = 0; i < originalDimCount; i++) {
            double range = originalMaxBounds.get(i) - originalMinBounds.get(i);
            sumSquared += range * range;
        }

        return Math.sqrt(sumSquared);
    }

    /**
     * To string representation
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Cluster ").append(clusterId).append(":\n");
        sb.append("  Size: ").append(size).append(" points\n");
        sb.append("  Core Points: ").append(corePoints.size()).append("\n");
        sb.append("  Bounds:\n");

        for (int i = 0; i < minBounds.size(); i++) {
            sb.append("    Dim ").append(i).append(": [")
                    .append(String.format("%.6f", minBounds.get(i))).append(", ")
                    .append(String.format("%.6f", maxBounds.get(i))).append("]\n");
        }

        if (originalDimCount > 0) {
            sb.append("  Bounds (Original):\n");
            for (int i = 0; i < originalDimCount; i++) {
                sb.append("    Dim ").append(i).append(": [")
                        .append(String.format("%.6f", originalMinBounds.get(i))).append(", ")
                        .append(String.format("%.6f", originalMaxBounds.get(i))).append("]\n");
            }
        }

        sb.append("  Sums:\n");
        for (int i = 0; i < sums.size(); i++) {
            sb.append("    Dim ").append(i).append(": [")
                    .append(String.format("%.6f", sums.get(i))).append(", ")
                    .append(String.format("%.6f", sums.get(i))).append("]\n");
        }

        sb.append("  Means:\n");
        for (int i = 0; i < means.size(); i++) {
            sb.append("    Dim ").append(i).append(": [")
                    .append(String.format("%.6f", means.get(i))).append(", ")
                    .append(String.format("%.6f", means.get(i))).append("]\n");
        }

        sb.append("  Standard Deviation:\n");
        for (int i = 0; i < standardDev.size(); i++) {
            sb.append("    Dim ").append(i).append(": [")
                    .append(String.format("%.6f", standardDev.get(i))).append(", ")
                    .append(String.format("%.6f", standardDev.get(i))).append("]\n");
        }

        sb.append("  Variance\n");
        for (int i = 0; i < standardDev.size(); i++) {
            sb.append("    Dim ").append(i).append(": [")
                    .append(String.format("%.6f", variance.get(i))).append(", ")
                    .append(String.format("%.6f", variance.get(i))).append("]\n");
        }

        return sb.toString();
    }
}
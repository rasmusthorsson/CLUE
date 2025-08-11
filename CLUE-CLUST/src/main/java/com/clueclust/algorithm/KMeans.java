package com.clueclust.algorithm;

import com.clueclust.core.Dataset;
import com.clueclust.core.Point;
import com.clueclust.util.Globals;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * K-Means clustering algorithm implementation
 */
public class KMeans {
    private final Dataset dataset;
    private final int k;
    private final int maxIterations;
    private final List<Point> centroids = new ArrayList<>();
    private final Random random = new Random();

    // For tracking convergence
    private int iterations;
    private double lastChangeRatio;

    /**
     * Constructor
     * 
     * @param dataset       The dataset to cluster
     * @param k             Number of clusters
     * @param maxIterations Maximum number of iterations to perform
     */
    public KMeans(Dataset dataset, int k, int maxIterations) {
        this.dataset = dataset;
        this.k = k;
        this.maxIterations = maxIterations;
    }

    /**
     * Initialize centroids using the k-means++ algorithm
     */
    private void initializeCentroids() {
        List<Point> points = dataset.getPoints();

        if (points.size() < k) {
            throw new IllegalStateException("Not enough points in dataset for " + k + " clusters");
        }

        // Choose the first centroid randomly from the dataset
        int firstIndex = random.nextInt(points.size());
        Point firstCentroid = new Point(new ArrayList<>(points.get(firstIndex).getFeatures()));
        firstCentroid.setId(-1); // Use negative IDs for centroids
        centroids.add(firstCentroid);

        // Choose remaining centroids using k-means++ approach
        for (int i = 1; i < k; i++) {
            // Calculate squared distances to nearest existing centroid for each point
            double[] distances = new double[points.size()];
            double sumOfSquaredDistances = 0.0;

            for (int j = 0; j < points.size(); j++) {
                Point point = points.get(j);
                double minDist = Double.MAX_VALUE;

                // Find distance to closest centroid
                for (Point centroid : centroids) {
                    double dist = Globals.DISTANCE_FUNCTION.calculate(point, centroid);
                    minDist = Math.min(minDist, dist);
                }

                distances[j] = minDist;
                sumOfSquaredDistances += minDist;
            }

            // Choose next centroid with probability proportional to squared distance
            double rand = random.nextDouble() * sumOfSquaredDistances;
            double cumulativeProb = 0.0;
            int selectedIndex = -1;

            for (int j = 0; j < distances.length; j++) {
                cumulativeProb += distances[j];
                if (cumulativeProb >= rand) {
                    selectedIndex = j;
                    break;
                }
            }

            // If somehow we didn't select a point, choose the last one
            if (selectedIndex == -1) {
                selectedIndex = points.size() - 1;
            }

            // Create new centroid
            Point newCentroid = new Point(new ArrayList<>(points.get(selectedIndex).getFeatures()));
            newCentroid.setId(-(i + 1)); // Use negative IDs for centroids
            centroids.add(newCentroid);
        }

        System.out.println("Initialized " + centroids.size() + " centroids using k-means++");
    }

    /**
     * Assign each point to the nearest centroid
     * 
     * @return The number of points that changed cluster assignment
     */
    private int assignPointsToClusters() {
        int changedAssignments = 0;
        List<Point> points = dataset.getPoints();

        for (Point point : points) {
            double minDist = Double.MAX_VALUE;
            int closestCentroidIndex = -1;

            // Find the closest centroid
            for (int i = 0; i < centroids.size(); i++) {
                double dist = Globals.DISTANCE_FUNCTION.calculate(point, centroids.get(i));
                if (dist < minDist) {
                    minDist = dist;
                    closestCentroidIndex = i;
                }
            }

            // Assign the point to the closest centroid's cluster
            if (point.getLabel() != closestCentroidIndex) {
                point.setLabel(closestCentroidIndex);
                changedAssignments++;
            }
        }

        return changedAssignments;
    }

    /**
     * Update centroids based on cluster assignments
     * 
     * @return True if centroids changed significantly, false otherwise
     */
    private boolean updateCentroids() {
        boolean significantChange = false;
        List<Point> points = dataset.getPoints();
        int dimensions = dataset.getNumberOfDimensions();

        // For each cluster, compute the new centroid
        for (int clusterId = 0; clusterId < k; clusterId++) {
            // Count points in this cluster
            int clusterSize = 0;
            for (Point point : points) {
                if (point.getLabel() == clusterId) {
                    clusterSize++;
                }
            }

            // If cluster is empty, keep current centroid
            if (clusterSize == 0) {
                System.out.println("Warning: Cluster " + clusterId + " is empty");
                continue;
            }

            // Calculate sum of all points in the cluster
            List<Double> sum = new ArrayList<>(dimensions);
            for (int d = 0; d < dimensions; d++) {
                sum.add(0.0);
            }

            for (Point point : points) {
                if (point.getLabel() == clusterId) {
                    List<Double> features = point.getFeatures();
                    for (int d = 0; d < dimensions; d++) {
                        sum.set(d, sum.get(d) + features.get(d));
                    }
                }
            }

            // Calculate new centroid
            Point oldCentroid = centroids.get(clusterId);
            List<Double> newFeatures = new ArrayList<>(dimensions);

            for (int d = 0; d < dimensions; d++) {
                newFeatures.add(sum.get(d) / clusterSize);
            }

            // Create new centroid
            Point newCentroid = new Point(newFeatures);
            newCentroid.setId(oldCentroid.getId());

            // Calculate change
            double change = Globals.DISTANCE_FUNCTION.calculate(oldCentroid, newCentroid);
            if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
                change = Math.sqrt(change); // Convert squared distance to actual distance
            }

            // Check if change is significant
            if (change > 1e-6) {
                significantChange = true;
            }

            // Update centroid
            centroids.set(clusterId, newCentroid);
        }

        return significantChange;
    }

    /**
     * Run the K-Means algorithm
     */
    public void performClustering() {
        // Initialize centroids
        initializeCentroids();

        iterations = 0;
        int totalPoints = dataset.getPoints().size();
        boolean converged = false;

        System.out.println("Starting K-Means iterations...");

        // Main K-Means loop
        while (iterations < maxIterations && !converged) {
            // Assign points to clusters
            int changed = assignPointsToClusters();
            lastChangeRatio = (double) changed / totalPoints;

            System.out.println("Iteration " + iterations + ": " +
                    changed + " points changed clusters (" +
                    String.format("%.2f", lastChangeRatio * 100) + "%)");

            // Update centroids
            boolean centroidsChanged = updateCentroids();

            // Check convergence
            if (!centroidsChanged || lastChangeRatio < 0.001) {
                converged = true;
                System.out.println("K-Means converged after " + iterations + " iterations");
            }

            iterations++;
        }

        // One final assignment
        assignPointsToClusters();

        if (iterations >= maxIterations) {
            System.out.println("K-Means reached maximum iterations (" + maxIterations + ")");
        }

        // Mark points in each cluster as core points (needed to maintain consistency
        // with DBSCAN)
        for (Point point : dataset.getPoints()) {
            point.setAsCorePoint();
        }
    }

    /**
     * Get number of iterations performed
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * Get change ratio in the last iteration
     */
    public double getLastChangeRatio() {
        return lastChangeRatio;
    }

    /**
     * Return the calculated centroids
     */
    public List<Point> getCentroids() {
        return centroids;
    }
}
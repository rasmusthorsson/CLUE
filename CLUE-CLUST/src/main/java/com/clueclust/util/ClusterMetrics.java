package com.clueclust.util;

import com.clueclust.core.Dataset;
import com.clueclust.core.Point;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Shared cluster metrics calculation - efficiently computes both
 * Davies-Bouldin index, Silhouette coefficient, and Calinski-Harabasz index
 */
public class ClusterMetrics {
    private final Dataset dataset;
    private final Map<Integer, Integer> clusterSizes;
    private final Map<Integer, List<Point>> pointsByCluster;
    private final Map<Integer, double[]> centroids;
    private final Map<Long, Map<Long, Double>> distanceCache;
    private final Map<Integer, List<Point>> sampledPointsByCluster;
    private final int dimensions;

    // Results
    private double daviesBouldinIndex = Double.MAX_VALUE;
    private double silhouetteCoefficient = -1.0;
    private double calinskiHarabaszIndex = 0.0;

    /**
     * Constructor that prepares shared computation
     */
    public ClusterMetrics(Dataset dataset, Map<Integer, Integer> clusterSizes) {
        this.dataset = dataset;
        this.clusterSizes = clusterSizes;
        this.dimensions = dataset.getNumberOfDimensions();
        this.distanceCache = new HashMap<>();

        // 1. Group points by cluster
        System.out.println("Preparing shared cluster metrics calculation...");
        pointsByCluster = new HashMap<>();
        for (Point p : dataset.getPoints()) {
            int label = p.getLabel();
            if (label != Point.NOISE) {
                pointsByCluster.computeIfAbsent(label, k -> new ArrayList<>()).add(p);
            }
        }

        // 2. Calculate centroids
        centroids = calculateClusterCentroids();

        // 3. Sample points for efficient calculation - primarily for silhouette
        sampledPointsByCluster = samplePointsFromClusters();
    }

    /**
     * Calculate cluster centroids - used by all metrics
     */
    private Map<Integer, double[]> calculateClusterCentroids() {
        Map<Integer, double[]> result = new HashMap<>();

        for (int clusterId : clusterSizes.keySet()) {
            List<Point> clusterPoints = pointsByCluster.get(clusterId);
            if (clusterPoints == null || clusterPoints.isEmpty())
                continue;

            double[] centroid = new double[dimensions];
            for (Point p : clusterPoints) {
                List<Double> features = p.getFeatures();
                for (int i = 0; i < features.size() && i < dimensions; i++) {
                    centroid[i] += features.get(i);
                }
            }

            for (int i = 0; i < dimensions; i++) {
                centroid[i] /= clusterPoints.size();
            }

            result.put(clusterId, centroid);
        }

        return result;
    }

    /**
     * Sample points from each cluster for efficient metric calculation
     */
    private Map<Integer, List<Point>> samplePointsFromClusters() {
        Map<Integer, List<Point>> result = new HashMap<>();
        int totalPoints = dataset.getPoints().size();

        // Determine overall sample size - cubic root scaling for efficiency
        int maxSamplePoints = Math.min(2000, totalPoints / 2);
        int sampleSize = Math.min(maxSamplePoints,
                (int) Math.ceil(Math.pow(totalPoints, 0.33) * Math.log10(dimensions + 1)));

        System.out.println("Using " + sampleSize + " sample points for metrics calculation");

        for (int clusterId : clusterSizes.keySet()) {
            List<Point> clusterPoints = pointsByCluster.get(clusterId);
            if (clusterPoints == null || clusterPoints.isEmpty()) {
                result.put(clusterId, new ArrayList<>());
                continue;
            }

            // Determine sample size for this cluster
            int clusterSampleSize = Math.max(
                    Math.min(30, clusterPoints.size()), // At least 30 points (or all if fewer)
                    (int) Math.ceil((double) clusterPoints.size() / totalPoints * sampleSize));

            // Cap at cluster size
            clusterSampleSize = Math.min(clusterSampleSize, clusterPoints.size());

            // Sample points from this cluster
            List<Point> sampledPoints = new ArrayList<>(clusterSampleSize);
            if (clusterPoints.size() <= clusterSampleSize) {
                // If cluster is small enough, use all points
                sampledPoints.addAll(clusterPoints);
            } else {
                // Use stratified sampling for better representation
                double[] centroid = centroids.get(clusterId);

                // Sort points by distance to centroid
                List<Point> sortedPoints = new ArrayList<>(clusterPoints);
                sortedPoints.sort((p1, p2) -> {
                    double d1 = distanceToPoint(p1, centroid);
                    double d2 = distanceToPoint(p2, centroid);
                    return Double.compare(d1, d2);
                });

                // Take points from different regions
                int centralPoints = clusterSampleSize / 3;
                int randomPoints = clusterSampleSize / 3;
                int edgePoints = clusterSampleSize - centralPoints - randomPoints;

                // Central points (closest to centroid)
                for (int i = 0; i < centralPoints && i < sortedPoints.size(); i++) {
                    sampledPoints.add(sortedPoints.get(i));
                }

                // Random points
                List<Point> remainingPoints = new ArrayList<>(clusterPoints);
                remainingPoints.removeAll(sampledPoints);
                Collections.shuffle(remainingPoints);

                for (int i = 0; i < randomPoints && i < remainingPoints.size(); i++) {
                    sampledPoints.add(remainingPoints.get(i));
                }

                // Edge points (furthest from centroid)
                int size = sortedPoints.size();
                for (int i = 0; i < edgePoints && i < size; i++) {
                    Point p = sortedPoints.get(size - 1 - i);
                    if (!sampledPoints.contains(p)) {
                        sampledPoints.add(p);
                    }
                }
            }

            result.put(clusterId, sampledPoints);
        }

        return result;
    }

    /**
     * Calculate distance from point to centroid
     */
    private double distanceToPoint(Point p, double[] centroid) {
        List<Double> features = p.getFeatures();
        double sum = 0.0;

        for (int i = 0; i < features.size() && i < centroid.length; i++) {
            double diff = features.get(i) - centroid[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    /**
     * Calculate distance between two centroids
     */
    private double distanceBetweenCentroids(double[] c1, double[] c2) {
        double sum = 0.0;

        for (int i = 0; i < c1.length && i < c2.length; i++) {
            double diff = c1[i] - c2[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    /**
     * Get cached distance between two points, or calculate and cache it
     */
    private double getCachedDistance(Point p1, Point p2) {
        long id1 = p1.getId();
        long id2 = p2.getId();

        // Ensure id1 <= id2 for consistent caching
        if (id1 > id2) {
            long temp = id1;
            id1 = id2;
            id2 = temp;

            // Swap points too
            Point tempPoint = p1;
            p1 = p2;
            p2 = tempPoint;
        }

        // Check if distance is already cached
        Map<Long, Double> innerCache = distanceCache.computeIfAbsent(id1, k -> new HashMap<>());
        Double cachedDistance = innerCache.get(id2);

        if (cachedDistance != null) {
            return cachedDistance;
        }

        // Calculate and cache the distance
        double distance = Globals.DISTANCE_FUNCTION.calculate(p1, p2);
        if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
            distance = Math.sqrt(distance); // Convert squared distance to actual distance
        }

        innerCache.put(id2, distance);
        return distance;
    }

    /**
     * Calculate the Davies-Bouldin index
     */
    public double calculateDaviesBouldinIndex() {
        long startTime = System.nanoTime();

        int numClusters = clusterSizes.size();

        if (numClusters <= 1) {
            daviesBouldinIndex = Double.MAX_VALUE;
            return daviesBouldinIndex;
        }

        // Calculate cluster scatter (average distance to centroid)
        Map<Integer, Double> clusterScatter = new HashMap<>();

        for (int clusterId : clusterSizes.keySet()) {
            double[] centroid = centroids.get(clusterId);
            if (centroid == null)
                continue;

            double totalDistance = 0.0;
            List<Point> sampledPoints = sampledPointsByCluster.get(clusterId);

            for (Point p : sampledPoints) {
                double distance = distanceToPoint(p, centroid);
                totalDistance += distance;
            }

            double scatter = totalDistance / sampledPoints.size();
            clusterScatter.put(clusterId, scatter);
        }

        // Calculate Davies-Bouldin Index
        double dbSum = 0.0;

        for (int i : clusterSizes.keySet()) {
            double maxRatio = 0.0;
            double[] centroidI = centroids.get(i);
            Double scatterI = clusterScatter.get(i);

            if (centroidI == null || scatterI == null)
                continue;

            for (int j : clusterSizes.keySet()) {
                if (i != j) {
                    double[] centroidJ = centroids.get(j);
                    Double scatterJ = clusterScatter.get(j);

                    if (centroidJ == null || scatterJ == null)
                        continue;

                    // Calculate distance between centroids
                    double centroidDistance = distanceBetweenCentroids(centroidI, centroidJ);

                    // Calculate ratio of scatters to centroid distance
                    if (centroidDistance > 0) {
                        double ratio = (scatterI + scatterJ) / centroidDistance;
                        maxRatio = Math.max(maxRatio, ratio);
                    }
                }
            }

            dbSum += maxRatio;
        }

        daviesBouldinIndex = dbSum / numClusters;

        long endTime = System.nanoTime();
        System.out.println("Davies-Bouldin calculation completed in " +
                ((endTime - startTime) / 1_000_000) + " ms");

        return daviesBouldinIndex;
    }

    /**
     * Calculate the Silhouette Coefficient
     */
    public double calculateSilhouetteCoefficient() {
        long startTime = System.nanoTime();

        int numClusters = clusterSizes.size();

        if (numClusters <= 1) {
            silhouetteCoefficient = -1.0;
            return silhouetteCoefficient;
        }

        // Prepare sample points from other clusters for b(i) calculation
        Map<Integer, Map<Integer, List<Point>>> interClusterSamples = new HashMap<>();

        for (int clusterId : clusterSizes.keySet()) {
            interClusterSamples.put(clusterId, new HashMap<>());

            for (int otherClusterId : clusterSizes.keySet()) {
                if (clusterId != otherClusterId) {
                    List<Point> otherSampled = sampledPointsByCluster.get(otherClusterId);
                    if (otherSampled == null || otherSampled.isEmpty())
                        continue;

                    // Use at most 100 points from other cluster for efficiency
                    List<Point> limitedSample = otherSampled;
                    if (otherSampled.size() > 100) {
                        limitedSample = new ArrayList<>(otherSampled.subList(0, 100));
                        Collections.shuffle(limitedSample);
                    }

                    interClusterSamples.get(clusterId).put(otherClusterId, limitedSample);
                }
            }
        }

        // Calculate silhouette for each sampled point
        double totalSilhouette = 0.0;
        int validPoints = 0;

        for (int clusterId : sampledPointsByCluster.keySet()) {
            List<Point> clusterSamples = sampledPointsByCluster.get(clusterId);

            for (Point p : clusterSamples) {
                // Calculate a(i) - average distance to points in same cluster
                double aSum = 0.0;
                int aCount = 0;

                for (Point other : clusterSamples) {
                    if (p.equals(other))
                        continue;

                    double distance = getCachedDistance(p, other);
                    aSum += distance;
                    aCount++;
                }

                if (aCount == 0)
                    continue; // Skip if alone in cluster
                double a = aSum / aCount;

                // Calculate b(i) - average distance to points in nearest other cluster
                double minB = Double.MAX_VALUE;

                for (int otherCluster : interClusterSamples.get(clusterId).keySet()) {
                    double bSum = 0.0;
                    int bCount = 0;

                    for (Point other : interClusterSamples.get(clusterId).get(otherCluster)) {
                        double distance = getCachedDistance(p, other);
                        bSum += distance;
                        bCount++;
                    }

                    if (bCount > 0) {
                        double b = bSum / bCount;
                        minB = Math.min(minB, b);
                    }
                }

                if (minB == Double.MAX_VALUE)
                    continue; // Skip if no other clusters

                // Calculate silhouette
                double silhouette = (minB - a) / Math.max(a, minB);
                totalSilhouette += silhouette;
                validPoints++;
            }
        }

        silhouetteCoefficient = validPoints > 0 ? totalSilhouette / validPoints : -1.0;

        long endTime = System.nanoTime();
        System.out.println("Silhouette calculation completed in " +
                ((endTime - startTime) / 1_000_000) + " ms for " +
                validPoints + " points");

        return silhouetteCoefficient;
    }

    /**
     * Calculate the Calinski-Harabasz index
     * Higher values indicate better clustering
     */
    public double calculateCalinskiHarabaszIndex() {
        long startTime = System.nanoTime();

        int numClusters = clusterSizes.size();
        int totalPoints = 0;
        for (Integer size : clusterSizes.values()) {
            totalPoints += size;
        }

        if (numClusters <= 1 || totalPoints <= numClusters) {
            calinskiHarabaszIndex = 0.0;
            return calinskiHarabaszIndex;
        }

        // Calculate global centroid
        double[] globalCentroid = new double[dimensions];
        int globalCount = 0;

        for (Point p : dataset.getPoints()) {
            if (p.getLabel() == Point.NOISE) {
                continue;
            }

            List<Double> features = p.getFeatures();
            for (int i = 0; i < features.size() && i < dimensions; i++) {
                globalCentroid[i] += features.get(i);
            }
            globalCount++;
        }

        if (globalCount == 0) {
            calinskiHarabaszIndex = 0.0;
            return calinskiHarabaszIndex;
        }

        for (int i = 0; i < dimensions; i++) {
            globalCentroid[i] /= globalCount;
        }

        // Calculate between-cluster scatter (weighted sum of squared distances between
        // cluster centroids and global centroid)
        double betweenClusterScatter = 0.0;

        for (int clusterId : clusterSizes.keySet()) {
            double[] centroid = centroids.get(clusterId);
            if (centroid == null)
                continue;

            double squaredDistance = 0.0;
            for (int i = 0; i < dimensions; i++) {
                double diff = centroid[i] - globalCentroid[i];
                squaredDistance += diff * diff;
            }

            betweenClusterScatter += clusterSizes.get(clusterId) * squaredDistance;
        }

        // Calculate within-cluster scatter (sum of squared distances between points and
        // their cluster centroids)
        double withinClusterScatter = 0.0;

        for (int clusterId : clusterSizes.keySet()) {
            double[] centroid = centroids.get(clusterId);
            if (centroid == null)
                continue;

            List<Point> clusterPoints = sampledPointsByCluster.get(clusterId);
            if (clusterPoints == null || clusterPoints.isEmpty())
                continue;

            double clusterScatter = 0.0;
            for (Point p : clusterPoints) {
                double squaredDistance = 0.0;
                List<Double> features = p.getFeatures();

                for (int i = 0; i < features.size() && i < dimensions; i++) {
                    double diff = features.get(i) - centroid[i];
                    squaredDistance += diff * diff;
                }

                clusterScatter += squaredDistance;
            }

            // Scale by ratio of actual cluster size to sample size
            int actualSize = clusterSizes.get(clusterId);
            int sampleSize = clusterPoints.size();
            double scaleFactor = (double) actualSize / sampleSize;

            withinClusterScatter += clusterScatter * scaleFactor;
        }

        // Calculate CH index
        if (withinClusterScatter > 0 && numClusters > 1) {
            calinskiHarabaszIndex = (betweenClusterScatter / (numClusters - 1)) /
                    (withinClusterScatter / (totalPoints - numClusters));
        } else {
            calinskiHarabaszIndex = 0.0;
        }

        long endTime = System.nanoTime();
        System.out.println("Calinski-Harabasz calculation completed in " +
                ((endTime - startTime) / 1_000_000) + " ms");

        if (Double.isInfinite(calinskiHarabaszIndex) || Double.isNaN(calinskiHarabaszIndex)) {
            calinskiHarabaszIndex = 0.0;
        }

        return calinskiHarabaszIndex;
    }

    /**
     * Calculate the Dunn Index
     * Higher values indicate better clustering with compact and well-separated
     * clusters
     */
    public double calculateDunnIndex() {
        long startTime = System.nanoTime();

        int numClusters = clusterSizes.size();

        if (numClusters <= 1) {
            return 0.0; // Not applicable for single cluster
        }

        // Calculate minimum inter-cluster distance
        double minInterClusterDist = Double.MAX_VALUE;

        // Calculate maximum intra-cluster diameter
        double maxIntraClusterDist = 0.0;

        // For each cluster, calculate its diameter (maximum distance between any two
        // points)
        for (int clusterId : clusterSizes.keySet()) {
            List<Point> sampledPoints = sampledPointsByCluster.get(clusterId);
            if (sampledPoints == null || sampledPoints.isEmpty())
                continue;

            // Calculate intra-cluster diameter
            double diameter = calculateClusterDiameter(sampledPoints);
            maxIntraClusterDist = Math.max(maxIntraClusterDist, diameter);
        }

        // Calculate minimum distance between clusters
        for (int i : clusterSizes.keySet()) {
            List<Point> clusterI = sampledPointsByCluster.get(i);
            if (clusterI == null || clusterI.isEmpty())
                continue;

            for (int j : clusterSizes.keySet()) {
                if (i >= j)
                    continue; // Avoid duplicate pairs

                List<Point> clusterJ = sampledPointsByCluster.get(j);
                if (clusterJ == null || clusterJ.isEmpty())
                    continue;

                double distance = calculateMinInterClusterDistance(clusterI, clusterJ);
                minInterClusterDist = Math.min(minInterClusterDist, distance);
            }
        }

        // Calculate Dunn Index
        double dunnIndex = 0.0;
        if (maxIntraClusterDist > 0 && minInterClusterDist < Double.MAX_VALUE) {
            dunnIndex = minInterClusterDist / maxIntraClusterDist;
        }

        long endTime = System.nanoTime();
        System.out.println("Dunn Index calculation completed in " +
                ((endTime - startTime) / 1_000_000) + " ms");

        // If there's an issue with calculations, return a default value
        if (maxIntraClusterDist <= 0.0 || minInterClusterDist >= Double.MAX_VALUE ||
                minInterClusterDist <= 0.0) {
            return 0.0; // Default "bad" value when computation fails
        }

        // Check for numerical issues
        if (Double.isInfinite(dunnIndex) || Double.isNaN(dunnIndex)) {
            return 0.0;
        }

        return dunnIndex;
    }

    /**
     * Calculate the diameter of a cluster (maximum distance between any two points)
     */
    private double calculateClusterDiameter(List<Point> clusterPoints) {
        double maxDistance = 0.0;

        // For very large clusters, use sampling to estimate diameter
        if (clusterPoints.size() > 100) {
            // Use a random subset of point pairs for approximation
            Random random = new Random();
            int numSamples = Math.min(500, clusterPoints.size() * (clusterPoints.size() - 1) / 2);

            for (int s = 0; s < numSamples; s++) {
                int i = random.nextInt(clusterPoints.size());
                int j = random.nextInt(clusterPoints.size());
                if (i != j) {
                    double distance = getCachedDistance(clusterPoints.get(i), clusterPoints.get(j));
                    maxDistance = Math.max(maxDistance, distance);
                }
            }
        } else {
            // For smaller clusters, check all pairs
            for (int i = 0; i < clusterPoints.size(); i++) {
                for (int j = i + 1; j < clusterPoints.size(); j++) {
                    double distance = getCachedDistance(clusterPoints.get(i), clusterPoints.get(j));
                    maxDistance = Math.max(maxDistance, distance);
                }
            }
        }

        return maxDistance;
    }

    /**
     * Calculate the minimum distance between points from two different clusters
     */
    private double calculateMinInterClusterDistance(List<Point> cluster1, List<Point> cluster2) {
        double minDistance = Double.MAX_VALUE;

        // For large clusters, sample points to calculate distance
        int maxPointsToCheck = 50; // Limit number of points to check

        List<Point> points1 = cluster1;
        List<Point> points2 = cluster2;

        if (cluster1.size() > maxPointsToCheck) {
            points1 = new ArrayList<>(maxPointsToCheck);
            for (int i = 0; i < maxPointsToCheck; i++) {
                points1.add(cluster1.get(i * cluster1.size() / maxPointsToCheck));
            }
        }

        if (cluster2.size() > maxPointsToCheck) {
            points2 = new ArrayList<>(maxPointsToCheck);
            for (int i = 0; i < maxPointsToCheck; i++) {
                points2.add(cluster2.get(i * cluster2.size() / maxPointsToCheck));
            }
        }

        // Calculate minimum distance between the point samples
        for (Point p1 : points1) {
            for (Point p2 : points2) {
                double distance = getCachedDistance(p1, p2);
                minDistance = Math.min(minDistance, distance);
            }
        }

        return minDistance;
    }

    /**
     * Calculate Density-Based Cluster Validity (DBCV) index
     * Higher values indicate better clustering
     * Reference: Moulavi et al., "Density-Based Clustering Validation", SDM 2014
     */
    public double calculateDBCV() {
        long startTime = System.nanoTime();

        int numClusters = clusterSizes.size();

        if (numClusters <= 1) {
            return 0.0; // Not applicable for single cluster
        }

        // Pre-compute core distances for all clusters at once
        Map<Integer, Map<Point, Double>> clusterCoreDistances = new HashMap<>();

        // For small datasets, calculate all core distances upfront
        boolean precomputeAll = dataset.getPoints().size() < 5000;

        if (precomputeAll) {
            System.out.println("Precomputing all core distances for DBCV...");
            // Organize points by cluster for faster access
            Map<Integer, List<Point>> clusters = new HashMap<>();
            for (Point p : dataset.getPoints()) {
                int label = p.getLabel();
                if (label != Point.NOISE) {
                    clusters.computeIfAbsent(label, k -> new ArrayList<>()).add(p);
                }
            }

            // Calculate core distances for each cluster
            for (Map.Entry<Integer, List<Point>> entry : clusters.entrySet()) {
                clusterCoreDistances.put(entry.getKey(), calculateCoreDistances(entry.getValue()));
            }
        }

        // Calculate density sparseness (DC) for each cluster
        Map<Integer, Double> clusterDensitySparseness = new HashMap<>();
        double totalValidityIndex = 0.0;
        int validClusters = 0;

        // Process each cluster
        for (int clusterId : clusterSizes.keySet()) {
            List<Point> clusterPoints = sampledPointsByCluster.get(clusterId);
            if (clusterPoints == null || clusterPoints.isEmpty())
                continue;

            // Get or calculate core distances
            Map<Point, Double> coreDistances;
            if (precomputeAll) {
                coreDistances = clusterCoreDistances.get(clusterId);
            } else {
                coreDistances = calculateCoreDistances(clusterPoints);
                clusterCoreDistances.put(clusterId, coreDistances); // Store for later use
            }

            if (coreDistances == null || coreDistances.isEmpty())
                continue;

            // Calculate minimum spanning tree
            List<Edge> mst = calculateMinimumSpanningTree(clusterPoints, coreDistances);

            // Calculate density sparseness as the maximum edge weight in the MST
            double maxEdgeWeight = 0.0;
            for (Edge edge : mst) {
                maxEdgeWeight = Math.max(maxEdgeWeight, edge.getWeight());
            }

            clusterDensitySparseness.put(clusterId, maxEdgeWeight);
        }

        // Calculate cluster validity indices and density separations
        for (int i : clusterSizes.keySet()) {
            List<Point> clusterI = sampledPointsByCluster.get(i);
            if (clusterI == null || clusterI.isEmpty())
                continue;

            Double densitySparseness = clusterDensitySparseness.get(i);
            if (densitySparseness == null)
                continue;

            // Find minimum separation to any other cluster
            double minSeparation = Double.MAX_VALUE;

            for (int j : clusterSizes.keySet()) {
                if (i == j)
                    continue;

                List<Point> clusterJ = sampledPointsByCluster.get(j);
                if (clusterJ == null || clusterJ.isEmpty())
                    continue;

                Map<Point, Double> coreDistancesI = clusterCoreDistances.get(i);
                Map<Point, Double> coreDistancesJ = clusterCoreDistances.get(j);

                if (coreDistancesI == null || coreDistancesJ == null)
                    continue;

                // Calculate density separation
                double separation = calculateDensitySeparation(
                        clusterI, clusterJ, coreDistancesI, coreDistancesJ);

                minSeparation = Math.min(minSeparation, separation);
            }

            if (minSeparation == Double.MAX_VALUE)
                continue;

            // Calculate validity index for this cluster
            double validityIndex = (minSeparation - densitySparseness) /
                    Math.max(minSeparation, densitySparseness);

            totalValidityIndex += validityIndex;
            validClusters++;
        }

        // Calculate overall DBCV
        double dbcv = validClusters > 0 ? totalValidityIndex / validClusters : 0.0;

        long endTime = System.nanoTime();
        System.out.println("DBCV calculation completed in " +
                ((endTime - startTime) / 1_000_000) + " ms");

        return dbcv;
    }

    /**
     * Calculate core distances for all points in a cluster
     * Core distance is the distance to the kth nearest neighbor
     */
    private Map<Point, Double> calculateCoreDistances(List<Point> clusterPoints) {
        int k = Math.min(Globals.MIN_PTS - 1, clusterPoints.size() - 1);
        if (k <= 0)
            k = 1;

        // Limit points for large clusters
        List<Point> pointsToUse = clusterPoints;
        if (clusterPoints.size() > 200) {
            // Sample 200 points for very large clusters
            pointsToUse = new ArrayList<>(200);
            for (int i = 0; i < 200; i++) {
                pointsToUse.add(clusterPoints.get(i * clusterPoints.size() / 200));
            }
        }

        Map<Point, Double> coreDistances = new HashMap<>();

        // For each point in the sample
        for (Point p : pointsToUse) {
            // Use a fixed-size array to track k nearest neighbors
            double[] kNearest = new double[k];
            Arrays.fill(kNearest, Double.MAX_VALUE);

            for (Point other : pointsToUse) {
                if (p == other)
                    continue;

                double distance = getCachedDistance(p, other);

                // Add to k-nearest if smaller than current max
                if (distance < kNearest[k - 1]) {
                    kNearest[k - 1] = distance;
                    // Re-sort the array
                    Arrays.sort(kNearest);
                }
            }

            coreDistances.put(p, kNearest[k - 1]);
        }

        return coreDistances;
    }

    /**
     * Calculate minimum spanning tree for a cluster
     * using Prim's algorithm with density-based edge weights
     */
    private List<Edge> calculateMinimumSpanningTree(List<Point> points, Map<Point, Double> coreDistances) {
        if (points.size() <= 1) {
            return new ArrayList<>();
        }

        // Sample for large clusters - more aggressive sampling
        List<Point> pointsToUse = points;
        if (points.size() > 50) {
            pointsToUse = new ArrayList<>(50);
            for (int i = 0; i < 50; i++) {
                pointsToUse.add(points.get(i * points.size() / 50));
            }
        }

        int n = pointsToUse.size();
        List<Edge> mst = new ArrayList<>(n - 1);
        boolean[] visited = new boolean[n];

        // Start with first point
        visited[0] = true;
        int visitedCount = 1;

        // Store edges in array for faster access
        Edge[] bestEdges = new Edge[n];

        // Initialize best edges from first point
        Point firstPoint = pointsToUse.get(0);
        double firstCoreDistance = coreDistances.getOrDefault(firstPoint, 0.0);

        for (int i = 1; i < n; i++) {
            Point to = pointsToUse.get(i);
            double toCoreDistance = coreDistances.getOrDefault(to, 0.0);
            double weight = calculateDensityReachabilityDistance(
                    firstPoint, to, firstCoreDistance, toCoreDistance);
            bestEdges[i] = new Edge(0, i, weight);
        }

        // Main Prim's algorithm loop
        while (visitedCount < n) {
            // Find minimum-weight edge to unvisited vertex
            int minVertex = -1;
            double minWeight = Double.MAX_VALUE;

            for (int i = 0; i < n; i++) {
                if (!visited[i] && bestEdges[i] != null && bestEdges[i].getWeight() < minWeight) {
                    minWeight = bestEdges[i].getWeight();
                    minVertex = i;
                }
            }

            if (minVertex == -1)
                break; // No more reachable vertices

            // Add edge to MST
            mst.add(bestEdges[minVertex]);
            visited[minVertex] = true;
            visitedCount++;

            // Update best edges
            Point newPoint = pointsToUse.get(minVertex);
            double newCoreDistance = coreDistances.getOrDefault(newPoint, 0.0);

            for (int i = 0; i < n; i++) {
                if (!visited[i]) {
                    Point to = pointsToUse.get(i);
                    double toCoreDistance = coreDistances.getOrDefault(to, 0.0);
                    double weight = calculateDensityReachabilityDistance(
                            newPoint, to, newCoreDistance, toCoreDistance);

                    if (bestEdges[i] == null || weight < bestEdges[i].getWeight()) {
                        bestEdges[i] = new Edge(minVertex, i, weight);
                    }
                }
            }
        }

        return mst;
    }

    /**
     * Calculate density reachability distance between two points
     * Max of the two mutual reachability distances
     */
    private double calculateDensityReachabilityDistance(
            Point p1, Point p2, double coreDistance1, double coreDistance2) {
        double distance = getCachedDistance(p1, p2);
        double mutualReachability1 = Math.max(coreDistance1, distance);
        double mutualReachability2 = Math.max(coreDistance2, distance);

        return Math.max(mutualReachability1, mutualReachability2);
    }

    /**
     * Calculate density separation between two clusters
     * Minimum density reachability distance between points from different clusters
     */
    private double calculateDensitySeparation(
            List<Point> cluster1, List<Point> cluster2,
            Map<Point, Double> coreDistances1, Map<Point, Double> coreDistances2) {

        // More aggressive sampling for performance
        int maxSampleSize = 20; // Reduced from 50

        List<Point> points1 = samplePoints(cluster1, maxSampleSize);
        List<Point> points2 = samplePoints(cluster2, maxSampleSize);

        double minSeparation = Double.MAX_VALUE;

        // We only need to find the minimum, so we can use early stopping
        for (Point p1 : points1) {
            double coreDistance1 = coreDistances1.getOrDefault(p1, 0.0);

            for (Point p2 : points2) {
                double coreDistance2 = coreDistances2.getOrDefault(p2, 0.0);

                double distance = calculateDensityReachabilityDistance(
                        p1, p2, coreDistance1, coreDistance2);

                minSeparation = Math.min(minSeparation, distance);

                // Early stopping optimization - if we find a very small separation
                // we can return it immediately as it's likely to be close to the minimum
                if (minSeparation < 0.0001) {
                    return minSeparation;
                }
            }
        }

        return minSeparation;
    }

    /**
     * Helper method to sample points from a list
     */
    private List<Point> samplePoints(List<Point> points, int maxSampleSize) {
        if (points.size() <= maxSampleSize) {
            return points;
        }

        List<Point> sampled = new ArrayList<>(maxSampleSize);
        for (int i = 0; i < maxSampleSize; i++) {
            sampled.add(points.get(i * points.size() / maxSampleSize));
        }
        return sampled;
    }

    /**
     * Edge class for minimum spanning tree
     */
    private static class Edge {
        private final int fromIndex;
        private final int toIndex;
        private final double weight;

        public Edge(int fromIndex, int toIndex, double weight) {
            this.fromIndex = fromIndex;
            this.toIndex = toIndex;
            this.weight = weight;
        }

        public int getFromIndex() {
            return fromIndex;
        }

        public int getToIndex() {
            return toIndex;
        }

        public double getWeight() {
            return weight;
        }
    }

    /**
     * Calculate all metrics at once and return combined results
     */
    public Map<String, Double> calculateAllMetrics() {
        Map<String, Double> results = new HashMap<>();

        // Calculate Davies-Bouldin Index
        double dbi = calculateDaviesBouldinIndex();
        results.put("daviesBouldinIndex", dbi);

        // Calculate Silhouette Coefficient
        double sc = calculateSilhouetteCoefficient();
        results.put("silhouetteCoefficient", sc);

        // Calculate Calinski-Harabasz Index
        double chi = calculateCalinskiHarabaszIndex();
        results.put("calinskiHarabaszIndex", chi);

        // Calculate Dunn Index
        double di = calculateDunnIndex();
        results.put("dunnIndex", di);

        // Calculate DBCV
        double dbcv = calculateDBCV();
        results.put("dbcv", dbcv);

        return results;
    }
}
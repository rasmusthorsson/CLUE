package com.clueclust.util;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import com.clueclust.algorithm.ConcurrentLSHDBSCAN;
import com.clueclust.core.Dataset;
import com.clueclust.core.Point;

/**
 * Utility class for finding optimal DBSCAN parameters without ground truth
 * Based on techniques from "DBSCAN Revisited, Revisited: Why and How You Should
 * (Still) Use DBSCAN"
 */
public class OptimalParameterFinder {
    private final Dataset dataset;
    private final int dimensions;
    private final int numberOfPoints;

    // Parameter ranges
    private double[] epsilonValues;
    private int[] minPtsValues;

    private int[] hashTablesValues;
    private int[] hyperplanesValues;

    // Results storage
    private List<ParameterResult> results = new ArrayList<>();
    private List<Map.Entry<ParameterResult, Double>> sortedResults = new ArrayList<>();
    private Map<Integer, List<Double>> kDistances = new HashMap<>();

    // Metrics weights
    private double daviesBouldinWeight = 0.10;
    private double silhouetteWeight = 0.25;
    private double calinskiHarabaszWeight = 0.05;
    private double dunnWeight = 0.20;
    private double dbcvWeight = 0.40;

    /**
     * Constructor
     */
    public OptimalParameterFinder(Dataset dataset) {
        this.dataset = dataset;
        this.dimensions = dataset.getNumberOfDimensions();
        this.numberOfPoints = dataset.getPoints().size();

        // Initialize parameter ranges based on dataset characteristics
        initializeParameterRanges();
    }

    /**
     * Initialize parameter ranges using heuristics
     */
    private void initializeParameterRanges() {
        // Epsilon values - log scale values appropriate for the metric
        if (Globals.METRIC == Globals.Metric.ANGULAR) {
            // Angular distances range from 0 to 1
            epsilonValues = new double[] {
                    0.010
            };
        } else {
            // For Euclidean, scale with dimensionality
            // (distances increase with sqrt of dimensions)
            double scaleFactor = 1.0 / Math.sqrt(Math.max(1, dimensions));
            double maxDist = estimateMaxDistance() * 0.1; // 10% of max distance
            epsilonValues = logSpace(0.001 * scaleFactor, maxDist, 15);
        }

        // MinPts values based on dimensionality
        // Traditional recommendation: 2*dimensions to 4*dimensions
        int minPtsBase = Math.max(4, 2 * dimensions);
        minPtsValues = new int[] {
                minPtsBase / 2, // Lower than recommended
                minPtsBase, // Lower recommended
                minPtsBase * 2, // Recommended
                minPtsBase * 4 // Higher than recommended
        };

        // Lower dimensions (hourly data)
        minPtsValues = new int[] {
                // 5, // Very permissive
                10,
                15, // Moderate
                20, // Standard recommendation for your dimensions
                25, // Slightly stricter
                50, // 2*dimensions (traditional recommendation)
                100,
                //150,
                //200
        };

        // LSH parameters - scale with dimensionality and explore a range
        hashTablesValues = new int[] {
                // 5,
                10,
                //15,
                //20,
                50,
                100
        };

        // Hyperplanes per table
        hyperplanesValues = new int[] {
                //5,
                10,
                //20,
                50,
        };

    }

    /**
     * Create logarithmically spaced values between min and max
     */
    private double[] logSpace(double min, double max, int count) {
        double[] result = new double[count];
        double logMin = Math.log10(min);
        double logMax = Math.log10(max);
        double delta = (logMax - logMin) / (count - 1);

        for (int i = 0; i < count; i++) {
            double logVal = logMin + i * delta;
            result[i] = Math.pow(10, logVal);
        }

        return result;
    }

    /**
     * Generate k-distance plots for various k values
     * as recommended in Section 4.1 of the paper
     */
    public void calculateKDistancePlots() {
        System.out.println("Calculating k-distance plots...");

        // Enable exact distance calculation
        Globals.CALCULATE_EXACT_DISTANCES = true;

        // k values to calculate (original DBSCAN used k=4)
        // Using 4, 2*dim, and 4*dim as suggested in the paper
        int[] kValues = {
                4, // Original DBSCAN recommendation
                Math.max(4, 2 * dimensions), // Lower recommended for high dimensions
                Math.max(8, 4 * dimensions) // Higher recommendation for noisy data
        };

        // For very large datasets, sample points to make computation feasible
        List<Point> pointsToProcess = dataset.getPoints();
        if (numberOfPoints > 5000) {
            // Sample 10% of points or 5000, whichever is larger
            int sampleSize = Math.max(5000, numberOfPoints / 10);
            if (numberOfPoints > 300000) {
                sampleSize = 5000;
            }
            sampleSize = 5000; //9999 points took too long, remove this if needed

            pointsToProcess = sampleRandomPoints(sampleSize);
            System.out.println("Using " + sampleSize + " sample points for k-distance calculation");
        }

        // Calculate distances for each k
        for (int k : kValues) {
            List<Double> distances = calculateKDistances(pointsToProcess, k);
            kDistances.put(k, distances);
        }
        // After k-distance calculation is done
        Globals.CALCULATE_EXACT_DISTANCES = false;
    }

    /**
     * Calculate k-distances for all points
     */
    private List<Double> calculateKDistances(List<Point> points, int k) {
        System.out.println("Calculating " + k + "-distances for " + points.size() + " points...");
        List<Double> kDists = new ArrayList<>(points.size());

        for (Point p : points) {
            // Calculate distances to all other points
            List<Double> distances = new ArrayList<>(points.size() - 1);
            for (Point other : points) {
                if (p != other) {
                    double dist = Globals.DISTANCE_FUNCTION.calculate(p, other);
                    if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
                        dist = Math.sqrt(dist); // Convert squared distance to actual distance
                    }
                    distances.add(dist);
                }
            }

            // Sort distances and get the k-th distance
            Collections.sort(distances);
            if (distances.size() >= k) {
                kDists.add(distances.get(k - 1));
            } else {
                // If not enough neighbors, use the max distance
                kDists.add(distances.isEmpty() ? 0.0 : distances.get(distances.size() - 1));
            }
        }

        // Sort the k-distances for the plot
        Collections.sort(kDists);
        return kDists;
    }

    /**
     * Find the "elbow" or "knee" points in the k-distance plots
     * to recommend epsilon values
     */
    public List<Double> detectElbowPoints() {
        System.out.println("Detecting elbow points in k-distance plots...");
        List<Double> recommendedEpsilons = new ArrayList<>();

        for (Map.Entry<Integer, List<Double>> entry : kDistances.entrySet()) {
            int k = entry.getKey();
            List<Double> distances = entry.getValue();

            // Use the curvature method to find the elbow point
            Double elbowEpsilon = findElbowPoint(distances);
            if (elbowEpsilon != null) {
                recommendedEpsilons.add(elbowEpsilon);
                System.out.println("Recommended epsilon for k=" + k + ": " + elbowEpsilon);
            }
        }

        return recommendedEpsilons;
    }

    /**
     * Detect the elbow point in a sorted k-distance plot
     * using the maximum curvature method
     */
    private Double findElbowPoint(List<Double> sortedDistances) {
        if (sortedDistances.size() < 10) {
            return null; // Not enough points for meaningful analysis
        }

        int n = sortedDistances.size();
        double[] x = new double[n];
        double[] y = new double[n];

        // Normalize to [0,1] range for both axes
        double maxX = n - 1;
        double maxY = sortedDistances.get(n - 1);
        double minY = sortedDistances.get(0);
        double rangeY = maxY - minY;

        if (rangeY == 0) {
            return null; // All distances are the same
        }

        for (int i = 0; i < n; i++) {
            x[i] = i / maxX;
            y[i] = (sortedDistances.get(i) - minY) / rangeY;
        }

        // Calculate curvature at each point
        // Higher curvature indicates sharper bend (elbow)
        double maxCurvature = -1;
        int elbowIndex = -1;

        // Skip the first and last few points to avoid boundary effects
        int skip = Math.max(1, n / 20);

        for (int i = skip; i < n - skip; i++) {
            // Use points before and after for derivative calculation
            int prev = Math.max(0, i - skip);
            int next = Math.min(n - 1, i + skip);

            // First derivatives (using central difference)
            // double dx = x[next] - x[prev];
            double dy = y[next] - y[prev];

            // Second derivatives
            // double dx2 = x[next] + x[prev] - 2 * x[i];
            double dy2 = y[next] + y[prev] - 2 * y[i];

            // Curvature formula: |y'' * x' - x'' * y'| / (x'^2 + y'^2)^(3/2)
            // But we can simplify for our case since x values are uniform
            double curvature = Math.abs(dy2) / Math.pow(1 + dy * dy, 1.5);

            if (curvature > maxCurvature) {
                maxCurvature = curvature;
                elbowIndex = i;
            }
        }

        // Return the original (non-normalized) distance at the elbow point
        return elbowIndex >= 0 ? sortedDistances.get(elbowIndex) : null;
    }

    /**
     * Estimate maximum pairwise distance in the dataset
     * (using sampling for large datasets)
     */
    private double estimateMaxDistance() {
        System.out.println("Estimating maximum distance in dataset...");

        boolean originalFlag = Globals.CALCULATE_EXACT_DISTANCES;
        Globals.CALCULATE_EXACT_DISTANCES = true;

        List<Point> pointsToProcess = dataset.getPoints();
        int maxPairs = 100000; // Limit the number of pairs to examine

        if (pointsToProcess.size() > 5000) {
            // Sample points for large datasets
            int sampleSize = Math.min(pointsToProcess.size(),
                    (int) Math.sqrt(maxPairs * 2)); // sqrt(maxPairs*2) gives us about maxPairs total pairs
            pointsToProcess = sampleRandomPoints(sampleSize);
            System.out.println("Using " + sampleSize + " sample points for max distance estimation");
        }

        double maxDist = 0;
        for (int i = 0; i < pointsToProcess.size(); i++) {
            for (int j = i + 1; j < pointsToProcess.size(); j++) {
                double dist = Globals.DISTANCE_FUNCTION.calculate(
                        pointsToProcess.get(i), pointsToProcess.get(j));
                if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
                    dist = Math.sqrt(dist); // Convert squared distance to actual distance
                }
                maxDist = Math.max(maxDist, dist);
            }
        }
        // Restore original flag
        Globals.CALCULATE_EXACT_DISTANCES = originalFlag;

        System.out.println("Estimated maximum distance: " + maxDist);
        return maxDist;
    }

    /**
     * Random sampling of points
     */
    private List<Point> sampleRandomPoints(int sampleSize) {
        List<Point> allPoints = dataset.getPoints();
        if (sampleSize >= allPoints.size()) {
            return new ArrayList<>(allPoints);
        }

        // Reservoir sampling algorithm
        List<Point> sampledPoints = new ArrayList<>(sampleSize);
        Random random = new Random();

        // Add first sampleSize elements to the reservoir
        for (int i = 0; i < sampleSize; i++) {
            sampledPoints.add(allPoints.get(i));
        }

        // Replace elements with gradually decreasing probability
        for (int i = sampleSize; i < allPoints.size(); i++) {
            int j = random.nextInt(i + 1);
            if (j < sampleSize) {
                sampledPoints.set(j, allPoints.get(i));
            }
        }

        return sampledPoints;
    }

    /**
     * Run KDistance parameter finder, returns one k-distance value.
     */
    public void runParameterKDistanceFinder() {
        System.out.println("Running K-Distance parameter finder...");
        results.clear();
        List<Double> recommendedEpsilons = Globals.PARAM_OPT_EPSILONS;
        if (Globals.PARAM_OPT_EPSILONS.size() < 1) {
            recommendedEpsilons = detectElbowPoints();
            results.add(new ParameterResult(recommendedEpsilons.get(2), Math.max(20, dimensions/4), 10, 5));

        } else {
            for (Double eps : recommendedEpsilons) {
                results.add(new ParameterResult(eps, Math.max(20, dimensions/4), 10, 5));
            }
        }

        // Output current result
        for (ParameterResult r : results) {
            System.out.println("Result: " + r);
            Map.Entry<ParameterResult, Double> res = new AbstractMap.SimpleEntry<ParameterResult, Double>(r, 1.0);
            sortedResults.add(res);
        }
    }

    /**
     * Run parameter exploration to find optimal parameters
     * 
     * @param numberOfThreads   Number of threads to use
     * @param useAdaptiveSearch Whether to use adaptive grid search (true) or
     *                          original grid search (false)
     */
    public void runParameterExploration(int numberOfThreads, boolean useAdaptiveSearch) {
        // Reset results
        results.clear();
        List<Double> recommendedEpsilons = Globals.PARAM_OPT_EPSILONS;
        // Get recommended epsilon values from k-distance plots
        if (Globals.PARAM_OPT_EPSILONS.size() < 1) {
            recommendedEpsilons = detectElbowPoints();
        }
        // If no recommendations, use the initialized values
        if (!recommendedEpsilons.isEmpty()) {
            // Add some values below and above the recommendations
            //Set<Double> epsilonSet = new HashSet<>(recommendedEpsilons);
            Set<Double> epsilonSet = new HashSet<>(); //Testing with only k=4 eps
            if (Globals.PARAM_OPT_EPSILONS.size() < 1) {
                epsilonSet.add(recommendedEpsilons.get(2));
            } else {
                epsilonSet = new HashSet<>(recommendedEpsilons);
            }
            
            for (double eps : epsilonSet) {
                epsilonSet.add(eps * 0.5); // 50% below
                epsilonSet.add(eps * 0.75);
                epsilonSet.add(eps * 1.25); 
                //epsilonSet.add(eps * 0.9);
                //epsilonSet.add(eps * 1.1);
                //epsilonSet.add(eps * 0.25);
                epsilonSet.add(eps * 1.5);
                //epsilonSet.add(eps * 2.0); // 100% above
            }
            
            // Convert to array and sort
            epsilonValues = epsilonSet.stream().mapToDouble(d -> d).sorted().toArray();
        }

        if (useAdaptiveSearch) {
            System.out.println("Using adaptive parameter search strategy");
            runAdaptiveParameterExploration(numberOfThreads, 20, 30);
        } else {
            System.out.println("Using original exhaustive grid search strategy");
            runOriginalParameterExploration(numberOfThreads);
        }
    }

    /**
     * Original grid search implementation
     */
    private void runOriginalParameterExploration(int numberOfThreads) {
        System.out.println("Starting parameter grid search with " +
                epsilonValues.length + " epsilon values, " +
                minPtsValues.length + " minPts values, " +
                hashTablesValues.length + " hashTables values, and " +
                hyperplanesValues.length + " hyperplanes values");

        // Calculate total combinations for progress tracking
        int totalCombinations = epsilonValues.length * minPtsValues.length *
                hashTablesValues.length * hyperplanesValues.length;
        int currentCombination = 0;

        // For each parameter combination, run clustering and evaluate
        for (double epsilon : epsilonValues) {
            for (int minPts : minPtsValues) {
                for (int hashTables : hashTablesValues) {
                    for (int hyperplanes : hyperplanesValues) {
                        currentCombination++;
                        System.out.println(String.format("Testing combination %d/%d (%.1f%%)",
                                currentCombination, totalCombinations,
                                (float) currentCombination / totalCombinations * 100));

                        // Set global parameters
                        Globals.EPSILON_ORIGINAL = epsilon;
                        if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
                            Globals.EPSILON = epsilon * epsilon;
                        } else {
                            Globals.EPSILON = epsilon;
                        }
                        Globals.MIN_PTS = minPts;

                        System.out.println(String.format("Testing with: epsilon=%.6f, minPts=%d, " +
                                "hashTables=%d, hyperplanes=%d",
                                epsilon, minPts, hashTables, hyperplanes));

                        // Reset dataset for new clustering
                        dataset.resetData();

                        // Run clustering
                        ConcurrentLSHDBSCAN clueclust = new ConcurrentLSHDBSCAN(
                                dataset,
                                hashTables,
                                hyperplanes,
                                numberOfThreads,
                                100, // batches
                                true // benchmark
                        );

                        System.out.println("Running clustering...");
                        clueclust.performClustering();
                        dataset.enforceMinClusterSize();

                        // Evaluate results
                        ParameterResult result = evaluateClusteringResult(
                                epsilon, minPts, hashTables, hyperplanes);
                        results.add(result);

                        // Output current result
                        System.out.println("Result: " + result);
                    }
                }
            }
        }

        // Find and report optimal parameters
        findOptimalParameters();
    }

    /**
     * Original runParameterExploration method (backward compatibility)
     */
    public void runParameterExploration(int numberOfThreads) {
        // By default, use the original method
        runParameterExploration(numberOfThreads, false);
    }

    /**
     * Evaluate clustering results for a parameter combination
     */
    public ParameterResult evaluateClusteringResult(
            double epsilon, int minPts, int hashTables, int hyperplanes) {

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
        if (noisePercentage > 40.0) {
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

        if (maxClusterPercentage > 90.0) {
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
     * Find optimal parameters based on clustering quality
     */
    public void findOptimalParameters() {
        if (results.isEmpty()) {
            System.out.println("No results to analyze!");
            return;
        }

        System.out.println("\n===== OPTIMAL PARAMETER RECOMMENDATIONS =====");

        // Filter out degenerate results
        List<ParameterResult> validResults = new ArrayList<>();
        for (ParameterResult result : results) {
            if (!result.isDegenerate()) {
                validResults.add(result);
            }
        }

        System.out.println(validResults.size() + " out of " + results.size() +
                " parameter combinations produced valid clustering results");

        if (validResults.isEmpty()) {
            System.out.println("All parameter combinations produced degenerate results!");
            return;
        }

        // Find best by Davies-Bouldin Index (lower is better)
        ParameterResult bestDBI = validResults.stream()
                .min(Comparator.comparingDouble(ParameterResult::getDaviesBouldinIndex))
                .orElse(null);

        // Find best by Silhouette Coefficient (higher is better)
        ParameterResult bestSilhouette = validResults.stream()
                .max(Comparator.comparingDouble(ParameterResult::getSilhouetteCoefficient))
                .orElse(null);

        // Find best by Calinski-Harabasz Index (higher is better)
        ParameterResult bestCH = validResults.stream()
                .max(Comparator.comparingDouble(ParameterResult::getCalinskiHarabaszIndex))
                .orElse(null);

        // Find best by Dunn Index (higher is better)
        ParameterResult bestDI = validResults.stream()
                .max(Comparator.comparingDouble(ParameterResult::getDunnIndex))
                .orElse(null);

        // Find best by DBCV (higher is better)
        ParameterResult bestDBCV = validResults.stream()
                .max(Comparator.comparingDouble(ParameterResult::getDbcv))
                .orElse(null);

        // Print recommendations
        if (bestDBI != null) {
            System.out.println("\nBest by Davies-Bouldin Index (lower is better): " +
                    bestDBI.getDaviesBouldinIndex());
            printParameterDetails(bestDBI);
        }

        if (bestSilhouette != null) {
            System.out.println("\nBest by Silhouette Coefficient (higher is better): " +
                    bestSilhouette.getSilhouetteCoefficient());
            printParameterDetails(bestSilhouette);
        }

        if (bestCH != null) {
            System.out.println("\nBest by Calinski-Harabasz Index (higher is better): " +
                    bestCH.getCalinskiHarabaszIndex());
            printParameterDetails(bestCH);
        }

        if (bestDI != null) {
            System.out.println("\nBest by Dunn Index (higher is better): " +
                    bestDI.getDunnIndex());
            printParameterDetails(bestDI);
        }
        if (bestDI != null) {
            System.out.println("\nBest by DBCV (higher is better): " +
                    bestDI.getDbcv());
            printParameterDetails(bestDBCV);
        }

        findOptimalParametersWeighted(daviesBouldinWeight, silhouetteWeight, calinskiHarabaszWeight, dunnWeight,
                dbcvWeight);
    }

    /**
     * Find optimal parameters using weighted metrics
     * This allows customization of the importance of each metric
     */
    public void findOptimalParametersWeighted1(double dbiWeight, double silhouetteWeight, double chWeight,
            double diWeight, double dbcvWeight) {
        if (results.isEmpty()) {
            System.out.println("No results to analyze!");
            return;
        }

        System.out.println("\n===== WEIGHTED OPTIMAL PARAMETER RECOMMENDATIONS =====");
        System.out.println("Using weights: Davies-Bouldin=" + dbiWeight +
                ", Silhouette=" + silhouetteWeight +
                ", Calinski-Harabasz=" + chWeight +
                ", Dunn Index=" + diWeight +
                ", DBCV=" + dbcvWeight);

        // Filter out degenerate results
        List<ParameterResult> validResults = new ArrayList<>();
        for (ParameterResult result : results) {
            if (!result.isDegenerate()) {
                validResults.add(result);
            }
        }

        System.out.println(validResults.size() + " out of " + results.size() +
                " parameter combinations produced valid clustering results");

        if (validResults.isEmpty()) {
            System.out.println("All parameter combinations produced degenerate results!");
            System.out.println("Try with different parameter ranges.");
            return;
        }

        // Find range of values for each metric for normalization
        double minDBI = Double.MAX_VALUE;
        double maxDBI = Double.MIN_VALUE;
        double minSilhouette = Double.MAX_VALUE;
        double maxSilhouette = Double.MIN_VALUE;
        double minCH = Double.MAX_VALUE;
        double maxCH = Double.MIN_VALUE;
        double minDunn = Double.MAX_VALUE;
        double maxDunn = Double.MIN_VALUE;
        double minDBCV = Double.MAX_VALUE;
        double maxDBCV = Double.MIN_VALUE;

        for (ParameterResult result : validResults) {
            minDBI = Math.min(minDBI, result.getDaviesBouldinIndex());
            maxDBI = Math.max(maxDBI, result.getDaviesBouldinIndex());

            minSilhouette = Math.min(minSilhouette, result.getSilhouetteCoefficient());
            maxSilhouette = Math.max(maxSilhouette, result.getSilhouetteCoefficient());

            minCH = Math.min(minCH, result.getCalinskiHarabaszIndex());
            maxCH = Math.max(maxCH, result.getCalinskiHarabaszIndex());

            minDunn = Math.min(minDunn, result.getDunnIndex());
            maxDunn = Math.max(maxDunn, result.getDunnIndex());

            minDBCV = Math.min(minDBCV, result.getDbcv());
            maxDBCV = Math.max(maxDBCV, result.getDbcv());
        }

        // Calculate weighted score for each result
        ParameterResult bestWeighted = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        double dbiRange = maxDBI - minDBI;
        double silhouetteRange = maxSilhouette - minSilhouette;
        double chRange = maxCH - minCH;
        double dunnRange = maxDunn - minDunn;
        double dbcvRange = maxDBCV - minDBCV;

        for (ParameterResult result : validResults) {
            // Normalize each metric to [0,1] range
            // NOTE: DBI is inverted (lower is better)
            double normDBI = (maxDBI - result.getDaviesBouldinIndex()) / dbiRange;
            double normSilhouette = (result.getSilhouetteCoefficient() - minSilhouette) / silhouetteRange;
            double normCH = (result.getCalinskiHarabaszIndex() - minCH) / chRange;
            double normDunn = (result.getDunnIndex() - minDunn) / dunnRange;
            double normDBCV = (result.getDbcv() - minDBCV) / dbcvRange;

            normDBI = Double.isNaN(normDBI) || Double.isInfinite(normDBI) ? 0.5 : Math.max(0, Math.min(1, normDBI));
            normDBCV = Double.isNaN(normSilhouette) || Double.isInfinite(normSilhouette) ? 0.5
                    : Math.max(0, Math.min(1, normSilhouette));
            normDBCV = Double.isNaN(normCH) || Double.isInfinite(normCH) ? 0.5 : Math.max(0, Math.min(1, normCH));
            normDBCV = Double.isNaN(normDunn) || Double.isInfinite(normDunn) ? 0.5 : Math.max(0, Math.min(1, normDunn));
            normDBCV = Double.isNaN(normDBCV) || Double.isInfinite(normDBCV) ? 0.5 : Math.max(0, Math.min(1, normDBCV));

            // Calculate weighted score - higher is better for all normalized metrics
            double score = (dbiWeight * normDBI) +
                    (silhouetteWeight * normSilhouette) +
                    (chWeight * normCH) +
                    (dunnWeight * normDunn) +
                    (dbcvWeight * normDBCV);

            if (score > bestScore) {
                bestScore = score;
                bestWeighted = result;
            }
        }

        System.out.println("DBI range: " + minDBI + " to " + maxDBI);
        System.out.println("Silhouette range: " + minSilhouette + " to " + maxSilhouette);
        System.out.println("CH range: " + minCH + " to " + maxCH);
        System.out.println("Dunn range: " + minDunn + " to " + maxDunn);
        System.out.println("DBCV range: " + minDBCV + " to " + maxDBCV);

        // Print the weighted best result
        if (bestWeighted != null) {
            System.out.println("\nBest by Weighted Metric Score: " + String.format("%.4f", bestScore)
                    + " in range [0,1], 1 is better");
            printParameterDetails(bestWeighted);
        }
    }

    /**
     * Find optimal parameters using a rank-based approach
     */
    public void findOptimalParametersWeighted(double dbiWeight, double silhouetteWeight,
            double chWeight, double dunnWeight, double dbcvWeight) {

        if (results.isEmpty()) {
            System.out.println("No results to analyze!");
            return;
        }

        // Filter out degenerate results
        List<ParameterResult> validResults = new ArrayList<>();
        for (ParameterResult result : results) {
            if (!result.isDegenerate()) {
                validResults.add(result);
            }
        }

        System.out.println("\n===== RANK-BASED WEIGHTED OPTIMAL PARAMETER RECOMMENDATIONS =====");
        System.out.println("Using weights: Davies-Bouldin=" + dbiWeight +
                ", Silhouette=" + silhouetteWeight +
                ", Calinski-Harabasz=" + chWeight +
                ", Dunn Index=" + dunnWeight +
                ", DBCV=" + dbcvWeight);

        System.out.println(validResults.size() + " out of " + results.size() +
                " parameter combinations produced valid clustering results");

        // Create sorted lists for each metric
        List<ParameterResult> rankedByDBI = new ArrayList<>(validResults);
        List<ParameterResult> rankedBySilhouette = new ArrayList<>(validResults);
        List<ParameterResult> rankedByCH = new ArrayList<>(validResults);
        List<ParameterResult> rankedByDunn = new ArrayList<>(validResults);
        List<ParameterResult> rankedByDBCV = new ArrayList<>(validResults);

        // Sort by each metric (for DBI lower is better, for others higher is better)
        rankedByDBI.sort(Comparator.comparingDouble(ParameterResult::getDaviesBouldinIndex));
        rankedBySilhouette.sort(Comparator.comparingDouble(ParameterResult::getSilhouetteCoefficient).reversed());
        rankedByCH.sort(Comparator.comparingDouble(ParameterResult::getCalinskiHarabaszIndex).reversed());
        rankedByDunn.sort(Comparator.comparingDouble(ParameterResult::getDunnIndex).reversed());
        rankedByDBCV.sort(Comparator.comparingDouble(ParameterResult::getDbcv).reversed());

        // Print range for each metric for informational purposes
        if (!rankedByDBI.isEmpty()) {
            System.out.println("DBI range: " + rankedByDBI.get(0).getDaviesBouldinIndex() +
                    " to " + rankedByDBI.get(rankedByDBI.size() - 1).getDaviesBouldinIndex());
        }
        if (!rankedBySilhouette.isEmpty()) {
            System.out.println("Silhouette range: "
                    + rankedBySilhouette.get(rankedBySilhouette.size() - 1).getSilhouetteCoefficient() +
                    " to " + rankedBySilhouette.get(0).getSilhouetteCoefficient());
        }
        if (!rankedByCH.isEmpty()) {
            System.out.println("CH range: " + rankedByCH.get(rankedByCH.size() - 1).getCalinskiHarabaszIndex() +
                    " to " + rankedByCH.get(0).getCalinskiHarabaszIndex());
        }
        if (!rankedByDunn.isEmpty()) {
            System.out.println("Dunn range: " + rankedByDunn.get(rankedByDunn.size() - 1).getDunnIndex() +
                    " to " + rankedByDunn.get(0).getDunnIndex());
        }
        if (!rankedByDBCV.isEmpty()) {
            System.out.println("DBCV range: " + rankedByDBCV.get(rankedByDBCV.size() - 1).getDbcv() +
                    " to " + rankedByDBCV.get(0).getDbcv());
        }

        // Calculate weighted rank score for each result
        Map<ParameterResult, Double> weightedRanks = new HashMap<>();

        for (ParameterResult result : validResults) {
            // Get rank for each metric (0 is best)
            int dbiRank = rankedByDBI.indexOf(result);
            int silhouetteRank = rankedBySilhouette.indexOf(result);
            int chRank = rankedByCH.indexOf(result);
            int dunnRank = rankedByDunn.indexOf(result);
            int dbcvRank = rankedByDBCV.indexOf(result);

            // Calculate weighted rank (lower is better)
            double weightedRank = (dbiWeight * dbiRank) +
                    (silhouetteWeight * silhouetteRank) +
                    (chWeight * chRank) +
                    (dunnWeight * dunnRank) +
                    (dbcvWeight * dbcvRank);

            weightedRanks.put(result, weightedRank);
        }

        // Find the result with the lowest weighted rank
        ParameterResult bestResult = null;
        double bestRank = Double.MAX_VALUE;

        for (Map.Entry<ParameterResult, Double> entry : weightedRanks.entrySet()) {
            if (entry.getValue() < bestRank) {
                bestRank = entry.getValue();
                bestResult = entry.getKey();
            }
        }

        // Print the top 3 results
        List<Map.Entry<ParameterResult, Double>> sortedResults = new ArrayList<>(weightedRanks.entrySet());
        sortedResults.sort(Map.Entry.comparingByValue());
        this.sortedResults = sortedResults;

        if (bestResult != null) {
            System.out.println("\nTop 3 parameter combinations by weighted rank:");

            int count = 0;
            for (Map.Entry<ParameterResult, Double> entry : sortedResults) {
                if (count >= 3)
                    break;

                System.out.println(
                        "\nRank #" + (count + 1) + " (score: " + String.format("%.2f", entry.getValue()) + ")");
                printParameterDetails(entry.getKey());
                count++;
            }

            System.out.println("\nBest overall parameter combination:");
            printParameterDetails(bestResult);
        }
    }

    /**
     * Method to print parameter details
     */
    private void printParameterDetails(ParameterResult result) {
        System.out.println("  Epsilon: " + result.getEpsilon());
        System.out.println("  MinPts: " + result.getMinPts());
        System.out.println("  Hash Tables: " + result.getHashTables());
        System.out.println("  Hyperplanes per Table: " + result.getHyperplanes());
        System.out.println("  Clusters found: " + result.getNumClusters());
        System.out.println("  Noise points: " + result.getNoiseCount() +
                " (" + String.format("%.1f", (double) result.getNoiseCount() /
                        (result.getNoiseCount() + result.getClusteredPoints()) * 100)
                + "%)");
        System.out.println("  Clustered points: " + result.getClusteredPoints());
        System.out.println("  Davies-Bouldin Index: " + String.format("%.4f", result.getDaviesBouldinIndex()));
        System.out.println("  Silhouette Coefficient: " + String.format("%.4f", result.getSilhouetteCoefficient()));
        System.out.println("  Calinski-Harabasz Index: " + String.format("%.4f", result.getCalinskiHarabaszIndex()));
        System.out.println("  Dunn Index: " + String.format("%.4f", result.getDunnIndex()));
        System.out.println("  DBCV: " + String.format("%.4f", result.getDbcv()));
    }

        /**
     * Method to write parameter details
     */
    public void saveSortedParametersAsCSV(String fd) throws IOException {
        if (this.sortedResults.isEmpty()) {
            System.out.println("No results found");
            return;
        }
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fd))) {
            writer.write("Rank,Epsilon,MinPts,HashTables,Hyperplanes,Clusters,Noise,Clustered,DBI,Silhouette,CH,Dunn,DBCV");
            int rank = 1;
            for (Map.Entry<ParameterResult, Double> result : this.sortedResults) {
                writer.write("\n");
                writer.write(Integer.toString(rank));
                writer.write(",");
                writer.write(Double.toString(result.getKey().getEpsilon()));
                writer.write(",");
                writer.write(Integer.toString(result.getKey().getMinPts()));
                writer.write(",");
                writer.write(Integer.toString(result.getKey().getHashTables()));
                writer.write(",");
                writer.write(Integer.toString(result.getKey().getHyperplanes()));
                writer.write(",");
                writer.write(Integer.toString(result.getKey().getNumClusters()));
                writer.write(",");
                writer.write(Integer.toString(result.getKey().getNoiseCount()));
                writer.write(",");
                writer.write(Integer.toString(result.getKey().getClusteredPoints()));
                writer.write(",");
                String DBIString = Double.toString(result.getKey().getDaviesBouldinIndex());
                writer.write(DBIString.substring(0, Math.min(DBIString.length(), 6)));
                writer.write(",");
                writer.write(Double.toString(result.getKey().getSilhouetteCoefficient()));
                writer.write(",");
                writer.write(Double.toString(result.getKey().getCalinskiHarabaszIndex()));
                writer.write(",");
                writer.write(Double.toString(result.getKey().getDunnIndex()));
                writer.write(",");
                writer.write(Double.toString(result.getKey().getDbcv()));
                rank++;
            }
        }
    }

    /**
     * Save k-distance plots to a CSV file
     */
    public void saveKDistancePlots(String fileName) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            // Write header
            writer.write("index");
            for (int k : kDistances.keySet()) {
                writer.write("," + k + "-distance");
            }
            writer.newLine();

            // Find the maximum length of any distance list
            int maxLength = kDistances.values().stream()
                    .mapToInt(List::size)
                    .max()
                    .orElse(0);

            // Write distances
            for (int i = 0; i < maxLength; i++) {
                writer.write(String.valueOf(i));

                for (int k : kDistances.keySet()) {
                    List<Double> distances = kDistances.get(k);
                    writer.write(",");
                    if (i < distances.size()) {
                        writer.write(String.valueOf(distances.get(i)));
                    }
                }

                writer.newLine();
            }
        }
        System.out.println("K-distance plots saved to " + fileName);
    }

    /**
     * Save parameter exploration results to a CSV file
     */
    public void saveResults(String fileName) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            // Write header
            writer.write("Epsilon,MinPts,HashTables,Hyperplanes,NumClusters,NoiseCount,ClusteredPoints," +
                    "DaviesBouldin,Silhouette,CalinskiHarabasz,DunnIndex,DBCV,Degenerate,DegenerateReason\n");

            // Write all results
            for (ParameterResult result : results) {
                writer.write(String.format("%.6f,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%b,%s\n",
                        result.getEpsilon(),
                        result.getMinPts(),
                        result.getHashTables(),
                        result.getHyperplanes(),
                        result.getNumClusters(),
                        result.getNoiseCount(),
                        result.getClusteredPoints(),
                        result.getDaviesBouldinIndex(),
                        result.getSilhouetteCoefficient(),
                        result.getCalinskiHarabaszIndex(),
                        result.getDunnIndex(),
                        result.getDbcv(),
                        result.isDegenerate(),
                        result.getDegenerateReason()));
            }
        }
        System.out.println("Parameter exploration results saved to " + fileName);
    }

    /**
     * Run adaptive parameter exploration to efficiently find optimal parameters
     * 
     * @param numberOfThreads            Number of threads to use
     * @param maxIterations              Maximum number of search iterations
     * @param maxEvaluationsPerIteration Maximum parameter combinations to evaluate
     *                                   per iteration
     */
    public void runAdaptiveParameterExploration(int numberOfThreads, int maxIterations,
            int maxEvaluationsPerIteration) {
        // Reset results
        results.clear();

        System.out.println("Starting adaptive parameter exploration...");

        // Track all evaluated results
        List<ParameterResult> allResults = new ArrayList<>();

        // Initial parameter ranges
        double minEpsilon = Arrays.stream(epsilonValues).min().orElse(0.001);
        double maxEpsilon = Arrays.stream(epsilonValues).max().orElse(0.1);
        int minMinPts = Arrays.stream(minPtsValues).min().orElse(5);
        int maxMinPts = Arrays.stream(minPtsValues).max().orElse(100);
        int minHashTables = Arrays.stream(hashTablesValues).min().orElse(5);
        int maxHashTables = Arrays.stream(hashTablesValues).max().orElse(50);
        int minHyperplanes = Arrays.stream(hyperplanesValues).min().orElse(5);
        int maxHyperplanes = Arrays.stream(hyperplanesValues).max().orElse(20);

        // Start with initial grid sampling
        List<ParameterCombination> currentCombinations = generateInitialCombinations(
                minEpsilon, maxEpsilon, minMinPts, maxMinPts,
                minHashTables, maxHashTables, minHyperplanes, maxHyperplanes);

        ParameterResult bestResult = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            System.out.println("\n=== Iteration " + (iteration + 1) + " of " + maxIterations + " ===");

            // Limit number of evaluations per iteration
            List<ParameterCombination> iterationCombinations = currentCombinations;
            if (iterationCombinations.size() > maxEvaluationsPerIteration) {
                iterationCombinations = iterationCombinations.subList(0, maxEvaluationsPerIteration);
            }

            System.out.println("Evaluating " + iterationCombinations.size() + " parameter combinations...");

            // Evaluate current parameter combinations
            List<ParameterResult> iterationResults = evaluateParameterCombinations(
                    iterationCombinations, numberOfThreads);

            // Add to overall results
            results.addAll(iterationResults);
            allResults.addAll(iterationResults);

            // Find best result in this iteration
            ParameterResult iterationBest = findBestResult(iterationResults);
            if (iterationBest != null) {
                double iterationScore = calculateScore(iterationBest);
                System.out.println("Best result in iteration " + (iteration + 1) + ": " +
                        String.format("Score: %.4f, Epsilon: %.6f, MinPts: %d, HashTables: %d, Hyperplanes: %d",
                                iterationScore, iterationBest.getEpsilon(), iterationBest.getMinPts(),
                                iterationBest.getHashTables(), iterationBest.getHyperplanes()));

                // Update best overall if improved
                if (bestResult == null || iterationScore > bestScore) {
                    bestResult = iterationBest;
                    bestScore = iterationScore;
                    System.out.println("New best overall result found!");
                } else {
                    System.out.println("No improvement over previous best score: " + bestScore);
                }
            }

            // Early stopping if no valid results in this iteration
            if (iterationResults.isEmpty() ||
                    iterationResults.stream().allMatch(ParameterResult::isDegenerate)) {
                System.out.println("No valid results in this iteration. Stopping search.");
                break;
            }

            // Generate new combinations to explore for next iteration
            currentCombinations = generateNextCombinations(
                    allResults,
                    minEpsilon, maxEpsilon, minMinPts, maxMinPts,
                    minHashTables, maxHashTables, minHyperplanes, maxHyperplanes,
                    maxEvaluationsPerIteration);

            // If we couldn't generate new combinations, stop
            if (currentCombinations.isEmpty()) {
                System.out.println("No new parameter combinations to explore. Stopping search.");
                break;
            }
        }

        System.out.println("\n=== Adaptive Parameter Search Complete ===");
        System.out.println("Total evaluated combinations: " + allResults.size());

        if (bestResult != null) {
            System.out.println("\nBest parameters found:");
            printParameterDetails(bestResult);
        } else {
            System.out.println("\nNo valid parameter combination found.");
        }
    }

    /**
     * Generate initial parameter combinations for exploration
     */
    private List<ParameterCombination> generateInitialCombinations(
            double minEpsilon, double maxEpsilon, int minMinPts, int maxMinPts,
            int minHashTables, int maxHashTables, int minHyperplanes, int maxHyperplanes) {

        List<ParameterCombination> combinations = new ArrayList<>();

        // Include all original grid points
        for (double epsilon : epsilonValues) {
            for (int minPts : minPtsValues) {
                for (int hashTables : hashTablesValues) {
                    for (int hyperplanes : hyperplanesValues) {
                        combinations.add(new ParameterCombination(
                                epsilon, minPts, hashTables, hyperplanes));
                    }
                }
            }
        }

        // If we have too many initial combinations, sample a reasonable number
        if (combinations.size() > 20) {
            // Take extremes and some random samples
            List<ParameterCombination> sampledCombinations = new ArrayList<>();

            // Get corners of parameter space
            sampledCombinations.add(new ParameterCombination(minEpsilon, minMinPts, minHashTables, minHyperplanes));
            sampledCombinations.add(new ParameterCombination(maxEpsilon, minMinPts, minHashTables, minHyperplanes));
            sampledCombinations.add(new ParameterCombination(minEpsilon, maxMinPts, minHashTables, minHyperplanes));
            sampledCombinations.add(new ParameterCombination(minEpsilon, minMinPts, maxHashTables, minHyperplanes));
            sampledCombinations.add(new ParameterCombination(minEpsilon, minMinPts, minHashTables, maxHyperplanes));
            sampledCombinations.add(new ParameterCombination(maxEpsilon, maxMinPts, maxHashTables, maxHyperplanes));

            // Add some random samples from the original combinations
            Random random = new Random();
            while (sampledCombinations.size() < 20 && !combinations.isEmpty()) {
                int index = random.nextInt(combinations.size());
                sampledCombinations.add(combinations.get(index));
                combinations.remove(index); // Remove to avoid duplicates
            }

            return sampledCombinations;
        }

        return combinations;
    }

    /**
     * Evaluate a list of parameter combinations
     */
    private List<ParameterResult> evaluateParameterCombinations(
            List<ParameterCombination> combinations, int numberOfThreads) {

        List<ParameterResult> results = new ArrayList<>();

        for (ParameterCombination combo : combinations) {
            // Set global parameters
            Globals.EPSILON_ORIGINAL = combo.getEpsilon();
            if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
                Globals.EPSILON = combo.getEpsilon() * combo.getEpsilon();
            } else {
                Globals.EPSILON = combo.getEpsilon();
            }
            Globals.MIN_PTS = combo.getMinPts();

            System.out.println("Testing: epsilon=" + String.format("%.6f", combo.getEpsilon()) +
                    ", minPts=" + combo.getMinPts() +
                    ", hashTables=" + combo.getHashTables() +
                    ", hyperplanes=" + combo.getHyperplanes());

            // Reset dataset for new clustering
            dataset.resetData();

            // Run clustering
            ConcurrentLSHDBSCAN clueclust = new ConcurrentLSHDBSCAN(
                    dataset,
                    combo.getHashTables(),
                    combo.getHyperplanes(),
                    numberOfThreads,
                    100, // batches
                    true // benchmark
            );

            try {
                clueclust.performClustering();
                dataset.enforceMinClusterSize();

                // Evaluate results
                ParameterResult result = evaluateClusteringResult(
                        combo.getEpsilon(), combo.getMinPts(),
                        combo.getHashTables(), combo.getHyperplanes());

                results.add(result);
                System.out.println("Result: " + result);
            } catch (Exception e) {
                System.err.println("Error evaluating combination: " + e.getMessage());
                // Create a degenerate result for failed runs
                results.add(new ParameterResult(
                        combo.getEpsilon(), combo.getMinPts(),
                        combo.getHashTables(), combo.getHyperplanes(),
                        0, 0, 0, Double.MAX_VALUE, -1.0, 0.0, 0.0, 0.0,
                        true, "Exception: " + e.getMessage()));
            }
        }

        return results;
    }

    /**
     * Generate new parameter combinations for next iteration based on previous
     * results
     */
    private List<ParameterCombination> generateNextCombinations(
            List<ParameterResult> previousResults,
            double minEpsilon, double maxEpsilon, int minMinPts, int maxMinPts,
            int minHashTables, int maxHashTables, int minHyperplanes, int maxHyperplanes,
            int maxCombinations) {

        List<ParameterCombination> combinations = new ArrayList<>();

        // Filter out degenerate results
        List<ParameterResult> validResults = previousResults.stream()
                .filter(r -> !r.isDegenerate())
                .collect(Collectors.toList());

        if (validResults.isEmpty()) {
            System.out.println("No valid results to base next exploration on!");
            return combinations;
        }

        // Sort by score
        validResults.sort((r1, r2) -> Double.compare(calculateScore(r2), calculateScore(r1)));

        // Take top N results
        int topN = Math.min(5, validResults.size());
        List<ParameterResult> topResults = validResults.subList(0, topN);

        // 1. EXPLOITATION: Generate points near the top results
        for (ParameterResult result : topResults) {
            // Local search with small variations
            for (int i = 0; i < 3; i++) {
                // ±10-20% variation
                double epsilonMultiplier = 0.8 + Math.random() * 0.4;
                double minPtsMultiplier = 0.8 + Math.random() * 0.4;
                double hashTablesMultiplier = 0.8 + Math.random() * 0.4;
                double hyperplanesMultiplier = 0.8 + Math.random() * 0.4;

                double newEpsilon = result.getEpsilon() * epsilonMultiplier;
                int newMinPts = (int) Math.max(5, Math.round(result.getMinPts() * minPtsMultiplier));
                int newHashTables = (int) Math.max(1, Math.round(result.getHashTables() * hashTablesMultiplier));
                int newHyperplanes = (int) Math.max(1, Math.round(result.getHyperplanes() * hyperplanesMultiplier));

                // Ensure we're within allowed ranges
                newEpsilon = Math.max(minEpsilon, Math.min(maxEpsilon, newEpsilon));
                newMinPts = Math.max(minMinPts, Math.min(maxMinPts, newMinPts));
                newHashTables = Math.max(minHashTables, Math.min(maxHashTables, newHashTables));
                newHyperplanes = Math.max(minHyperplanes, Math.min(maxHyperplanes, newHyperplanes));

                ParameterCombination newCombo = new ParameterCombination(
                        newEpsilon, newMinPts, newHashTables, newHyperplanes);

                // Only add if we haven't evaluated this combination before
                if (!hasBeenEvaluated(newCombo, previousResults) && !combinations.contains(newCombo)) {
                    combinations.add(newCombo);
                }
            }
        }

        // 2. EXPLORATION: Add some completely new points
        // Calculate parameter value performance
        Map<Double, Double> epsilonScores = new HashMap<>();
        Map<Integer, Double> minPtsScores = new HashMap<>();
        Map<Integer, Double> hashTablesScores = new HashMap<>();
        Map<Integer, Double> hyperplanesScores = new HashMap<>();

        // For each parameter value, track best score achieved
        for (ParameterResult result : validResults) {
            double score = calculateScore(result);

            epsilonScores.put(result.getEpsilon(),
                    Math.max(score, epsilonScores.getOrDefault(result.getEpsilon(), Double.NEGATIVE_INFINITY)));

            minPtsScores.put(result.getMinPts(),
                    Math.max(score, minPtsScores.getOrDefault(result.getMinPts(), Double.NEGATIVE_INFINITY)));

            hashTablesScores.put(result.getHashTables(),
                    Math.max(score, hashTablesScores.getOrDefault(result.getHashTables(), Double.NEGATIVE_INFINITY)));

            hyperplanesScores.put(result.getHyperplanes(),
                    Math.max(score, hyperplanesScores.getOrDefault(result.getHyperplanes(), Double.NEGATIVE_INFINITY)));
        }

        // Add exploration points until we reach desired number
        Random random = new Random();
        int attempts = 0;
        while (combinations.size() < maxCombinations && attempts < 100) {
            attempts++;

            // Select epsilon with bias toward better performing values
            double newEpsilon;
            if (random.nextDouble() < 0.7 && !epsilonScores.isEmpty()) {
                // 70% chance to use biased selection
                newEpsilon = selectParameterValueByPerformance(epsilonScores);
                // Add random variation
                newEpsilon *= 0.7 + random.nextDouble() * 0.6; // ±30% variation
            } else {
                // 30% chance of purely random value
                newEpsilon = minEpsilon + random.nextDouble() * (maxEpsilon - minEpsilon);
            }

            // Select minPts with bias
            int newMinPts;
            if (random.nextDouble() < 0.7 && !minPtsScores.isEmpty()) {
                newMinPts = (int) selectParameterValueByPerformance(minPtsScores);
                newMinPts = (int) Math.round(newMinPts * (0.7 + random.nextDouble() * 0.6));
            } else {
                newMinPts = minMinPts + random.nextInt(maxMinPts - minMinPts + 1);
            }

            // Select hashTables with bias
            int newHashTables;
            if (random.nextDouble() < 0.7 && !hashTablesScores.isEmpty()) {
                newHashTables = (int) selectParameterValueByPerformance(hashTablesScores);
                newHashTables = (int) Math.round(newHashTables * (0.7 + random.nextDouble() * 0.6));
            } else {
                newHashTables = minHashTables + random.nextInt(maxHashTables - minHashTables + 1);
            }

            // Select hyperplanes with bias
            int newHyperplanes;
            if (random.nextDouble() < 0.7 && !hyperplanesScores.isEmpty()) {
                newHyperplanes = (int) selectParameterValueByPerformance(hyperplanesScores);
                newHyperplanes = (int) Math.round(newHyperplanes * (0.7 + random.nextDouble() * 0.6));
            } else {
                newHyperplanes = minHyperplanes + random.nextInt(maxHyperplanes - minHyperplanes + 1);
            }

            // Ensure we're within allowed ranges
            newEpsilon = Math.max(minEpsilon, Math.min(maxEpsilon, newEpsilon));
            newMinPts = Math.max(minMinPts, Math.min(maxMinPts, newMinPts));
            newHashTables = Math.max(minHashTables, Math.min(maxHashTables, newHashTables));
            newHyperplanes = Math.max(minHyperplanes, Math.min(maxHyperplanes, newHyperplanes));

            ParameterCombination newCombo = new ParameterCombination(
                    newEpsilon, newMinPts, newHashTables, newHyperplanes);

            // Only add if we haven't evaluated this combination before
            if (!hasBeenEvaluated(newCombo, previousResults) && !combinations.contains(newCombo)) {
                combinations.add(newCombo);
                attempts = 0; // Reset attempts counter when we successfully add a combination
            }
        }

        return combinations;
    }

    /**
     * Calculate an overall score for a parameter result (higher is better)
     */
    private double calculateScore(ParameterResult result) {
        if (result.isDegenerate()) {
            return Double.NEGATIVE_INFINITY;
        }

        // Invert Davies-Bouldin Index since lower is better
        double dbiScore = result.getDaviesBouldinIndex() == 0 ? 0 : 1.0 / result.getDaviesBouldinIndex();

        // Weights for different metrics
        double daviesBouldinWeight = 0.10;
        double silhouetteWeight = 0.25;
        double calinskiHarabaszWeight = 0.05;
        double dunnWeight = 0.20;
        double dbcvWeight = 0.40;

        // Calculate weighted score
        double score = (daviesBouldinWeight * dbiScore) +
                (silhouetteWeight * result.getSilhouetteCoefficient()) +
                (calinskiHarabaszWeight * result.getCalinskiHarabaszIndex() / 1000.0) + // Scale down CH index
                (dunnWeight * result.getDunnIndex()) +
                (dbcvWeight * result.getDbcv());

        return score;
    }

    /**
     * Check if a parameter combination has already been evaluated
     */
    private boolean hasBeenEvaluated(ParameterCombination combo, List<ParameterResult> results) {
        double epsilon = combo.getEpsilon();
        int minPts = combo.getMinPts();
        int hashTables = combo.getHashTables();
        int hyperplanes = combo.getHyperplanes();

        // Use a small tolerance for floating-point epsilon comparison
        double tolerance = 0.000001;

        for (ParameterResult result : results) {
            if (Math.abs(result.getEpsilon() - epsilon) < tolerance &&
                    result.getMinPts() == minPts &&
                    result.getHashTables() == hashTables &&
                    result.getHyperplanes() == hyperplanes) {
                return true;
            }
        }

        return false;
    }

    /**
     * Select a parameter value with probability proportional to its performance
     */
    private <T> double selectParameterValueByPerformance(Map<T, Double> valueScores) {
        // Convert to arrays for selection
        List<T> values = new ArrayList<>(valueScores.keySet());
        double[] scores = new double[values.size()];

        // Ensure all scores are positive for probability calculation
        double minScore = Double.MAX_VALUE;
        for (double score : valueScores.values()) {
            minScore = Math.min(minScore, score);
        }

        // Calculate selection probabilities (weighted by score)
        double totalProb = 0;
        for (int i = 0; i < values.size(); i++) {
            // Add a small constant to ensure no zero probabilities
            scores[i] = valueScores.get(values.get(i)) - minScore + 0.1;
            totalProb += scores[i];
        }

        // Select with probability proportional to score
        double rand = Math.random() * totalProb;
        double cumProb = 0;

        for (int i = 0; i < values.size(); i++) {
            cumProb += scores[i];
            if (rand < cumProb) {
                // Convert any type to double
                if (values.get(i) instanceof Number) {
                    return ((Number) values.get(i)).doubleValue();
                } else {
                    return Double.parseDouble(values.get(i).toString());
                }
            }
        }

        // Fallback - return the last value
        if (values.get(values.size() - 1) instanceof Number) {
            return ((Number) values.get(values.size() - 1)).doubleValue();
        } else {
            return Double.parseDouble(values.get(values.size() - 1).toString());
        }
    }

    /**
     * Find the best result based on our scoring system
     */
    private ParameterResult findBestResult(List<ParameterResult> results) {
        if (results.isEmpty()) {
            return null;
        }

        ParameterResult best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (ParameterResult result : results) {
            double score = calculateScore(result);
            if (score > bestScore) {
                bestScore = score;
                best = result;
            }
        }

        return best;
    }

    private static class ParameterCombination {
        private final double epsilon;
        private final int minPts;
        private final int hashTables;
        private final int hyperplanes;

        public ParameterCombination(double epsilon, int minPts, int hashTables, int hyperplanes) {
            this.epsilon = epsilon;
            this.minPts = minPts;
            this.hashTables = hashTables;
            this.hyperplanes = hyperplanes;
        }

        public double getEpsilon() {
            return epsilon;
        }

        public int getMinPts() {
            return minPts;
        }

        public int getHashTables() {
            return hashTables;
        }

        public int getHyperplanes() {
            return hyperplanes;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;
            ParameterCombination that = (ParameterCombination) o;
            return Double.compare(that.epsilon, epsilon) == 0 &&
                    minPts == that.minPts &&
                    hashTables == that.hashTables &&
                    hyperplanes == that.hyperplanes;
        }

        @Override
        public int hashCode() {
            return Objects.hash(epsilon, minPts, hashTables, hyperplanes);
        }
    }

}
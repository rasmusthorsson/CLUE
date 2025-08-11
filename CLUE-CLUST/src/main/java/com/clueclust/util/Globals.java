package com.clueclust.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.AbstractMap.SimpleEntry;

import com.clueclust.core.BasePoint;
import com.clueclust.core.Point;

//Global variables and static utility functions
public class Globals {

    public interface DistanceFunction {
        double calculate(Point a, Point b);
    }

    public interface HashFunction {
        int calculate(Point a, BasePoint b);
    }

    // Default parameter values
    public static double EPSILON_ORIGINAL = 0.001;
    public static double EPSILON = EPSILON_ORIGINAL * EPSILON_ORIGINAL;
    public static int MIN_PTS = 500;

    // List of core points
    public static List<Point> CORE_POINTS = new ArrayList<>();

    // Distance metric types
    public enum Metric {
        EUCLIDEAN,
        ANGULAR,
        DTW
    }

    public static int DTW_WINDOW_SIZE = 10; // Default window size
    public static boolean USE_LB_KEOGH = true; // Use LB_Keogh lower bound
    private static Map<SimpleEntry<Long, Long>, Double> dtwDistanceCache = new ConcurrentHashMap<>();
    public static boolean CALCULATE_EXACT_DISTANCES = false; // Default to false for normal clustering
    public static int PARAM_OPT_LEVEL = 2;
    public static ArrayList<Double> PARAM_OPT_EPSILONS = new ArrayList<Double>();

    // Current metric
    public static Metric METRIC = Metric.EUCLIDEAN;

    // Distance functions
    public static final DistanceFunction EUCLIDEAN_DISTANCE = (a, b) -> {
        return a.squaredEuclideanDistance(b);
    };

    public static final DistanceFunction ANGULAR_DISTANCE = (a, b) -> {
        double result = a.innerProduct(b);
        result = result / (a.norm() * b.norm());

        if (result > 1) {
            if (result - 1 < 0.00001) {
                result = 1;
            } else {
                throw new IllegalStateException("Angular distance calculation error: cosine > 1");
            }
        }

        result = Math.acos(result);
        result = result / Math.PI;

        return result;
    };

    // DTW distance function, with early abandoning, if the distance exceeds
    // epsilson we cancel
    public static final DistanceFunction DTW_DISTANCE = (a, b) -> {
        // Create a unique key for these two points
        SimpleEntry<Long, Long> key = new SimpleEntry<>(Math.min(a.getId(), b.getId()), Math.max(a.getId(), b.getId()));

        // Check if we already computed this distance
        Double cachedDist = dtwDistanceCache.get(key);
        if (cachedDist != null) {
            return cachedDist;
        }
        // If calculating exact distances, skip LB_Keogh pruning
        if (!CALCULATE_EXACT_DISTANCES && USE_LB_KEOGH) {
            double lb = calculateLBKeogh(a.getFeatures(), b.getFeatures());
            if (lb > EPSILON) {
                return Double.POSITIVE_INFINITY; // Early pruning only during clustering
            }
        }

        // Calculate actual DTW distance
        double distance = calculateDTW(a.getFeatures(), b.getFeatures());

        // Cache the result if it's a finite distance or if we're calculating exact
        // distances
        if (CALCULATE_EXACT_DISTANCES || distance < Double.POSITIVE_INFINITY) {
            dtwDistanceCache.put(key, distance);
        }

        return distance;
    };

    // Window mode
    private static double calculateDTW(List<Double> series1, List<Double> series2) {
        // Ensure series1 is the shorter series to minimize memory usage
        if (series1.size() > series2.size()) {
            return calculateDTW(series2, series1);
        }

        int n = series1.size();
        int m = series2.size();
        int window = Math.min(DTW_WINDOW_SIZE, m - 1);

        // Use just two rows for DTW calculation
        double[] prev = new double[m + 1];
        double[] curr = new double[m + 1];

        // Initialize first row
        for (int j = 0; j <= m; j++) {
            prev[j] = Double.POSITIVE_INFINITY;
        }
        prev[0] = 0;

        // Fill the DTW matrix
        for (int i = 1; i <= n; i++) {
            curr[0] = Double.POSITIVE_INFINITY;

            // Calculate window boundaries
            int start = Math.max(1, i - window);
            int end = Math.min(m, i + window);

            // Initialize cells outside the window
            for (int j = 1; j < start; j++) {
                curr[j] = Double.POSITIVE_INFINITY;
            }

            // Calculate only cells within the window
            for (int j = start; j <= end; j++) {
                double cost = Math.pow(series1.get(i - 1) - series2.get(j - 1), 2);
                curr[j] = cost + Math.min(
                        Math.min(prev[j], curr[j - 1]),
                        prev[j - 1]);

                // Only do early abandoning if not calculating exact distances
                if (!CALCULATE_EXACT_DISTANCES && curr[j] > EPSILON) {

                    return Double.POSITIVE_INFINITY;
                }
            }

            // Initialize remaining cells
            for (int j = end + 1; j <= m; j++) {
                curr[j] = Double.POSITIVE_INFINITY;
            }

            // Swap rows
            double[] temp = prev;
            prev = curr;
            curr = temp;
        }

        // Always return the actual distance when calculating exact distances
        if (CALCULATE_EXACT_DISTANCES) {
            return prev[m];
        }

        // Otherwise return infinity if it exceeds epsilon (for normal clustering)
        return prev[m] <= EPSILON ? prev[m] : Double.POSITIVE_INFINITY;
    }

    // LB_Keogh lower bound calculation
    private static double calculateLBKeogh(List<Double> series1, List<Double> series2) {
        // Ensure series1 is the shorter series
        if (series1.size() > series2.size()) {
            return calculateLBKeogh(series2, series1);
        }

        int n = series1.size();
        int m = series2.size();
        int window = Math.min(DTW_WINDOW_SIZE, m - 1);

        // Create envelope arrays
        double[] upper = new double[n];
        double[] lower = new double[n];

        // Initialize envelope bounds
        for (int i = 0; i < n; i++) {
            upper[i] = Double.NEGATIVE_INFINITY;
            lower[i] = Double.POSITIVE_INFINITY;
        }

        // Compute envelope for series2
        for (int i = 0; i < n; i++) {
            int start = Math.max(0, i - window);
            int end = Math.min(m - 1, i + window);

            for (int j = start; j <= end; j++) {
                double val = series2.get(j);
                if (val > upper[i])
                    upper[i] = val;
                if (val < lower[i])
                    lower[i] = val;
            }
        }

        // Compute LB_Keogh distance
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double val = series1.get(i);
            if (val > upper[i]) {
                double diff = val - upper[i];
                sum += diff * diff;
            } else if (val < lower[i]) {
                double diff = lower[i] - val;
                sum += diff * diff;
            }

            // Early abandoning only if not calculating exact distances
            if (!CALCULATE_EXACT_DISTANCES && sum > EPSILON) {
                return Double.POSITIVE_INFINITY;
            }
        }

        return sum;
    }

    // First attempt, does all comparisons always. (Very slow if epsilon is too big)
    private static double calculateDTW_v1(List<Double> series1, List<Double> series2) {
        int n = series1.size();
        int m = series2.size();

        // Create DTW matrix
        double[][] dtw = new double[n + 1][m + 1];

        // Initialize first row and column to infinity
        for (int i = 0; i <= n; i++) {
            dtw[i][0] = Double.POSITIVE_INFINITY;
        }

        for (int j = 0; j <= m; j++) {
            dtw[0][j] = Double.POSITIVE_INFINITY;
        }

        // Base case
        dtw[0][0] = 0;

        // Fill the matrix
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                double cost = Math.pow(series1.get(i - 1) - series2.get(j - 1), 2);
                dtw[i][j] = cost + Math.min(
                        Math.min(dtw[i - 1][j], dtw[i][j - 1]),
                        dtw[i - 1][j - 1]);
            }
        }

        // Return the DTW distance
        return dtw[n][m];
    }

    // Current distance function
    public static DistanceFunction DISTANCE_FUNCTION = EUCLIDEAN_DISTANCE;

    // Hash functions
    public static final HashFunction EUCLIDEAN_HASH = (point, hyperplane) -> {
        double result = point.innerProduct(hyperplane);
        result = result / EPSILON_ORIGINAL;
        result = Math.floor(result);
        return (int) result;
    };

    public static final HashFunction ANGULAR_HASH = (point, hyperplane) -> {
        return point.innerProduct(hyperplane) >= 0 ? 1 : -1;
    };

    // Current hash function
    public static HashFunction HASH_FUNCTION = EUCLIDEAN_HASH;

    // Set metric to use
    public static void setMetric(Metric metric) {
        METRIC = metric;

        if (metric == Metric.ANGULAR) {
            DISTANCE_FUNCTION = ANGULAR_DISTANCE;
            HASH_FUNCTION = ANGULAR_HASH;
        } else if (metric == Metric.DTW) {
            DISTANCE_FUNCTION = DTW_DISTANCE;
            HASH_FUNCTION = EUCLIDEAN_HASH;
        } else { // EUCLIDEAN
            DISTANCE_FUNCTION = EUCLIDEAN_DISTANCE;
            HASH_FUNCTION = EUCLIDEAN_HASH;
            EPSILON = EPSILON_ORIGINAL * EPSILON_ORIGINAL;
        }

    }
}

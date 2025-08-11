package com.clueclust.util;

import java.util.List;

/**
 * Utility class for detecting elbow points in distance plots
 */
public class ElbowDetector {
    /**
     * Find the index of the elbow point in a sorted list of distances
     * 
     * @param sortedDistances List of distances in ascending order
     * @return Index of the detected elbow point, or -1 if none found
     */
    public int findElbowIndex(List<Double> sortedDistances) {
        if (sortedDistances.size() < 10) {
            return -1; // Not enough points for meaningful analysis
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
            return -1; // All distances are the same
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

        return elbowIndex;
    }
}
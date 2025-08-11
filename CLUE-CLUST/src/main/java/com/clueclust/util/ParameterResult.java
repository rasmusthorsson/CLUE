package com.clueclust.util;

public class ParameterResult {
    private final double epsilon;
    private final int minPts;
    private final int hashTables;
    private final int hyperplanes;
    private final int numClusters;
    private final int noiseCount;
    private final int clusteredPoints;
    private final double daviesBouldinIndex;
    private final double silhouetteCoefficient;
    private final double calinskiHarabaszIndex;
    private final double dunnIndex;
    private final double dbcv;
    private final boolean degenerate;
    private final String degenerateReason;

public ParameterResult(double epsilon, int minPts, int hashTables, int hyperplanes) {
        this.epsilon = epsilon;
        this.minPts = minPts;
        this.hashTables = hashTables;
        this.hyperplanes = hyperplanes;
        this.numClusters = 0;
        this.noiseCount = 0;
        this.clusteredPoints = 0;
        this.daviesBouldinIndex = 0;
        this.silhouetteCoefficient = 0;
        this.calinskiHarabaszIndex = 0;
        this.dunnIndex = 0;
        this.dbcv = 0;
        this.degenerate = false;
        this.degenerateReason = "";
        }

    public ParameterResult(double epsilon, int minPts, int hashTables, int hyperplanes,
            int numClusters, int noiseCount, int clusteredPoints,
            double daviesBouldinIndex, double silhouetteCoefficient, double calinskiHarabaszIndex, double dunnIndex,
            double dbcv,
            boolean degenerate, String degenerateReason) {
        this.epsilon = epsilon;
        this.minPts = minPts;
        this.hashTables = hashTables;
        this.hyperplanes = hyperplanes;
        this.numClusters = numClusters;
        this.noiseCount = noiseCount;
        this.clusteredPoints = clusteredPoints;
        this.daviesBouldinIndex = daviesBouldinIndex;
        this.silhouetteCoefficient = silhouetteCoefficient;
        this.calinskiHarabaszIndex = calinskiHarabaszIndex;
        this.dunnIndex = dunnIndex;
        this.dbcv = dbcv;
        this.degenerate = degenerate;
        this.degenerateReason = degenerateReason;
    }

    // Getters
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

    public int getNumClusters() {
        return numClusters;
    }

    public int getNoiseCount() {
        return noiseCount;
    }

    public int getClusteredPoints() {
        return clusteredPoints;
    }

    public double getDaviesBouldinIndex() {
        return daviesBouldinIndex;
    }

    public double getSilhouetteCoefficient() {
        return silhouetteCoefficient;
    }

    public double getCalinskiHarabaszIndex() {
        return calinskiHarabaszIndex;
    }

    public double getDunnIndex() {
        return dunnIndex;
    }

    public double getDbcv() {
        return dbcv;
    }

    public boolean isDegenerate() {
        return degenerate;
    }

    public String getDegenerateReason() {
        return degenerateReason;
    }

    @Override
    public String toString() {
        return String.format(
                "ε=%.4f, minPts=%d → %d clusters, %d noise (%.1f%%), %s%s",
                epsilon, minPts, numClusters, noiseCount,
                (double) noiseCount / (noiseCount + clusteredPoints) * 100,
                degenerate ? "DEGENERATE: " + degenerateReason : "",
                degenerate ? ""
                        : String.format(", DBI=%.4f, Silhouette=%.4f, CH=%.4f, Dunn=%.4f, DBCV=%.4f",
                                daviesBouldinIndex, silhouetteCoefficient, calinskiHarabaszIndex, dunnIndex, dbcv));
    }
    public String toStringNoDegen() {
        return String.format(
                "ε=%.4f, minPts=%d → %d clusters, %d noise (%.1f%%), %s",
                epsilon, minPts, numClusters, noiseCount,
                (double) noiseCount / (noiseCount + clusteredPoints) * 100,
                String.format(", DBI=%.4f, Silhouette=%.4f, CH=%.4f, Dunn=%.4f, DBCV=%.4f",
                                daviesBouldinIndex, silhouetteCoefficient, calinskiHarabaszIndex, dunnIndex, dbcv));
    }
}

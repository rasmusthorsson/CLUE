package com.clueclust.algorithm;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.clueclust.core.Dataset;
import com.clueclust.core.HashTable;
import com.clueclust.core.Point;
import com.clueclust.util.Globals;
import com.clueclust.util.RandomGenerator;

/**
 * Base LSHDBSCAN algorithm implementation
 */
public class LSHDBSCAN {
    protected Dataset dataset;
    protected int numberOfHashTables;
    protected int numberOfHyperplanesPerTable;
    protected List<HashTable> hashTables = new ArrayList<>();
    protected RandomGenerator randomGenerator;
    protected boolean benchmark;

    // Timing variables for benchmarking
    protected long initializingHashTablesTime;
    protected long populatingHashTablesTime;
    protected long identifyingCoreBucketsTime;
    protected long identifyingMergeTasksTime;
    protected long performingMergeTasksTime;
    protected long relabelingDataTime;

    /**
     * Constructor with dataset, hash tables, and hyperplanes
     */
    public LSHDBSCAN(Dataset dataset, int numberOfHashTables, int numberOfHyperplanesPerTable, boolean benchmark) {
        this.dataset = dataset;
        this.numberOfHashTables = numberOfHashTables;
        this.numberOfHyperplanesPerTable = numberOfHyperplanesPerTable;
        this.benchmark = benchmark;
        this.randomGenerator = new RandomGenerator();

        long startTime = System.nanoTime();

        // Initialize hash tables
        for (int i = 0; i < numberOfHashTables; i++) {
            hashTables.add(new HashTable(dataset, numberOfHyperplanesPerTable, randomGenerator));
        }
        long endTime = System.nanoTime();
        initializingHashTablesTime = endTime - startTime;
    }

    /**
     * Write benchmark results to a file
     */
    public void writeBenchmarkResults(String fileName, char delimiter) throws IOException {
        if (!benchmark) {
            System.out.println("Benchmarking was not activated!");
            return;
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName, true))) {
            writer.write((int) (initializingHashTablesTime / 1_000_000_000.0 + delimiter +
                    populatingHashTablesTime / 1_000_000_000.0 + delimiter +
                    identifyingCoreBucketsTime / 1_000_000_000.0 + delimiter +
                    identifyingMergeTasksTime / 1_000_000_000.0 + delimiter +
                    performingMergeTasksTime / 1_000_000_000.0 + delimiter +
                    relabelingDataTime / 1_000_000_000.0));
            writer.newLine();
        }
    }

    /**
     * Introduce the algorithm configuration
     */
    public void introduceMe() {
        System.out.println("LSHDBSCAN:");
        System.out.println("\t#points: " + dataset.getPoints().size());
        System.out.println("\t#dims: " + dataset.getNumberOfDimensions());
        System.out.println("\tmetric: " + (Globals.METRIC == Globals.Metric.ANGULAR ? "angular" : "euclidean"));
        System.out.println("\t#HashTables: " + numberOfHashTables);
        System.out.println("\t#HyperPlanesPerHashTable: " + numberOfHyperplanesPerTable);
        System.out.println("\tEps: " + Globals.EPSILON_ORIGINAL);
        System.out.println("\tminPts: " + Globals.MIN_PTS);
    }

    /**
     * Constructor with dataset and hash tables from files
     */
    public LSHDBSCAN(Dataset dataset, List<String> fileNames, boolean benchmark) {
        this.dataset = dataset;
        this.numberOfHashTables = fileNames.size();
        this.benchmark = benchmark;

        long startTime = System.nanoTime();

        // Initialize hash tables from files
        try {
            for (String fileName : fileNames) {
                hashTables.add(new HashTable(dataset, fileName));
            }
        } catch (IOException e) {
            System.err.println("Error initializing hash tables from files: " + e.getMessage());
            System.exit(1);
        }

        long endTime = System.nanoTime();
        initializingHashTablesTime = endTime - startTime;
    }

    /**
     * Populate hash tables with points
     */
    protected void populateHashTables() {
        for (HashTable hashTable : hashTables) {
            hashTable.populateHashTable();
        }
    }

    /**
     * Identify core buckets in hash tables
     */
    protected void identifyCoreBuckets() {
        for (HashTable hashTable : hashTables) {
            hashTable.identifyCoreBucketsDensityStyle();
        }
    }

    /**
     * Identify merge tasks between core buckets
     */
    protected void identifyMergeTasks() {
        for (HashTable hashTable : hashTables) {
            hashTable.identifyAndPerformMergeTasks();
        }
    }

    /**
     * Perform merge tasks to link clusters
     */
    protected void performMergeTasks() {
        // Empty in base class as merge tasks are performed in
        // identifyAndPerformMergeTasks
    }

    /**
     * Perform final relabeling of points
     */
    protected void performRelabeling() {
        dataset.relabelData();
    }

    /**
     * Perform the complete clustering algorithm
     */
    public void performClustering() {
        dataset.resetData();

        if (!benchmark) {
            populateHashTables();
            identifyCoreBuckets();
            identifyMergeTasks();
            performMergeTasks();
            performRelabeling();
        } else {
            // Benchmark each step
            long startTime, endTime;

            startTime = System.nanoTime();
            populateHashTables();
            endTime = System.nanoTime();
            populatingHashTablesTime = endTime - startTime;

            startTime = System.nanoTime();
            identifyCoreBuckets();
            endTime = System.nanoTime();
            identifyingCoreBucketsTime = endTime - startTime;

            startTime = System.nanoTime();
            identifyMergeTasks();
            endTime = System.nanoTime();
            identifyingMergeTasksTime = endTime - startTime;

            startTime = System.nanoTime();
            performMergeTasks();
            endTime = System.nanoTime();
            performingMergeTasksTime = endTime - startTime;

            startTime = System.nanoTime();
            performRelabeling();
            endTime = System.nanoTime();
            relabelingDataTime = endTime - startTime;

        }
        // Validation check
        Set<Integer> uniqueLabels = new HashSet<>();
        int nonNoiseWithoutCore = 0;

        for (Point p : dataset.getPoints()) {
            if (p.getLabel() != Point.NOISE) {
                uniqueLabels.add(p.getLabel());

                // Find the root point for this cluster
                Point root = p.findRoot();
                if (!root.isCore()) {
                    nonNoiseWithoutCore++;
                    System.err.println("WARNING: Point " + p.getId() + " has non-noise label " +
                            p.getLabel() + " but its root (ID=" + root.getId() +
                            ") is not a core point!");
                }
            }
        }

        System.out.println("Found " + uniqueLabels.size() + " clusters after clustering");
        if (nonNoiseWithoutCore > 0) {
            System.err.println("ERROR: " + nonNoiseWithoutCore +
                    " non-noise points have roots that are not core points!");
        }
    }
}

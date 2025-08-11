package com.clueclust.algorithm;

import com.clueclust.core.Dataset;
import com.clueclust.core.HashTable;
import com.clueclust.core.Point;
import com.clueclust.tasks.*;
import com.clueclust.util.Globals;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Concurrent implementation of LSHDBSCAN using multiple threads
 */
public class ConcurrentLSHDBSCAN extends LSHDBSCAN {
    private final int numberOfThreads;
    private final int numberOfBatches;
    private ExecutorService executorService;

    // Task lists for concurrent execution
    private final List<PopulationTask> populationTasks = new ArrayList<>();
    private final List<RelabelingTask> relabelingTasks = new ArrayList<>();
    private final List<CoreBucketIdentificationTask> coreBucketIdentificationTasks = new ArrayList<>();
    private final List<MergeTaskIdentificationTask> mergeTasksIdentificationTasks = new ArrayList<>();

    /**
     * Constructor
     */
    public ConcurrentLSHDBSCAN(Dataset dataset, int numberOfHashTables, int numberOfHyperplanesPerTable,
            int numberOfThreads, int numberOfBatches, boolean benchmark) {
        super(dataset, numberOfHashTables, numberOfHyperplanesPerTable, benchmark);

        if (numberOfThreads < 1) {
            throw new IllegalArgumentException("Number of threads must be at least 1");
        }
        if (numberOfBatches < 1) {
            throw new IllegalArgumentException("Number of batches must be at least 1");
        }

        this.numberOfThreads = numberOfThreads;
        this.numberOfBatches = numberOfBatches;

        // Create tasks
        initializeTasks();
    }

    /**
     * Initialize tasks for parallel execution
     */
    private void initializeTasks() {
        // Distribute points across batches
        List<Point> points = dataset.getPoints();
        int batchSize = (int) Math.ceil((double) points.size() / numberOfBatches);

        for (int batch = 0; batch < numberOfBatches; batch++) {
            int startIndex = batch * batchSize;
            int endIndex = Math.min(startIndex + batchSize, points.size());

            // Skip empty batches
            if (startIndex >= points.size()) {
                break;
            }

            List<Point> batchPoints = points.subList(startIndex, endIndex);

            // Create population tasks for each hash table
            for (HashTable hashTable : hashTables) {
                PopulationTask task = new PopulationTask(batchPoints, hashTable);
                populationTasks.add(task);
            }

            // Create relabeling task
            RelabelingTask relabelingTask = new RelabelingTask(batchPoints);
            relabelingTasks.add(relabelingTask);
        }

        // Create core bucket identification tasks
        for (HashTable hashTable : hashTables) {
            CoreBucketIdentificationTask task = new CoreBucketIdentificationTask(hashTable);
            coreBucketIdentificationTasks.add(task);
        }

        // Create merge tasks identification tasks
        for (HashTable hashTable : hashTables) {
            MergeTaskIdentificationTask task = new MergeTaskIdentificationTask(hashTable);
            mergeTasksIdentificationTasks.add(task);
        }
    }

    /**
     * Override populateHashTables for concurrent execution
     */
    @Override
    protected void populateHashTables() {
        System.out.println("Populating hash tables concurrently with " + numberOfThreads + " threads");

        executorService = Executors.newFixedThreadPool(numberOfThreads);

        for (PopulationTask task : populationTasks) {
            executorService.submit(() -> {
                if (!task.isBooked()) {
                    if (task.book()) {
                        task.execute();
                    }
                }
            });
        }

        shutdownAndAwaitTermination();
    }

    /**
     * Override identifyCoreBuckets for concurrent execution
     */
    @Override
    protected void identifyCoreBuckets() {
        System.out.println("Identifying core buckets concurrently with " + numberOfThreads + " threads");

        executorService = Executors.newFixedThreadPool(numberOfThreads);

        for (CoreBucketIdentificationTask task : coreBucketIdentificationTasks) {
            executorService.submit(() -> {
                if (!task.isBooked()) {
                    if (task.book()) {
                        if (Globals.METRIC == Globals.Metric.EUCLIDEAN) {
                            task.executeDensityStyle();
                        } else {
                            task.execute();
                        }
                    }
                }
            });
        }

        shutdownAndAwaitTermination();
    }

    /**
     * Override identifyMergeTasks for concurrent execution
     */
    @Override
    protected void identifyMergeTasks() {
        System.out.println("Identifying and performing merge tasks concurrently with " + numberOfThreads + " threads");

        executorService = Executors.newFixedThreadPool(numberOfThreads);

        for (MergeTaskIdentificationTask task : mergeTasksIdentificationTasks) {
            executorService.submit(() -> {
                if (!task.isBooked()) {
                    if (task.book()) {
                        task.executeAndPerform();
                    }
                }
            });
        }

        shutdownAndAwaitTermination();
    }

    /**
     * Override performRelabeling for concurrent execution
     */
    @Override
    protected void performRelabeling() {
        System.out.println("Performing relabeling concurrently with " + numberOfThreads + " threads");

        executorService = Executors.newFixedThreadPool(numberOfThreads);

        for (RelabelingTask task : relabelingTasks) {
            executorService.submit(() -> {
                if (!task.isBooked()) {
                    if (task.book()) {
                        task.execute();
                    }
                }
            });
        }

        shutdownAndAwaitTermination();
    }

    /**
     * Shutdown executor service and wait for termination
     */
    private void shutdownAndAwaitTermination() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(1, TimeUnit.HOURS)) {
                executorService.shutdownNow();
                if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                    System.err.println("Executor service did not terminate");
                }
            }
        } catch (InterruptedException ie) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Override introduceMe to include thread and batch information
     */
    @Override
    public void introduceMe() {
        System.out.println("Concurrent LSHDBSCAN:");
        System.out.println("\t#points: " + dataset.getPoints().size());
        System.out.println("\t#dims: " + dataset.getNumberOfDimensions());
        System.out.println("\tmetric: " + (Globals.METRIC == Globals.Metric.ANGULAR ? "angular" : "euclidean"));
        System.out.println("\t#HashTables: " + numberOfHashTables);
        System.out.println("\t#HyperPlanesPerHashTable: " + numberOfHyperplanesPerTable);
        System.out.println("\t#threads: " + numberOfThreads);
        System.out.println("\t#batches: " + numberOfBatches);
        System.out.println("\tEps: " + Globals.EPSILON_ORIGINAL);
        System.out.println("\tminPts: " + Globals.MIN_PTS);
    }
}
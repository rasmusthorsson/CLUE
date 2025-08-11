package com.clueclust.tasks;

import com.clueclust.core.CoreBucket;
import com.clueclust.core.HashTable;
import com.clueclust.core.Point;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Task for identifying core buckets in a hash table
 */
public class CoreBucketIdentificationTask {
    private final HashTable hashTable;
    private final AtomicBoolean booked = new AtomicBoolean(false);

    /**
     * Constructor
     */
    public CoreBucketIdentificationTask(HashTable hashTable) {
        this.hashTable = hashTable;
    }

    /**
     * Execute the task - identify core buckets
     */
    public void execute() {
        hashTable.identifyCoreBuckets();
    }

    /**
     * Execute the task using density-style approach
     */
    public void executeDensityStyle() {
        // Make sure we're marking core points correctly
        for (CoreBucket bucket : hashTable.getCoreBuckets()) {
            Point rep = bucket.getRepresentative();
            if (!rep.isCore()) {
                System.err.println("WARNING: Core bucket representative " + rep.getId() +
                        " was not marked as a core point!");
                rep.setAsCorePoint();
            }
        }

        hashTable.identifyCoreBucketsDensityStyle();
    }

    /**
     * Attempt to book this task for execution
     */
    public boolean book() {
        return booked.compareAndSet(false, true);
    }

    /**
     * Check if this task is already booked
     */
    public boolean isBooked() {
        return booked.get();
    }
}
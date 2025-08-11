package com.clueclust.tasks;

import com.clueclust.core.HashTable;
import com.clueclust.core.Point;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Task for populating a hash table with a batch of points
 */
public class PopulationTask {
    private final List<Point> points;
    private final HashTable hashTable;
    private final AtomicBoolean booked = new AtomicBoolean(false);
    
    /**
     * Constructor
     */
    public PopulationTask(List<Point> points, HashTable hashTable) {
        this.points = points;
        this.hashTable = hashTable;
    }
    
    /**
     * Execute the task - populate the hash table with points
     */
    public void execute() {
        // Use the list version of populate hash table
        hashTable.populateHashTable(points);
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
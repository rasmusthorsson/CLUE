package com.clueclust.tasks;

import com.clueclust.core.HashTable;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Task for identifying and performing merge tasks
 */
public class MergeTaskIdentificationTask {
    private final HashTable hashTable;
    private final AtomicBoolean booked = new AtomicBoolean(false);
    
    /**
     * Constructor
     */
    public MergeTaskIdentificationTask(HashTable hashTable) {
        this.hashTable = hashTable;
    }
    
    /**
     * Execute the task - identify merge tasks
     */
    public void execute() {
        hashTable.identifyAndPerformMergeTasks();
    }
    
    /**
     * Execute and perform merge tasks
     */
    public void executeAndPerform() {
        hashTable.identifyAndPerformMergeTasks();
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
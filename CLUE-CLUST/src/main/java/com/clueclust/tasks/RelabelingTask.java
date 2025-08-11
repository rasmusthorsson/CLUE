package com.clueclust.tasks;

import com.clueclust.core.Point;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Task for relabeling points
 */
public class RelabelingTask {
    private final List<Point> points;
    private final AtomicBoolean booked = new AtomicBoolean(false);
    
    /**
     * Constructor
     */
    public RelabelingTask(List<Point> points) {
        this.points = points;
    }
    
    /**
     * Execute the task - relabel points
     */
    public void execute() {
        for (Point point : points) {
            point.unsafeCompress();
            point.reLabel();
        }
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
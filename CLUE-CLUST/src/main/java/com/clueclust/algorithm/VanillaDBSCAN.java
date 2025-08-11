package com.clueclust.algorithm;

import com.clueclust.core.Dataset;
import com.clueclust.core.Point;
import com.clueclust.util.Globals;

import java.util.ArrayList;
import java.util.List;

/**
 * VanillaDBSCAN implements the original DBSCAN algorithm
 */
public class VanillaDBSCAN {
    private final Dataset dataset;
    
    /**
     * Constructor
     */
    public VanillaDBSCAN(Dataset dataset) {
        this.dataset = dataset;
    }
    
    /**
     * Perform DBSCAN clustering
     */
    public void performClustering() {
        System.out.println("Performing Vanilla DBSCAN");
        
        int clusterIndex = 0;
        
        for (Point p : dataset.getPoints()) {
            if (!p.isProcessed()) {
                p.setProcessed(true);
                List<Point> pointsToExplore = getEpsNeighbors(p);
                
                if (pointsToExplore.size() >= Globals.MIN_PTS) {
                    p.setAsCorePoint();
                    p.setLabel(++clusterIndex);
                    
                    while (!pointsToExplore.isEmpty()) {
                        Point q = pointsToExplore.remove(pointsToExplore.size() - 1);
                        
                        if (q.isNoise()) {
                            q.setLabel(clusterIndex);
                        }
                        
                        if (!q.isProcessed()) {
                            q.setProcessed(true);
                            List<Point> neighborsOfQ = getEpsNeighbors(q);
                            
                            if (neighborsOfQ.size() >= Globals.MIN_PTS) {
                                q.setAsCorePoint();
                                
                                for (Point neighbor : neighborsOfQ) {
                                    if (neighbor.isNoise()) {
                                        neighbor.setLabel(clusterIndex);
                                    }
                                    if (!neighbor.isProcessed()) {
                                        pointsToExplore.add(neighbor);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Get epsilon neighbors of a point
     */
    private List<Point> getEpsNeighbors(Point query) {
        List<Point> result = new ArrayList<>();
        
        for (Point p : dataset.getPoints()) {
            if (Globals.DISTANCE_FUNCTION.calculate(query, p) <= Globals.EPSILON) {
                result.add(p);
            }
        }
        
        return result;
    }
}
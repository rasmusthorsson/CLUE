package com.clueclust.algorithm;

import com.clueclust.core.Dataset;
import com.clueclust.core.HashTable;
import com.clueclust.core.Point;
import com.clueclust.util.Globals;
import com.clueclust.util.RandomGenerator;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * VanillaDBSCANLSH - A non-concurrent implementation of DBSCAN that uses LSH
 * for efficient neighborhood queries
 */
public class VanillaDBSCANLSH {
    private final Dataset dataset;
    private final int numberOfHashTables;
    private final int numberOfHyperplanesPerTable;
    private final List<HashTable> hashTables = new ArrayList<>();
    private final RandomGenerator randomGenerator = new RandomGenerator();
    
    /**
     * Constructor
     */
    public VanillaDBSCANLSH(Dataset dataset, int numberOfHashTables, int numberOfHyperplanesPerTable) {
        this.dataset = dataset;
        this.numberOfHashTables = numberOfHashTables;
        this.numberOfHyperplanesPerTable = numberOfHyperplanesPerTable;
        
        // Initialize hash tables
        for (int i = 0; i < numberOfHashTables; i++) {
            hashTables.add(new HashTable(dataset, numberOfHyperplanesPerTable, randomGenerator));
        }
        
        // Populate hash tables
        for (HashTable hashTable : hashTables) {
            hashTable.populateHashTable();
        }
    }
    
    /**
     * Get epsilon neighbors of a point using LSH
     */
    public List<Point> getEpsNeighbors(Point query) {
        Set<Point> resultSet = new HashSet<>();
        
        for (HashTable hashTable : hashTables) {
            List<Point> neighbors = hashTable.getEpsNeighbors(query);
            resultSet.addAll(neighbors);
        }
        
        return new ArrayList<>(resultSet);
    }
    
    /**
     * Perform DBSCAN clustering using LSH for neighborhood queries
     */
    public void performClustering() {
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
}
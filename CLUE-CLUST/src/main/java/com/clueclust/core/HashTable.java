package com.clueclust.core;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.clueclust.util.Globals;
import com.clueclust.util.RandomGenerator;

//HashTable for LSH
//Manages buckets of points based on their hash values
public class HashTable {

    private final Dataset dataset;
    private final List<Hyperplane> hyperplanes = new ArrayList<>();
    private final RandomGenerator randomGenerator;
    private final int numberOfHyperplanes;

    private final Map<HashedPoint, List<Point>> hashMap = new ConcurrentHashMap<>();
    private final List<CoreBucket> coreBuckets = new ArrayList<>();

    // Constructor with dataset, number of hyperplanes, and random generator
    public HashTable(Dataset dataset, int numberOfHyperplanes, RandomGenerator randomGenerator) {
        this.dataset = dataset;
        this.numberOfHyperplanes = numberOfHyperplanes;
        this.randomGenerator = randomGenerator;
        initializeHashTable(numberOfHyperplanes);
    }

    // Constructor with dataset and hyperplanes from file
    public HashTable(Dataset dataset, String hyperplanesFile) throws IOException {
        this.dataset = dataset;
        this.randomGenerator = null;

        try (BufferedReader reader = new BufferedReader(new FileReader(hyperplanesFile))) {
            String line;
            long id = 0;
            while ((line = reader.readLine()) != null) {
                Hyperplane hyperplane = new Hyperplane(line, id++);
                hyperplanes.add(hyperplane);
            }
        }

        this.numberOfHyperplanes = hyperplanes.size();

        for (Hyperplane h : hyperplanes) {
            if (h.getFeatures().size() != dataset.getNumberOfDimensions()) {
                throw new IOException("Error while reading hyperplanes from file: Inconsistent dimensionality");
            }
        }
    }

    // Init hash table with random hyperplanes
    private void initializeHashTable(int numberOfHyperplanes) {
        for (int i = 0; i < numberOfHyperplanes; i++) {
            List<Double> features = new ArrayList<>();
            for (int d = 0; d < dataset.getNumberOfDimensions(); d++) {
                features.add(randomGenerator.getRandomValue());
            }

            Hyperplane hyperplane = new Hyperplane(features);
            hyperplane.normalize(false);
            hyperplanes.add(hyperplane);

        }
    }

    // Populate hash table with all points from dataset
    public void populateHashTable() {
        for (Point point : dataset.getPoints()) {
            HashedPoint hashedPoint = hashPoint(point);
            hashMap.computeIfAbsent(hashedPoint, k -> new ArrayList<>()).add(point);
        }
    }

    public void populateHashTable(List<Point> points) {
        for (Point point : points) {
            HashedPoint hashedPoint = hashPoint(point);
            hashMap.computeIfAbsent(hashedPoint, k -> new ArrayList<>()).add(point);
        }
    }

    // Populate hash table with points from a subset
    public void populateHashTable(Iterator<Point> begin, Iterator<Point> end) {
        while (begin != end) {
            Point point = begin.next();
            HashedPoint hashedPoint = hashPoint(point);
            hashMap.computeIfAbsent(hashedPoint, k -> new ArrayList<>()).add(point);
        }
    }

    // Hash a point using hyperplanes
    private HashedPoint hashPoint(Point point) {
        HashedPoint hashedPoint = new HashedPoint();
        for (Hyperplane hyperplane : hyperplanes) {
            int hashValue = Globals.HASH_FUNCTION.calculate(point, hyperplane);
            hashedPoint.addHashValue(hashValue);
        }
        return hashedPoint;
    }

    // Identify core buckets - buckets with atleast minPts points
    public void identifyCoreBuckets() {
        for (Map.Entry<HashedPoint, List<Point>> entry : hashMap.entrySet()) {
            List<Point> bucket = entry.getValue();
            if (bucket.size() >= Globals.MIN_PTS) {
                Point core = bucket.get(getMedian(bucket));
                int count = 0;

                for (Point point : bucket) {
                    if (point.findRoot() == point &&
                            Globals.DISTANCE_FUNCTION.calculate(core, point) <= Globals.EPSILON) {
                        point.link(core);
                        count++;
                    }
                }

                if (count >= Globals.MIN_PTS) {
                    core.setAsCorePoint();
                    CoreBucket coreBucket = new CoreBucket();
                    coreBucket.setRepresentative(core);
                    coreBucket.setMembers(new ArrayList<>(bucket));
                    coreBuckets.add(coreBucket);
                }
            }
        }
    }

    // Identify core buckets using density-based approach
    public void identifyCoreBucketsDensityStyle() {
        for (Map.Entry<HashedPoint, List<Point>> entry : hashMap.entrySet()) {
            List<Point> bucket = entry.getValue();
            if (bucket.size() >= Globals.MIN_PTS) {
                List<Point> candidates = new ArrayList<>(bucket);
                Point core = candidates.get(getClosestToMean(candidates));
                List<Point> coreBucketMembers = new ArrayList<>();

                int count = 0;
                for (Point point : candidates) {
                    if (Globals.DISTANCE_FUNCTION.calculate(core, point) <= Globals.EPSILON) {
                        count++;
                        coreBucketMembers.add(point);
                    }
                }

                if (count >= Globals.MIN_PTS) {
                    // Create CoreBucket first (before linking operations)
                    CoreBucket coreBucket = new CoreBucket();
                    coreBucket.setRepresentative(core);
                    coreBucket.setMembers(coreBucketMembers);

                    // Mark core point
                    core.setAsCorePoint();

                    // Now perform linking
                    for (Point point : coreBucketMembers) {
                        if (point.findRoot() == point) {
                            point.link(core);
                        }
                    }

                    // Add to coreBuckets list
                    coreBuckets.add(coreBucket);
                }

            }
        }
    }

    // Identify and perform merge tasks - merge core buckets
    public void identifyAndPerformMergeTasks() {
        for (CoreBucket coreBucket : coreBuckets) {
            Point core = coreBucket.getRepresentative();
            // Ensure the representative is marked as a core point
            core.setAsCorePoint();

            for (Point element : coreBucket.getMembers()) {
                if (element.isCore() &&
                        core.findRoot() != element.findRoot() &&
                        Globals.DISTANCE_FUNCTION.calculate(core, element) <= Globals.EPSILON) {

                    // Make sure the core point becomes the root
                    Point rootOfCore = core.findRoot();
                    Point rootOfElement = element.findRoot();

                    if (!rootOfCore.isCore()) {
                        rootOfCore.setAsCorePoint();
                    }
                    if (!rootOfElement.isCore()) {
                        rootOfElement.setAsCorePoint();
                    }

                    element.link(core);
                }
            }
        }
    }

    // Find the median point in a list of points
    private int getMedian(List<Point> points) {
        List<Double> projVals = new ArrayList<>();

        // Create a hyperplane of all 1s
        List<Double> ones = new ArrayList<>();
        for (int i = 0; i < points.get(0).getFeatures().size(); i++) {
            ones.add(1.0);
        }
        Hyperplane onesHyperplane = new Hyperplane(ones);

        for (Point point : points) {
            projVals.add(point.innerProduct(onesHyperplane));
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            indices.add(i);
        }

        indices.sort(Comparator.comparingDouble(projVals::get));

        return indices.get(points.size() / 2);
    }

    // Find the point closest to the mean of a list of points
    private int getClosestToMean(List<Point> points) {
        // calculate mean point
        Point mean = new Point();
        for (int i = 0; i < dataset.getNumberOfDimensions(); i++) {
            mean.getFeatures().add(0.0);
        }

        for (Point p : points) {
            mean.addInPlace(p);
        }

        mean.divideInPlace(points.size());

        // Find the point closest to the mean
        double minDistance = Double.MAX_VALUE;
        int minIndex = 0;

        for (int i = 0; i < points.size(); i++) {
            double distance = Globals.DISTANCE_FUNCTION.calculate(points.get(i), mean);
            if (distance < minDistance) {
                minDistance = distance;
                minIndex = i;
            }
        }

        return minIndex;
    }

    // Get epsilon neighbors of a point
    public List<Point> getEpsNeighbors(Point query) {
        HashedPoint hashedPoint = hashPoint(query);
        List<Point> result = new ArrayList<>();

        List<Point> bucket = hashMap.get(hashedPoint);
        if (bucket != null) {
            for (Point neighbor : bucket) {
                if (Globals.DISTANCE_FUNCTION.calculate(query, neighbor) <= Globals.EPSILON) {
                    result.add(neighbor);
                }
            }
        }
        return result;
    }

    // Get the hyperplanes
    public List<Hyperplane> getHyperplanes() {
        return hyperplanes;
    }

    // Get the hash map
    public Map<HashedPoint, List<Point>> getHashMap() {
        return hashMap;
    }

    // Get the core buckets
    public List<CoreBucket> getCoreBuckets() {
        return coreBuckets;
    }

}

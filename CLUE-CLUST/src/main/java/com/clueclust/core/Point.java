package com.clueclust.core;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

public class Point extends BasePoint {

    // Constants for node labels
    public static final int NOISE = -1;
    public static final int NON_NOISE = -2;

    // Thread safe reference to parent point in cluster
    private final AtomicReference<Point> parent = new AtomicReference<>(null);

    // Label for this point, noise, non-noise or cluster ID
    private int label = NOISE;
    private boolean corePoint = false;
    private boolean processed = false; // Used in vanilla DBSCAN

    // Default constructor
    public Point() {
        super();
    }

    // Construct from list of features
    public Point(List<Double> features) {
        super(features);
    }

    // Construct from string and ID
    public Point(String line, long id) {
        super(line, id);
    }

    // Check if this point is processed
    public boolean isProcessed() {
        return processed;
    }

    // Mark this point as processed
    public void setProcessed(boolean processed) {
        this.processed = processed;
    }

    // Set the cluster label for this point
    public void setLabel(int label) {
        this.label = label;
    }

    // Check if this point is noise
    public boolean isNoise() {
        return label == NOISE;
    }

    // Find the root point of the cluster containing this point
    public Point findRoot() {
        Point u = this;
        while (true) {
            Point v = u.parent.get();
            if (v == null) {
                return u;
            }

            Point w = v.parent.get();
            if (w != null) {
                u.parent.compareAndSet(v, w); // Path compression -> point directly to grandparent
            }
            u = v;
        }
    }

    // Find the root point
    public Point cFindRoot() {
        Point p = this;
        Point father = p.parent.get();

        while (father != null) {
            p = father;
            father = p.parent.get();
        }
        return p;
    }

    // Compress the path to the root, used for final relabeling
    public void unsafeCompress() {
        if (parent.get() != null) {
            parent.set(findRoot());
        }
    }

    // Link this point to another point, implements union operation in union-find
    public void link(Point other) {
        Point rootOfThis = this.findRoot();
        Point rootOfOther = other.findRoot();

        if (rootOfThis.label == NOISE) {
            rootOfThis.label = NON_NOISE;
        }

        if (rootOfOther.label == NOISE) {
            rootOfOther.label = NON_NOISE;
        }

        // if already in same cluster, do nothing
        if (rootOfThis.id == rootOfOther.id) {
            assert rootOfThis == rootOfOther : "Roots with same ID should be the same object";
            return;
        }

        // IMPORTANT: Ensure core point status is maintained
        // If we're linking a core point to a non-core point, the core point should
        // become the root
        if (rootOfThis.isCore() && !rootOfOther.isCore()) {
            if (rootOfOther.parent.compareAndSet(null, rootOfThis)) {
                return;
            }
        } else if (!rootOfThis.isCore() && rootOfOther.isCore()) {
            if (rootOfThis.parent.compareAndSet(null, rootOfOther)) {
                return;
            }
        } else {
            // If both are core or both are non-core, use ID as rank
            if (rootOfThis.id < rootOfOther.id) {
                if (rootOfThis.parent.compareAndSet(null, rootOfOther)) {
                    // Transfer core status if necessary
                    if (rootOfThis.isCore()) {
                        rootOfOther.setAsCorePoint();
                    }
                    return;
                }
            } else {
                if (rootOfOther.parent.compareAndSet(null, rootOfThis)) {
                    // Transfer core status if necessary
                    if (rootOfOther.isCore()) {
                        rootOfThis.setAsCorePoint();
                    }
                    return;
                }
            }
        }

        // If compare and set fails, try again with updated roots
        rootOfThis.link(rootOfOther);
    }

    // Relabel this point with the cluster ID
    public void reLabel() {
        if (label == NON_NOISE) {
            Point root = findRoot();
            // Only assign a cluster label if the root is a core point
            if (root.isCore()) {
                label = (int) root.id;
            } else {
                // If the root is not a core point, this should be marked as noise
                label = NOISE;
            }
        }
    }

    // Reset this point to initial state
    public void reset() {
        parent.set(null);
        label = NOISE;
        corePoint = false;
        processed = false;
    }

    // Check if this is a core point
    public boolean isCore() {
        return corePoint;
    }

    // Mark as core point
    public void setAsCorePoint() {
        this.corePoint = true;
    }

    // Get the label of this point
    public int getLabel() {
        return label;
    }

    // Print this point with optional label-only flag

    public String toString(boolean onlyLabel) {
        if (onlyLabel) {
            return String.valueOf(label);
        } else {
            StringBuilder sb = new StringBuilder(super.toString());
            sb.append("\t").append(cFindRoot().id);
            sb.append("\t").append(corePoint ? "(c) " : "   ");
            sb.append("\t").append(label);
            return sb.toString();
        }
    }

    @Override
    public String toString() {
        return toString(false);
    }

}

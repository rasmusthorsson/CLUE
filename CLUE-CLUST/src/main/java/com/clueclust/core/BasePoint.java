package com.clueclust.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class BasePoint {
    protected long id = -1;
    protected List<Double> features = new ArrayList<>();
    protected boolean normalized = false;

    // Default constructor
    public BasePoint() {
    }

    // Construct from a list of features
    public BasePoint(List<Double> features) {
        this.features = new ArrayList<>(features);
    }

    // Construct from a string with feature values
    public BasePoint(String line, long id) {
        this.id = id;

        try (Scanner scanner = new Scanner(line)) {
            while (scanner.hasNextDouble()) {
                features.add(scanner.nextDouble());
            }
        }
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public List<Double> getFeatures() {
        return features;
    }

    // Calculate the inner product with another point
    public double innerProduct(BasePoint other) {
        if (features.size() != other.features.size()) {
            throw new IllegalArgumentException("Cannot calculate inner product of vectors with different dimensions");
        }
        double result = 0.0;
        for (int i = 0; i < features.size(); i++) {
            result += features.get(i) * other.features.get(i);
        }
        return result;
    }

    // Calculate the squared Euclidean distance to another point
    public double squaredEuclideanDistance(BasePoint other) {
        if (features.size() != other.features.size()) {
            throw new IllegalArgumentException("Cannot calculate distance between vectors with different dimensions");
        }
        double result = 0.0;
        for (int i = 0; i < features.size(); i++) {
            double diff = features.get(i) - other.features.get(i);
            result += diff * diff;
        }
        return result;
    }

    // Calculate the L2 norm (Euclidean length) of this vector
    public double norm() {
        if (normalized) {
            return 1.0;
        }
        double sumOfSquares = 0.0;
        for (double val : features) {
            sumOfSquares += val * val;
        }
        return Math.sqrt(sumOfSquares);
    }

    // Normalize this vector to unit length
    public void normalize(boolean auxDim) {
        if (normalized) {
            throw new IllegalStateException("Point is already normalized");
        }
        if (auxDim) {
            features.add(1.0);
        }
        double norm = norm();

        for (int i = 0; i < features.size(); i++) {
            features.set(i, features.get(i) / norm);
        }
        normalized = true;
    }

    // Add another vector to this one
    public BasePoint add(BasePoint other) {
        if (features.size() != other.features.size()) {
            throw new IllegalArgumentException("Cannot add vectors with different dimensions");
        }

        BasePoint result = new BasePoint();
        result.features = new ArrayList<>(features.size());

        for (int i = 0; i < features.size(); i++) {
            result.features.add(features.get(i) + other.features.get(i));
        }
        return result;
    }

    // Add another vector to this one in place
    public void addInPlace(BasePoint other) {
        if (features.size() != other.features.size()) {
            throw new IllegalArgumentException("Cannot add vectors with different dimensions");
        }

        for (int i = 0; i < features.size(); i++) {
            features.set(i, features.get(i) + other.features.get(i));
        }
    }

    // Subtract another vector from this one in place
    public void subtractInPlace(BasePoint other) {
        if (features.size() != other.features.size()) {
            throw new IllegalArgumentException("Cannot subtract vectors with different dimensions");
        }

        for (int i = 0; i < features.size(); i++) {
            features.set(i, features.get(i) - other.features.get(i));
        }
    }

    // Divide this vector by a scalar in place
    public void divideInPlace(double scalar) {
        if (scalar == 0.0) {
            throw new IllegalArgumentException("Division by zero");
        }

        for (int i = 0; i < features.size(); i++) {
            features.set(i, features.get(i) / scalar);
        }
    }

    // Convert to string representation
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(id).append("\t");

        for (double feature : features) {
            sb.append(String.format("%.3f", feature)).append("\t");
        }

        return sb.toString();
    }

}

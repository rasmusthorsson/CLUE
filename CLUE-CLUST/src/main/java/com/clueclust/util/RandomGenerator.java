package com.clueclust.util;

import java.util.Random;

//RandomGenerator for generating random hyperplanes
//Produces normally distributed random values

public class RandomGenerator {

    private final Random random;

    
    //Default constructor    
    public RandomGenerator() {
        this.random = new Random();
    }
    
    
    //Constructor with seed    
    public RandomGenerator(long seed) {
        this.random = new Random(seed);
    }
    
    
    //Get a normally distributed random value
    //Using Box-Muller transform    
    public double getRandomValue() {
        // Box-Muller transform for normally distributed random numbers
        double u1 = random.nextDouble(); // Uniform (0,1)
        double u2 = random.nextDouble(); // Uniform (0,1)
        
        // Standard normal distribution with mean 0 and standard deviation 1
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }

}

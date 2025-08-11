package com.clueclust.core;

import java.util.List;

//Hyperplane class used for locality-sensitive hashing
//Represents a hyperplane in the feature space

public class Hyperplane extends BasePoint{

    //Default constructor
    public Hyperplane(){
        super();
    }

    
    //Construct from list of features    
    public Hyperplane(List<Double> features) {
        super(features);
    }
    
    
    //Construct from a string representation    
    public Hyperplane(String line, long id) {
        super(line, id);
    }
}

package com.clueclust.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class HashedPoint {

    private final List<Integer> hashValues = new ArrayList<>();


    //Create empty HashedPoint
    public HashedPoint(){

    }

    //Add a hash value to this HashedPoint
    public void addHashValue(int value){
        hashValues.add(value);
    }

    //Get all hash values
    public List<Integer> getHashValues(){
        return hashValues;
    }

    //Override equals to compare hash values
    @Override
    public boolean equals(Object o) {
        if(this == o) return true;
        if(o == null || getClass() != o.getClass()) return false;
        HashedPoint that = (HashedPoint) o;

        if(hashValues.size() != that.hashValues.size()){
            return false;
        }

        for(int i = 0; i < hashValues.size(); i++){
            if(!Objects.equals(hashValues.get(i), that.hashValues.get(i))){
                return false;
            }
        }
        return true;
    }

    //Override hashCode to use the hash values
    @Override
    public int hashCode(){
        int result = 1;
        for(Integer hashValue : hashValues){
            result = 31 * result + hashValue.hashCode();
        }
        return result;
    }

    //String representation of this hashed point
    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        for(Integer hashValue : hashValues) {
            sb.append(hashValue).append(",");
        }
        return sb.toString();
    }


}

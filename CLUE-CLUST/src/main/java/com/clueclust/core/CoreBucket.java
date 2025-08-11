package com.clueclust.core;

import java.util.ArrayList;
import java.util.List;

public class CoreBucket {

    private List<Point> members = new ArrayList<>();
    private Point representative = null;

    //Default constructor
    public CoreBucket(){    
    }

    //Constructor with members and representative
    public CoreBucket(List<Point> members, Point representative) {
        this.members = members;
        this.representative = representative;
    }

    //Get the members of this corebucket
    public List<Point> getMembers() {
        return members;
    }


    //Set the members of this corebucket
    public void setMembers(List<Point> members) {
        this.members = members;
    }

    //Add member to this core bucket
    public void addMember(Point point) {
        members.add(point);
    }

    //Get the representative point of this core bucket
    public Point getRepresentative() {
        return representative;
    }

    //Set the representative point of this core bucket
    public void setRepresentative(Point representative) {
        this.representative = representative;
    }

    //String representation of this core bucket
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("CoreBucket [representative=").append(representative.getId());
        sb.append(", members=");
        for (Point p : members) {
            sb.append(p.getId()).append(" ");
        }
        sb.append("]");
        return sb.toString();
    }
}

package org.apache.sysds.runtime.einsum;

public abstract class EOpNode {
    public Character c1;
    public Character c2; // nullable
    public EOpNode(Character c1, Character c2){
        this.c1 = c1;
        this.c2 = c2;
    }

    @Override
    public String toString() {
        if(c1 == null) return "-";

        if(c2 == null) return c1.toString();
        return c1.toString() + c2.toString();
    }
}


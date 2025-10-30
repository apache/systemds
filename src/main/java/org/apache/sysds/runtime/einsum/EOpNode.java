package org.apache.sysds.runtime.einsum;

import org.apache.commons.logging.Log;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;

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

    public abstract MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numOfThreads, Log LOG);

    public abstract void reorderChildren(Character outChar1, Character outChar2);
}


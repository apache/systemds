package org.apache.sysds.runtime.einsum;

public class EOpNodeData extends EOpNode {
    public int matrixIdx;
    public EOpNodeData(Character c1, Character c2, int matrixIdx){
        super(c1,c2);
        this.matrixIdx = matrixIdx;
    }
}

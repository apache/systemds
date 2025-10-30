package org.apache.sysds.runtime.einsum;

import org.apache.commons.logging.Log;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;

public class EOpNodeData extends EOpNode {
    public int matrixIdx;
    public EOpNodeData(Character c1, Character c2, int matrixIdx){
        super(c1,c2);
        this.matrixIdx = matrixIdx;
    }

    @Override
    public MatrixBlock computeEOpNode(ArrayList<MatrixBlock> inputs, int numOfThreads, Log LOG) {
        return inputs.get(matrixIdx);
    }

    @Override
    public void reorderChildren(Character outChar1, Character outChar2) {

    }
}

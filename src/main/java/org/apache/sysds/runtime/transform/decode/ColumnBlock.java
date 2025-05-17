package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * helper class for column input
 */
public class ColumnBlock {
    // column block, part of original MatrixBlock
    public MatrixBlock data;

    // position in original MatrixBlock
    public int[] targetCols;
}

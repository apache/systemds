package org.apache.sysds.runtime.transform.encode;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.Externalizable;

public interface Encoder extends Externalizable {

    /**
     * Construct a frame block out of the transform meta data.
     *
     * @param out output frame block
     * @return output frame block?
     */
    FrameBlock getMetaData(FrameBlock out);

    /**
     * Sets up the required meta data for a subsequent call to apply.
     *
     * @param meta frame block
     */
    void initMetaData(FrameBlock meta);


    /**
     * Allocates internal data structures for partial build.
     */
    public void prepareBuildPartial();

    /**
     * Partial build of internal data structures (e.g., in distributed spark operations).
     *
     * @param in input frame block
     */
    public void buildPartial(FrameBlock in);


    /**
     * Obtain the column mapping of encoded frames based on the passed
     * meta data frame.
     *
     * @param meta meta data frame block
     * @param out output matrix
     * @return matrix with column mapping (one row per attribute)
     */
    public MatrixBlock getColMapping(FrameBlock meta, MatrixBlock out);

    /**
     * Update index-ranges to after encoding. Note that only Dummycoding changes the ranges.
     *
     * @param beginDims begin dimensions of range
     * @param endDims end dimensions of range
     */
    public void updateIndexRanges(long[] beginDims, long[] endDims);

}

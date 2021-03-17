package org.apache.sysds.runtime.transform.encode;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.Externalizable;

/*
Interface for all Encoder like objects
 */

public interface Encoder extends Externalizable {


    /**
     * Build the transform meta data for the given block input. This call modifies
     * and keeps meta data as encoder state.
     *
     * @param in input frame block
     */
    void build(FrameBlock in);

    /**
     * Apply the generated metadata to the FrameBlock and saved the result in out.
     *
     * @param in input frame block
     * @param out output matrix block
     * @param outputCol is a offset in the output matrix.
     *                  column in FrameBlock + outputCol = column in out
     * @return output matrix block
     */
    MatrixBlock apply(FrameBlock in, MatrixBlock out, int outputCol);


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
     * Update index-ranges to after encoding. Note that only Dummycoding changes the ranges.
     *
     * @param beginDims begin dimensions of range
     * @param endDims end dimensions of range
     * @param offset is applied to begin and endDims
     */
    public void updateIndexRanges(long[] beginDims, long[] endDims, int offset);

}

package org.apache.sysds.runtime.transform.encode;

import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.Externalizable;

public interface Encoder extends Externalizable {


    /**
     * Block encode: build and apply (transform encode).
     *
     * @param in input frame block
     * @param out output matrix block
     * @return output matrix block
     */
    MatrixBlock encode(FrameBlock in);

    /**
     * Build the transform meta data for the given block input. This call modifies
     * and keeps meta data as encoder state.
     *
     * @param in input frame block
     */
    void build(FrameBlock in);

    /**
     * Encode input data blockwise according to existing transform meta
     * data (transform apply).
     *
     * @param in input frame block
     * @param out output matrix block
     * @return output matrix block
     */
    MatrixBlock apply(FrameBlock in);


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
     */
    public void updateIndexRanges(long[] beginDims, long[] endDims);

}

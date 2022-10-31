/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.transform.encode;

import java.io.Externalizable;

import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/*
Interface for all Encoder like objects
 */

public interface Encoder extends Externalizable {

	/**
	 * Build the transform meta data for the given block input. This call modifies and keeps meta data as encoder state.
	 *
	 * @param in input frame block
	 */
	void build(CacheBlock in);

	/**
	 * Apply the generated metadata to the FrameBlock and saved the result in out.
	 *
	 * @param in        input frame block
	 * @param out       output matrix block
	 * @param outputCol is a offset in the output matrix. column in FrameBlock + outputCol = column in out
	 * @return output matrix block
	 */
	MatrixBlock apply(CacheBlock in, MatrixBlock out, int outputCol);
	
	/** 
	 * Pre-allocate a FrameBlock for metadata collection.
	 * @param meta      frame block
	 */
	void allocateMetaData(FrameBlock meta);

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
	void prepareBuildPartial();

	/**
	 * Partial build of internal data structures (e.g., in distributed spark operations).
	 *
	 * @param in input frame block
	 */
	void buildPartial(FrameBlock in);

	/**
	 * Update index-ranges to after encoding. Note that only Dummycoding changes the ranges.
	 *
	 * @param beginDims begin dimensions of range
	 * @param endDims   end dimensions of range
	 * @param offset    is applied to begin and endDims
	 */
	void updateIndexRanges(long[] beginDims, long[] endDims, int offset);

}

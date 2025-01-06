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

package org.apache.sysds.runtime.compress.estim.encoding;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt;

/**
 * This interface covers an intermediate encoding for the samples to improve the efficiency of the joining of sample
 * column groups.
 */
public interface IEncode {

	public static final Log LOG = LogFactory.getLog(IEncode.class.getName());

	/**
	 * Combine two encodings, note it should be guaranteed by the caller that the number of unique multiplied does not
	 * overflow Integer.
	 * 
	 * @param e The other side to combine with
	 * @return The combined encoding
	 */
	public IEncode combine(IEncode e);

	/**
	 * Combine two encodings without resizing the output. meaning the mapping of the indexes should be consistent with
	 * left hand side Dictionary indexes and right hand side indexes.
	 * <p>
	 * 
	 * 
	 * NOTE: Require both encodings to contain the correct metadata for number of unique values.
	 * 
	 * @param e The other side to combine with
	 * @return The combined encoding
	 */
	public Pair<IEncode, HashMapLongInt> combineWithMap(IEncode e);

	/**
	 * Get the number of unique values in this encoding
	 * 
	 * @return The number of unique values.
	 */
	public int getUnique();

	/**
	 * Extract the compression facts for this column group.
	 * 
	 * @param nRows          The total number of rows
	 * @param tupleSparsity  The Sparsity of the unique tuples
	 * @param matrixSparsity The matrix sparsity
	 * @param cs             The compression settings
	 * @return A EstimationFactors object
	 */
	public EstimationFactors extractFacts(int nRows, double tupleSparsity, double matrixSparsity,
		CompressionSettings cs);

	/**
	 * Signify if the counts are including zero or without zero.
	 * 
	 * @return is dense
	 */
	public abstract boolean isDense();

	@Override
	public abstract boolean equals(Object e);

	/**
	 * Indicate if the given encoding is equivalent to this encoding
	 * 
	 * @param e The other encoding to be compared with this
	 * @return If the encoding is equivalent
	 */
	public abstract boolean equals(IEncode e);
}

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

package org.apache.sysds.runtime.matrix.data.sketch.countdistinct;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.instructions.spark.data.CorrMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.sketch.CountDistinctSketch;
import org.apache.sysds.runtime.matrix.operators.Operator;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class CountDistinctFunctionSketch extends CountDistinctSketch {

	private static final Log LOG = LogFactory.getLog(CountDistinctFunctionSketch.class.getName());
	public CountDistinctFunctionSketch(Operator op) {
		super(op);
	}

	@Override
	public MatrixBlock getValue(MatrixBlock blkIn) {
		return null;
	}

	@Override
	public MatrixBlock getValueFromSketch(CorrMatrixBlock blkIn) {
		MatrixBlock blkInCorr = blkIn.getCorrection();
		MatrixBlock blkOut = new MatrixBlock(1, 1, false);

		long res = 0;
		for (int i=0; i<blkInCorr.getNumRows(); ++i) {
			res += blkInCorr.getValue(i, 1);
		}

		blkOut.setValue(0, 0, res);
		return blkOut;
	}

	@Override
	public CorrMatrixBlock create(MatrixBlock blkIn) {
		int R = blkIn.getNumRows();
		int C = blkIn.getNumColumns();

		if (R == 1 && R == C) {
			MatrixBlock blkOutCorr = new MatrixBlock(1, 2, false);
			blkOutCorr.setValue(0, 1, 1);
			return new CorrMatrixBlock(blkIn, blkOutCorr);
		}

		if (blkIn.isEmpty()) {
			MatrixBlock blkOutCorr = new MatrixBlock(1, 2, false);
			// New matrix block will be initialized to 0, which is the correct answer in this case
			return new CorrMatrixBlock(blkIn, blkOutCorr);
		}

		// Double bit repr: [ sign (1) | exponent (11) | fraction (52) ]

		// As we iterate through the input matrix block, we will perform a 2-step lookup:
		// 1. We will first index into the map using the long repr of the <sign | exponent> bits
		// 2. Then, we will perform a lookup on the set containing the fraction parts for that given key (exponent)

		// Key      -> [ sign (1) | exponent (11) ] needs 12 bits -> a short int is sufficient
		// Value    -> each [ fraction (52) ] needs a long int
		Map<Short, Set<Long>> bitMap = new HashMap<>();

		// In the worst case, blkIn can have 1000x1000 = 10e6 unique values.
		// Therefore, total memory is bounded above by:
		// Max of
		//  (10e6 * 2) + (10e6 * 8) bytes   -> 10e6 keys, each with a single value
		//  and 2 + (10e6 * 8) bytes        -> a single key, with unique 10e6 fraction parts
		// = (10e6 * 2) + (10e6 * 8) = 10e6 * 10 bytes < 10 MB

		// We have to keep track of all fraction_j in the sketch for the union() op to work for large input datasets

		int maxColumns = (int) Math.pow(OptimizerUtils.DEFAULT_BLOCKSIZE, 2);
		for (int i=0; i<R; ++i) {
			for (int j=0; j<C; ++j) {
				short key = (short) extractRightKBitsFromIndex((long) blkIn.getValue(i, j), 52, 12);
				long value = extractRightKBitsFromIndex((long) blkIn.getValue(i, j), 0, 52);

				// Update bit map with new (key, value)
				Set<Long> fractions = bitMap.getOrDefault(key, new HashSet<>());
				fractions.add(value);
				bitMap.put(key, fractions);

				maxColumns = Math.max(maxColumns, fractions.size());
			}
		}

		MatrixBlock blkOutCorr = serializeInputMatrixBlock(bitMap, maxColumns);

		// The sketch contains all relevant info, so the input matrix can be discarded at this point
		return new CorrMatrixBlock(blkIn, blkOutCorr);
	}

	private long extractRightKBitsFromIndex(long n, int startingIndex, int k) {
		long kMask = (1 << k) - 1;
		return kMask & (n >> startingIndex);
	}

	private MatrixBlock serializeInputMatrixBlock(Map<Short, Set<Long>> bitMap, int maxWidth) {

		// Each row in output matrix corresponds to a key and each column to a fraction value for that key.
		// The first column will store the exponent value itself:
		// M x N matrix: row_i: [exponent_i, fraction_i0, fraction_i1, .., fraction_iN]

		// Each key has a variable number of fraction values. To avoid a jagged matrix,
		// we will always store the size of the fractions set in the second col:
		// row_i: [exponent_i, N_i, fraction_i0, fraction_i1, .., fraction_iN, 0, .., 0]
		MatrixBlock blkOut = new MatrixBlock(bitMap.size(), maxWidth + 2, false);

		int i = 0;
		for (short key : bitMap.keySet()) {
			Set<Long> fractions = bitMap.get(key);

			blkOut.setValue(i, 0, key);
			blkOut.setValue(i, 1, fractions.size());

			int j = 2;
			for (long fraction : fractions) {
				blkOut.setValue(i, j, fraction);
				++j;
			}

			++i;
		}
		return blkOut;
	}

	@Override
	public CorrMatrixBlock union(CorrMatrixBlock arg0, CorrMatrixBlock arg1) {
		throw new NotImplementedException("MULTI_BLOCK aggregation is not supported yet");
	}

	@Override
	public CorrMatrixBlock intersection(CorrMatrixBlock arg0, CorrMatrixBlock arg1) {
		return null;
	}
}

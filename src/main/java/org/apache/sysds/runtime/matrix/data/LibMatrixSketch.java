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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

public class LibMatrixSketch {

	private enum MatrixShape {
		SKINNY,  // rows > cols
		WIDE,    // rows < cols
	}

	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir) {

		int R = blkIn.getNumRows();
		int C = blkIn.getNumColumns();
		List<HashSet<Double>> hashSets = new ArrayList<>();

		MatrixShape matrixShape = (R >= C)? MatrixShape.SKINNY : MatrixShape.WIDE;
		MatrixBlock blkOut;
		switch (dir)
		{
			case RowCol:
				HashSet<Double> hashSet = new HashSet<>();
				// TODO optimize for sparse and compressed inputs
				for (int i=0; i<R; ++i) {
					for (int j=0; j<C; ++j) {
						hashSet.add(blkIn.getValue(i, j));
					}
				}
				hashSets.add(hashSet);
				blkOut = serializeRowCol(hashSets, dir, matrixShape);
				break;

			case Row:
			case Col:
				throw new NotImplementedException("Unique Row/Col has not been implemented yet");

			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}

		return blkOut;
	}

	private static MatrixBlock serializeRowCol(List<HashSet<Double>> hashSets, Types.Direction dir, MatrixShape matrixShape) {

		if (dir != Types.Direction.RowCol) {
			throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}

		MatrixBlock blkOut;

		if (hashSets.isEmpty()) {
			throw new IllegalArgumentException("Corrupt sketch: metadata cannot be empty");
		}

		int R, C;
		HashSet<Double> hashSet = hashSets.get(0);
		Iterator<Double> iter = hashSet.iterator();

		if (hashSet.size() <= OptimizerUtils.DEFAULT_BLOCKSIZE) {
			if (matrixShape == MatrixShape.SKINNY) {
				// Rx1 column vector
				R = hashSet.size();
				C = 1;
			} else {  // WIDE
				// 1xC row vector
				R = 1;
				C = hashSet.size();
			}
		} else {
			if (matrixShape == MatrixShape.SKINNY) {
				R = OptimizerUtils.DEFAULT_BLOCKSIZE;
				C = (hashSet.size() / OptimizerUtils.DEFAULT_BLOCKSIZE) + 1;
			} else {  // WIDE
				R = (hashSet.size() / OptimizerUtils.DEFAULT_BLOCKSIZE) + 1;
				C = OptimizerUtils.DEFAULT_BLOCKSIZE;
			}
		}

		blkOut = new MatrixBlock(R, C, false);
		for (int i=0; i<R; ++i) {
			// C is guaranteed to be > 0
			for (int j=0; j<C; ++j) {
				blkOut.setValue(i, j, iter.next());
			}
		}

		return blkOut;
	}
}

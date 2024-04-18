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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.common.Types;

import java.util.HashSet;
import java.util.Iterator;

public class LibMatrixSketch {

	public static MatrixBlock getUniqueValues(MatrixBlock blkIn, Types.Direction dir) {
		//similar to R's unique, this operation takes a matrix and computes the
		//unique values (or rows in case of multiple column inputs)
		
		int rlen = blkIn.getNumRows();
		int clen = blkIn.getNumColumns();

		MatrixBlock blkOut = null;
		switch (dir) {
			case RowCol:
				if( clen != 1 )
					throw new NotImplementedException("Unique only support single-column vectors yet");
				// TODO optimize for dense/sparse/compressed (once multi-column support added)
				
				// obtain set of unique items (dense input vector)
				HashSet<Double> hashSet = new HashSet<>();
				for( int i=0; i<rlen; i++ ) {
					hashSet.add(blkIn.get(i, 0));
				}
				
				// allocate output block and place values
				int rlen2 = hashSet.size();
				blkOut = new MatrixBlock(rlen2, 1, false).allocateBlock();
				Iterator<Double> iter = hashSet.iterator();
				for( int i=0; i<rlen2; i++ ) {
					blkOut.set(i, 0, iter.next());
				}
				break;

			case Row:
			case Col:
				throw new NotImplementedException("Unique Row/Col has not been implemented yet");

			default:
				throw new IllegalArgumentException("Unrecognized direction: " + dir);
		}

		return blkOut;
	}
}

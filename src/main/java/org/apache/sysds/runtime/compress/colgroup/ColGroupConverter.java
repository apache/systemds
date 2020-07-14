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

package org.apache.sysds.runtime.compress.colgroup;

import java.util.Arrays;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;

/**
 * Utility functions for ColGroup to convert ColGroups or MatrixBlocks. to other representations.
 */
public class ColGroupConverter {
	/**
	 * Copy col group instance with deep copy of column indices but shallow copy of actual contents;
	 * 
	 * @param group column group
	 * @return column group (deep copy of indices but shallow copy of contents)
	 */
	public static ColGroup copyColGroup(ColGroup group) {
		ColGroup ret = null;

		// deep copy col indices
		int[] colIndices = Arrays.copyOf(group.getColIndices(), group.getNumCols());

		// create copy of column group
		if(group instanceof ColGroupUncompressed) {
			ColGroupUncompressed in = (ColGroupUncompressed) group;
			ret = new ColGroupUncompressed(colIndices, in._numRows, in.getData());
		}
		else if(group instanceof ColGroupRLE) {
			ColGroupRLE in = (ColGroupRLE) group;
			ret = new ColGroupRLE(colIndices, in._numRows, in.hasZeros(), in._dict, in.getBitmaps(),
				in.getBitmapOffsets());
		}
		else if(group instanceof ColGroupOLE) {
			ColGroupOLE in = (ColGroupOLE) group;
			ret = new ColGroupOLE(colIndices, in._numRows, in.hasZeros(), in._dict, in.getBitmaps(),
				in.getBitmapOffsets());
		}
		else if(group instanceof ColGroupDDC1) {
			ColGroupDDC1 in = (ColGroupDDC1) group;
			ret = new ColGroupDDC1(colIndices, in._numRows, in._dict, in.getData(), in._zeros);
		}
		else {
			throw new RuntimeException("Using '" + group.getClass() + "' instance of ColGroup not fully supported");
		}

		return ret;
	}

	/**
	 * Extract the double array primitive from a Matrix Block that is an vector.
	 * 
	 * @param vector The Matrix block vector
	 * @return The double array primitive
	 */
	public static double[] getDenseVector(MatrixBlock vector) {
		return DataConverter.convertToDoubleVector(vector, false);
	}

	/**
	 * Extracts the Uncompressed MatrixBlock representation of a Col Group
	 * 
	 * @param group an ColGroup to decompress
	 * @return A MatrixBlock.
	 */
	public static MatrixBlock getUncompressedColBlock(ColGroup group) {
		return (group instanceof ColGroupUncompressed) ? ((ColGroupUncompressed) group)
			.getData() : new ColGroupUncompressed(Arrays.asList(group)).getData();
	}
}

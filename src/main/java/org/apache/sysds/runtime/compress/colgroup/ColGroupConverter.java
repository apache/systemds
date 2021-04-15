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
	public static MatrixBlock getUncompressedColBlock(AColGroup group) {
		return (group instanceof ColGroupUncompressed) ? ((ColGroupUncompressed) group)
			.getData() : new ColGroupUncompressed(Arrays.asList(group)).getData();
	}
}

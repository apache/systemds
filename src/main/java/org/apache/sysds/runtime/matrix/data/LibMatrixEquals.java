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

/**
 * 
 * <p>
 * Equals library for MatrixBLocks:
 * </p>
 * 
 * <p>
 * The implementations adhere to the properties of equals of:
 * </p>
 * 
 * <ul>
 * <li>Reflective</li>
 * <li>Symmetric</li>
 * <li>Transitive</li>
 * <li>Consistent</li>
 * </ul>
 * 
 */
public class LibMatrixEquals {

	/** first block */
	private final MatrixBlock a;
	/** second block */
	private final MatrixBlock b;

	/**
	 * Private instance of a comparison, constructed to reduce the arguments to method calls.
	 * 
	 * @param a first block
	 * @param b second block
	 */
	private LibMatrixEquals(MatrixBlock a, MatrixBlock b) {
		this.a = a;
		this.b = b;
	}

	/**
	 * <p>
	 * Analyze if the two matrix blocks are equivalent, this functions even if the underlying allocation and data
	 * structure varies.
	 * </p>
	 * 
	 * <p>
	 * The implementations adhere to the properties of equals of:
	 * </p>
	 * 
	 * <ul>
	 * <li>Reflective</li>
	 * <li>Symmetric</li>
	 * <li>Transitive</li>
	 * <li>Consistent</li>
	 * </ul>
	 * 
	 * @param a Matrix Block a to compare
	 * @param b Matrix Block b to compare
	 * @return If the block are equivalent.
	 */
	public static boolean equals(MatrixBlock a, MatrixBlock b) {
		// Same object
		if(a == b)
			return true;
		return new LibMatrixEquals(a, b).exec();
	}

	/**
	 * Execute the comparison
	 * 
	 * @return if the blocks are equivalent
	 */
	private boolean exec() {
		if(isMetadataDifferent())
			return false;

		throw new NotImplementedException("Not implemented matrixBlock compare");
	}

	/**
	 * Compare metadata, and return true if metadata is different
	 * 
	 * @param a MatrixBlock a
	 * @param b MatrixBlock b
	 * @return If the metadata was comparable
	 */
	private boolean isMetadataDifferent() {
		boolean diff = false;

		diff |= a.getNumRows() != b.getNumRows();
		diff |= a.getNumColumns() != b.getNumColumns();
		final long nnzA = a.getNonZeros();
		final long nnzB = b.getNonZeros();
		diff |= nnzA != -1 && nnzB != -1 && nnzA != nnzB;

		return diff;
	}
}

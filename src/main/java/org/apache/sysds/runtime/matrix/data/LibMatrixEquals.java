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
		throw new NotImplementedException("Not implemented matrixBlock compare");
	}
}

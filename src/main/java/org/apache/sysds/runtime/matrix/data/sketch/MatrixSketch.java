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

package org.apache.sysds.runtime.matrix.data.sketch;

import org.apache.sysds.runtime.instructions.spark.data.CorrMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public interface MatrixSketch<T> {

	/**
	 * Get scalar distinct count from a input matrix block.
	 * 
	 * @param blkIn A input block to estimate the number of distinct values in
	 * @return The distinct count estimate
	 */
	T getScalarValue(MatrixBlock blkIn);

	/**
	 * Obtain matrix distinct count value from estimation Used for estimating distinct in rows or columns.
	 * 
	 * @param blkIn The sketch block to extract the count from
	 * @return The result matrix block
	 */
	public MatrixBlock getMatrixValue(CorrMatrixBlock blkIn);

	/**
	 * Create a initial sketch of a given block.
	 * 
	 * @param blkIn A block to process
	 * @return A sketch
	 */
	public CorrMatrixBlock create(MatrixBlock blkIn);

	/**
	 * Union two sketches together to from a combined sketch.
	 * 
	 * @param arg0 Sketch one
	 * @param arg1 Sketch two
	 * @return The combined sketch
	 */
	public CorrMatrixBlock union(CorrMatrixBlock arg0, CorrMatrixBlock arg1);

	/**
	 * Intersect two sketches
	 * 
	 * @param arg0 Sketch one
	 * @param arg1 Sketch two
	 * @return The intersected sketch
	 */
	public CorrMatrixBlock intersection(CorrMatrixBlock arg0, CorrMatrixBlock arg1);
}

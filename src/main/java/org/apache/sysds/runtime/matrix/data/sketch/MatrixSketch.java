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

public interface MatrixSketch {

	/**
	 * Get scalar distinct count from an input matrix block.
	 * 
	 * @param blkIn An input block to estimate the number of distinct values in
	 * @return The result matrix block containing the distinct count estimate
	 */
	MatrixBlock getValue(MatrixBlock blkIn);

	/**
	 * Obtain matrix distinct count value from estimation Used for estimating distinct in rows or columns.
	 * 
	 * @param blkIn The sketch block to extract the count from
	 * @return The result matrix block
	 */
	MatrixBlock getValueFromSketch(CorrMatrixBlock blkIn);

	/**
	 * Create an initial sketch of a given block.
	 * 
	 * @param blkIn A block to process
	 * @return A sketch
	 */
	CorrMatrixBlock create(MatrixBlock blkIn);

	/**
	 * Union two sketches together to from a combined sketch.
	 * 
	 * @param arg0 Sketch one
	 * @param arg1 Sketch two
	 * @return The sketch union is a sketch
	 */
	CorrMatrixBlock union(CorrMatrixBlock arg0, CorrMatrixBlock arg1);

	/**
	 * Intersect two sketches
	 * 
	 * @param arg0 Sketch one
	 * @param arg1 Sketch two
	 * @return The sketch intersection is a sketch
	 */
	CorrMatrixBlock intersection(CorrMatrixBlock arg0, CorrMatrixBlock arg1);
}

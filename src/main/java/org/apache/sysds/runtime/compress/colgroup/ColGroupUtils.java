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

import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

public class ColGroupUtils {

	/**
	 * Calculate the result of performing the binary operation on an empty row to the left
	 * 
	 * v op empty
	 * 
	 * @param op         The operator
	 * @param v          The values to use on the left side of the operator
	 * @param colIndexes The column indexes to extract
	 * @return The result as a double array.
	 */
	protected final static double[] binaryDefRowLeft(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final int len = colIndexes.length;
		final double[] ret = new double[len];
		for(int i = 0; i < len; i++)
			ret[i] = fn.execute(v[colIndexes[i]], 0);
		return ret;
	}

	/**
	 * Calculate the result of performing the binary operation on an empty row to the right
	 * 
	 * empty op v
	 * 
	 * @param op         The operator
	 * @param v          The values to use on the left side of the operator
	 * @param colIndexes The column indexes to extract
	 * @return The result as a double array.
	 */
	protected final static double[] binaryDefRowRight(BinaryOperator op, double[] v, int[] colIndexes) {
		final ValueFunction fn = op.fn;
		final int len = colIndexes.length;
		final double[] ret = new double[len];
		for(int i = 0; i < len; i++)
			ret[i] = fn.execute(0, v[colIndexes[i]]);
		return ret;
	}

}

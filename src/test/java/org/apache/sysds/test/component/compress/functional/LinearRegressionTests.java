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

package org.apache.sysds.test.component.compress.functional;

import org.apache.sysds.runtime.compress.colgroup.functional.LinearRegression;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class LinearRegressionTests {

	@Test
	public void testLinearRegression() {
		double[][] data = new double[][] {{1, 1, -3, 4, 5}, {2, 2, 3, 4, 5}, {3, 3, 3, 4, 5}};
		int[] colIndexes = new int[]{0, 1, 3, 4};
		boolean isTransposed = false;

		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		double[][] coefficients = LinearRegression.regressMatrixBlock(mbt, colIndexes, isTransposed);

		assertArrayEquals(coefficients, new double[][] {{0, 0, 4, 5}, {1, 1, 0, 0}});
	}

}

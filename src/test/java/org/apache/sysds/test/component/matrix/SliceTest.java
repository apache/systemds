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

package org.apache.sysds.test.component.matrix;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Test;

public class SliceTest {
	MatrixBlock a = genIncMatrix(10, 10);

	@Test
	public void sliceTest_01() {
		MatrixBlock b = a.slice(0, 4);
		assertEquals(5, b.getNumRows());
	}

	@Test
	public void sliceTest_02() {
		MatrixBlock b = a.slice(0, 9);
		assertEquals(10, b.getNumRows());
	}

	@Test
	public void sliceTest_03() {
		MatrixBlock b = a.slice(9, 9);
		assertEquals(1, b.getNumRows());
	}

	private static MatrixBlock gen(int[][] v) {
		return DataConverter.convertToMatrixBlock(v);
	}

	private static MatrixBlock genIncMatrix(int rows, int cols) {
		int[][] ret = new int[rows][cols];
		int x = 0;
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				ret[i][j] = x++;
			}
		}
		return gen(ret);
	}
}

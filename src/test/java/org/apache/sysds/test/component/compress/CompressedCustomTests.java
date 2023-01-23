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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public class CompressedCustomTests {
	@Test
	public void compressNaNDense() {
		MatrixBlock m = new MatrixBlock(100, 100, Double.NaN);

		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m).getLeft();

		for(int i = 0; i < m.getNumRows(); i++)
			for(int j = 0; j < m.getNumColumns(); j++)
				assertEquals(0.0, m2.quickGetValue(i, j), 0.0);
	}

	@Test
	public void compressNaNSparse() {
		MatrixBlock m = new MatrixBlock(100, 100, true);
		for(int i = 0; i < m.getNumRows(); i++)
			m.setValue(i, i, Double.NaN);
		assertTrue(m.isInSparseFormat());
		MatrixBlock m2 = CompressedMatrixBlockFactory.compress(m).getLeft();
		for(int i = 0; i < m.getNumRows(); i++)
			for(int j = 0; j < m.getNumColumns(); j++)
				assertEquals(0.0, m2.quickGetValue(i, j), 0.0);
	}
}

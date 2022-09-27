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

package org.apache.sysds.test.component.compress.combine;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.HashMap;
import java.util.Map;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.lib.CLALibCombine;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.junit.Test;

public class CombineTest {

	@Test
	public void combineEmpty() {
		CompressedMatrixBlock m1 = CompressedMatrixBlockFactory.createConstant(100, 10, 0.0);

		Map<MatrixIndexes, MatrixBlock> data = new HashMap<>();

		data.put(new MatrixIndexes(1, 1), m1);
		data.put(new MatrixIndexes(2, 1), m1);

		try {
			MatrixBlock c = CLALibCombine.combine(data, 100 * 2, 10, 100);
			assertTrue("The result is not in compressed format", c instanceof CompressedMatrixBlock);
			assertEquals(0.0, c.sum(), 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to combine empty");
		}

	}


	@Test
	public void combineConst() {
		CompressedMatrixBlock m1 = CompressedMatrixBlockFactory.createConstant(100, 10, 1.0);

		Map<MatrixIndexes, MatrixBlock> data = new HashMap<>();

		data.put(new MatrixIndexes(1, 1), m1);
		data.put(new MatrixIndexes(2, 1), m1);

		try {
			MatrixBlock c = CLALibCombine.combine(data, 100 * 2, 10, 100);
			assertTrue("The result is not in compressed format", c instanceof CompressedMatrixBlock);
			assertEquals(0.0, c.sum(), 100.0 * 10.0 * 2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to combine empty");
		}

	}

}

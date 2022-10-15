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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.lib.CLALibCombine;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CombineTest {

	protected static final Log LOG = LogFactory.getLog(CombineTest.class.getName());

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

	@Test
	public void combineDDC() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(165, 2, 1, 3, 1.0, 2514));
		CompressedMatrixBlock csb = (CompressedMatrixBlock) CompressedMatrixBlockFactory
			.compress(mb,
				new CompressionSettingsBuilder().clearValidCompression().addValidCompression(CompressionType.DDC))
			.getLeft();

		AColGroup g = csb.getColGroups().get(0);
		double sum = g.getSum(165);
		AColGroup ret = g.append(g);
		double sum2 = ret.getSum(165 * 2);
		assertEquals(sum * 2, sum2, 0.001);
		AColGroup ret2 = ret.append(g);
		double sum3 = ret2.getSum(165 * 3);
		assertEquals(sum * 3, sum3, 0.001);

	}

}

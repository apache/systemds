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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibApplySchema;
import org.apache.sysds.runtime.frame.data.lib.MatrixBlockFromFrame;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class MatrixFromFrameTest {
	protected static final Log LOG = LogFactory.getLog(MatrixFromFrameTest.class.getName());

	public final FrameBlock fb;

	public MatrixFromFrameTest(ValueType[] schema, int rows, long seed) {
		FrameBlock tmp = TestUtils.generateRandomFrameBlock(10, schema, 3214L);
		fb = FrameLibApplySchema.applySchema(tmp, schema);
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		try {
			tests.add(new Object[] {new ValueType[] {ValueType.FP64, ValueType.BOOLEAN}, 10, 1324L});
			tests.add(new Object[] {new ValueType[] {ValueType.CHARACTER, ValueType.INT32}, 10, 1324L});
			tests.add(new Object[] {new ValueType[] {ValueType.HASH64, ValueType.INT64, ValueType.BOOLEAN}, 10, 1324L});
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed constructing tests");
		}

		return tests;
	}

	@Test
	public void singleThread() {

		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, 1);
		compare(fb, mb);
	}

	@Test
	public void parallelThread() {

		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, 2);

		compare(fb, mb);
	}

	@Test
	public void dynamicThread() {

		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, -1);

		compare(fb, mb);
	}

	@Test
	public void allocatedOut() {

		MatrixBlock mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, new MatrixBlock(3, 3, true), -1);

		compare(fb, mb);
	}

	@Test
	public void allocatedOutDense() {
		MatrixBlock mb = new MatrixBlock(fb.getNumRows(), fb.getNumRows(), false);
		mb.allocateBlock();

		mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, mb, -1);

		compare(fb, mb);
	}

	@Test
	public void allocatedOutSparse() {
		MatrixBlock mb = new MatrixBlock(fb.getNumRows(), fb.getNumRows(), true);
		mb.allocateBlock();

		mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, mb, -1);

		compare(fb, mb);
	}

	@Test
	public void allocatedOutNonContinuous() {
		MatrixBlock mb = new MatrixBlock(fb.getNumRows(), fb.getNumRows(), false);
		mb.allocateBlock();
		DenseBlock spy = spy(mb.getDenseBlock());
		when(spy.isContiguous()).thenReturn(false);
		mb.setDenseBlock(spy);

		mb = MatrixBlockFromFrame.convertToMatrixBlock(fb, mb, -1);

		compare(fb, mb);
	}

	@Test
	public void testException() {
		MatrixBlock mb = new MatrixBlock(fb.getNumRows(), fb.getNumRows(), false);
		mb.allocateBlock();
		MatrixBlock spy = spy(mb);
		when(spy.getDenseBlockValues()).thenThrow(new RuntimeException());

		Exception e = assertThrows(DMLRuntimeException.class,
			() -> MatrixBlockFromFrame.convertToMatrixBlock(fb, spy, -1));

		assertTrue(e.getMessage().contains("Failed to convert FrameBlock to MatrixBlock"));
	}

	private void compare(FrameBlock fb, MatrixBlock mb) {
		for(int i = 0; i < fb.getNumRows(); i++) {
			for(int j = 0; j < fb.getNumColumns(); j++) {
				assertEquals(fb.getColumn(j).getAsNaNDouble(i), mb.get(i, j), 0.0);
			}
		}
	}
}

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

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FrameCustomTest {

	@Test
	public void castToFrame() {
		double maxp1 = ((double) Integer.MAX_VALUE) + 1.0;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, maxp1, maxp1, 1.0, 23);
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.INT64);
	}

	@Test
	public void castToFrame3() {
		double maxp1 = ((double) Integer.MAX_VALUE) - 1.0;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, maxp1, maxp1, 1.0, 23);
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.INT32);
	}

	@Test
	public void castErrorValue() {
		MatrixBlock mb = new MatrixBlock(10, 10, Double.parseDouble("2.572306572E9"));
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.INT64);

	}

	@Test
	public void castToFrame2() {
		double maxp1 = ((double) Integer.MAX_VALUE) + 1.1111;
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, maxp1, maxp1, 1.0, 23);
		FrameBlock f = DataConverter.convertToFrameBlock(mb);
		assertTrue(f.getSchema()[0] == ValueType.FP64);
	}

}

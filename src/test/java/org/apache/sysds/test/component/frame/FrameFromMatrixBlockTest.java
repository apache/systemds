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
import org.apache.sysds.runtime.frame.data.lib.FrameFromMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FrameFromMatrixBlockTest {

	@Test
	public void booleanColumn() {
		MatrixBlock mb = new MatrixBlock(10, 3, 1.0);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN);
		for(int i = 0; i < mb.getNumColumns(); i++) {
			assertTrue(fb.getColumn(i).getValueType() == ValueType.BOOLEAN);
		}
	}

	@Test
	public void booleanColumnEmpty() {
		MatrixBlock mb = new MatrixBlock(10, 3, 0.0);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN);
		for(int i = 0; i < mb.getNumColumns(); i++) {
			assertTrue(fb.getColumn(i).getValueType() == ValueType.BOOLEAN);
		}
	}

	@Test
	public void booleanColumnSparse() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 100, 1, 1, 0.2, 213);
		FrameBlock fb = FrameFromMatrixBlock.convertToFrameBlock(mb, ValueType.BOOLEAN);
		for(int i = 0; i < mb.getNumColumns(); i++) {
			assertTrue(fb.getColumn(i).getValueType() == ValueType.BOOLEAN);
		}
	}

}

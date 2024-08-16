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
import org.junit.Test;

public class FillMatrixTest {

	@Test
	public void fill1() {
		MatrixBlock a = new MatrixBlock(10, 10, true);
		a.fill(1.0);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertEquals(1.0, a.get(i, j), 0.0);
			}
		}
	}

	@Test
	public void fill2() {
		MatrixBlock a = new MatrixBlock(10, 10, 32.0);
		a.fill(1.0);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertEquals(1.0, a.get(i, j), 0.0);
			}
		}
	}

	@Test
	public void fill3() {
		MatrixBlock a = new MatrixBlock(10, 10, true);
		a.appendValue(5, 4, 2.0);
		a.appendValue(4, 2, 2.3);
		a.fill(1.0);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertEquals(1.0, a.get(i, j), 0.0);
			}
		}
	}

	@Test
	public void fill1_zero() {
		MatrixBlock a = new MatrixBlock(10, 10, true);
		a.fill(0.0);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertEquals(0.0, a.get(i, j), 0.0);
			}
		}
	}

	@Test
	public void fill2_zero() {
		MatrixBlock a = new MatrixBlock(10, 10, 32.0);
		a.fill(0.0);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertEquals(0.0, a.get(i, j), 0.0);
			}
		}
	}

	@Test
	public void fill3_zero() {
		MatrixBlock a = new MatrixBlock(10, 10, true);
		a.appendValue(5, 4, 2.0);
		a.appendValue(4, 2, 2.3);
		a.fill(0.0);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertEquals(0.0, a.get(i, j), 0.0);
			}
		}
	}
}

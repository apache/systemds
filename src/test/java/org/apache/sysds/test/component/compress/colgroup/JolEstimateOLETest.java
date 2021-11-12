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

package org.apache.sysds.test.component.compress.colgroup;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.CompressibleInputGenerator;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class JolEstimateOLETest extends JolEstimateTest {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		MatrixBlock mb;
		// base tests
		mb = DataConverter.convertToMatrixBlock(new double[][] {{1}});
		tests.add(new Object[] {mb});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0}});
		tests.add(new Object[] {mb});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 0}});
		tests.add(new Object[] {mb});

		// The size of the compression increase at repeated values.
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 0}});
		tests.add(new Object[] {mb});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 0}});
		tests.add(new Object[] {mb});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 5, 0}});
		tests.add(new Object[] {mb});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 5, 5, 5, 5}});
		tests.add(new Object[] {mb});

		// all values grow by 1 if new value is introduced
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 7, 0}});
		tests.add(new Object[] {mb});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 2, 1, 0}});
		tests.add(new Object[] {mb});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 2, 1, 3, 6, 7}});
		tests.add(new Object[] {mb});

		// Dense random... Horrible compression at full precision
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 100, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 1000, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 10000, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb});

		// Random rounded numbers dense
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 1523, 0, 99, 1.0, 7));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 4000, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb});

		// Sparse rounded numbers
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 1523, 0, 99, 0.1, 7));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 1621, 0, 99, 0.1, 142));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 2321, 0, 99, 0.1, 512));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 4000, 0, 100, 0.1, 7));
		tests.add(new Object[] {mb});

		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 1523, 0, 99, 0.5, 7));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 1621, 0, 99, 0.5, 142));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 2321, 0, 99, 0.5, 512));
		tests.add(new Object[] {mb});
		mb = TestUtils.round(TestUtils.generateTestMatrixBlock(1, 4000, 0, 100, 0.5, 7));
		tests.add(new Object[] {mb});

		// Paper
		mb = DataConverter
			.convertToMatrixBlock(new double[][] {{7, 3, 7, 7, 3, 7, 3, 3, 7, 3}, {6, 4, 6, 5, 4, 5, 4, 4, 6, 4}});
		tests.add(new Object[] {mb});

		// Dream Inputs
		int[] cols = new int[] {2, 6, 111};
		int[] rows = new int[] {10, 121, 513};
		int[] unique = new int[] {3, 5};
		for(int y : cols) {
			for(int x : rows) {
				for(int u : unique) {
					mb = CompressibleInputGenerator.getInput(x, y, CompressionType.OLE, u, 1.0, 5);
					tests.add(new Object[] {mb});
				}
			}
		}

		// Sparse test.
		mb = CompressibleInputGenerator.getInput(571, 1, CompressionType.OLE, 40, 0.6, 5);
		tests.add(new Object[] {mb});

		// Multi Columns
		mb = CompressibleInputGenerator.getInput(412,5,CompressionType.OLE, 20, 0.4, 5);
		tests.add(new Object[] {mb});
		mb = CompressibleInputGenerator.getInput(1000,5,CompressionType.OLE, 20, 0.4, 5);
		tests.add(new Object[] {mb});

		return tests;
	}

	public JolEstimateOLETest(MatrixBlock mb) {
		super(mb);
	}

	@Override
	public CompressionType getCT() {
		return ole;
	}
}
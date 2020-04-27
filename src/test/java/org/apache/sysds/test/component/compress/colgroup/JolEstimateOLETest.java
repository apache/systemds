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

import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
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
		tests.add(new Object[] {mb, new int[]{1, 2, 2, 1}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0}});
		tests.add(new Object[] {mb, new int[]{0, 1, 0, 0}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 0}});
		tests.add(new Object[] {mb, new int[]{0, 1, 0, 0}, 0});

		// The size of the compression increase at repeated values.
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 0}});
		tests.add(new Object[] {mb, new int[]{1, 2, 2, 1}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 0}});
		tests.add(new Object[] {mb, new int[]{1, 2, 3, 1}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 5, 0}});
		tests.add(new Object[] {mb, new int[]{1, 2, 4, 1}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 5, 5, 5, 5}});
		tests.add(new Object[] {mb, new int[]{1, 2, 7, 1}, 0});

		// all values grow by 1 if new value is introduced
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 7, 0}});
		tests.add(new Object[] {mb, new int[]{2, 3, 4, 2}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 2, 1, 0}});
		tests.add(new Object[] {mb, new int[]{3, 4, 6, 3}, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 2, 1, 3, 6, 7}});
		tests.add(new Object[] {mb, new int[]{6, 7, 12, 6}, 0});

		// Dense random... Horrible compression
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 100, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb, new int[]{100, 100 + 1, 200, 100}, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 1000, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb, new int[]{1000, 1000 + 1, 2000, 1000}, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 10000, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb, new int[]{10000, 10000 + 1, 20000, 10000}, 0});

		// Random rounded numbers dense
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1523, 0, 99, 1.0, 7)));
		tests.add(new Object[] {mb, new int[]{99, 100, 1616, 99}, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 255, 1.0, 7)));
		tests.add(new Object[] {mb, new int[]{255, 256, 4250, 255}, 0});

		// Sparse rounded numbers
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1523, 0, 99, 0.1, 7)));
		tests.add(new Object[] {mb, new int[]{76, 77, 225, 76}, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1621, 0, 99, 0.1, 142)));
		tests.add(new Object[] {mb, new int[]{81, 82, 238, 81}, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 2321, 0, 99, 0.1, 512)));
		tests.add(new Object[] {mb, new int[]{92, 93, 332, 92}, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 255, 0.1, 7)));
		tests.add(new Object[] {mb, new int[]{195, 196, 573, 195}, 0});

		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1523, 0, 99, 0.5, 7)));
		tests.add(new Object[] {mb, new int[]{98, 99, 826, 99}, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1621, 0, 99, 0.5, 142)));
		tests.add(new Object[] {mb, new int[]{99, 100, 913, 99}, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 2321, 0, 99, 0.5, 512)));
		tests.add(new Object[] {mb, new int[]{99, 100, 1292, 99}, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 255, 0.5, 7)));
		tests.add(new Object[] {mb, new int[]{255, 256, 2208, 255}, 0});

		// Paper
		mb = DataConverter
			.convertToMatrixBlock(new double[][] {{7, 3, 7, 7, 3, 7, 3, 3, 7, 3}, {6, 4, 6, 5, 4, 5, 4, 4, 6, 4}});
		tests.add(new Object[] {mb, new int[]{6, 4, 13, 3}, 0});

		// Dream Inputs
		int[] cols = new int[]{2,6,111};
		int[] rows = new int[]{10,121,513};
		int[] unique = new int[]{3,5};
		for(int y : cols){
			for (int x : rows){
				for (int u : unique){
					mb = CompressibleInputGenerator.getInput(x, y, CompressionType.OLE, u, 1.0, 5);
					tests.add(new Object[] {mb, new int[]{u * y, u + 1, x + u , u}, 0});
				}
			}
		}

		// Sparse test.
		mb = CompressibleInputGenerator.getInput(571, 1, CompressionType.OLE, 40, 0.6, 5);
		tests.add(new Object[] {mb, new int[]{40 * 1, 40 + 1, ((571 + 40) / 10 ) * 6 ,40}, 0});

		return tests;
	}

	public JolEstimateOLETest(MatrixBlock mb, int[] sizes, int tolerance) {
		super(mb, sizes, tolerance);
	}

	@Override
	public CompressionType getCT() {
		return ole;
	}
}
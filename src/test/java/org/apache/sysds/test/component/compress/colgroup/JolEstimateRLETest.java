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
public class JolEstimateRLETest extends JolEstimateTest {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		MatrixBlock mb;
		mb = DataConverter.convertToMatrixBlock(new double[][] {{1}});
		tests.add(new Object[] {mb, 0});

		// The size of the compression is the same even at different numbers of repeated values.
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 0}});
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 0}});
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 5, 0}});
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(new double[][] {{0, 0, 0, 0, 5, 5, 5, 5, 5, 5}});
		tests.add(new Object[] {mb, 0});

		// Worst case all random numbers dense.
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 100, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 1000, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(1, 10000, 0, 100, 1.0, 7));
		tests.add(new Object[] {mb, 0});

		// Random rounded numbers dense
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1523, 0, 99, 1.0, 7)));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 255, 1.0, 7)));
		tests.add(new Object[] {mb, 0});

		// Sparse rounded numbers
		// Scale directly with sparsity
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1523, 0, 99, 0.1, 7)));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1621, 0, 99, 0.1, 142)));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 2321, 0, 99, 0.1, 512)));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 255, 0.1, 7)));
		tests.add(new Object[] {mb, 250});

		// Medium sparsity
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1523, 0, 99, 0.5, 7)));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 1621, 0, 99, 0.5, 142)));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter
			.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 2321, 0, 99, 0.5, 512)));
		tests.add(new Object[] {mb, 0});
		mb = DataConverter.convertToMatrixBlock(TestUtils.round(TestUtils.generateTestMatrix(1, 4000, 0, 255, 0.5, 7)));
		tests.add(new Object[] {mb, 0});

		// Dream inputs.
		// 1 unique value
		mb = CompressibleInputGenerator.getInput(10000, 1, CompressionType.RLE, 1, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		// when the rows length is larger than overflowing the character value,
		// the run gets split into two
		// char overflows into the next position increasing size by 1 char.
		int charMax = Character.MAX_VALUE;
		mb = CompressibleInputGenerator.getInput(charMax, 1, CompressionType.RLE, 1, 1.0, 132);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(charMax + 1, 1, CompressionType.RLE, 1, 1.0, 132);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(charMax * 2 + 1, 1, CompressionType.RLE, 1, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		// 10 unique values ordered such that all 10 instances is in the same run.
		// Results in same size no matter the number of original rows.
		mb = CompressibleInputGenerator.getInput(100, 1, CompressionType.RLE, 10, 1.0, 1);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(1000, 1, CompressionType.RLE, 10, 1.0, 1312);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(10000, 1, CompressionType.RLE, 10, 1.0, 14512);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(100000, 1, CompressionType.RLE, 10, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		// Sparse Dream inputs.
		mb = CompressibleInputGenerator.getInput(100, 1, CompressionType.RLE, 10, 0.1, 1);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(1000, 1, CompressionType.RLE, 10, 0.1, 1312);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(10000, 1, CompressionType.RLE, 10, 0.1, 14512);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(100000, 1, CompressionType.RLE, 10, 0.1, 132);
		tests.add(new Object[] {mb, 0});
		mb = CompressibleInputGenerator.getInput(1000000, 1, CompressionType.RLE, 10, 0.1, 132);
		tests.add(new Object[] {mb, 0});

		mb = CompressibleInputGenerator.getInput(1000000, 1, CompressionType.RLE, 1, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		// Multi Column
		// two identical columns
		mb = CompressibleInputGenerator.getInput(10, 2, CompressionType.RLE, 2, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		mb = CompressibleInputGenerator.getInput(10, 6, CompressionType.RLE, 2, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		mb = CompressibleInputGenerator.getInput(10, 100, CompressionType.RLE, 2, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		mb = CompressibleInputGenerator.getInput(101, 17, CompressionType.RLE, 2, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		mb = CompressibleInputGenerator.getInput(101, 17, CompressionType.RLE, 3, 1.0, 132);
		tests.add(new Object[] {mb, 0});

		return tests;
	}

	public JolEstimateRLETest(MatrixBlock mb, int tolerance) {
		super(mb, tolerance);
	}

	@Override
	public CompressionType getCT() {
		return rle;
	}
}
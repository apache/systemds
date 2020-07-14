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

import static org.junit.Assert.assertTrue;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;

public class TestBase {

	protected ValueType valType;
	protected ValueRange valRange;

	protected int rows;
	protected int cols;
	protected int min;
	protected int max;
	protected int seed = 7;
	protected double samplingRatio;
	protected double sparsity;

	protected CompressionSettings compressionSettings;

	// Input
	protected double[][] input;
	protected MatrixBlock mb;

	public TestBase(SparsityType sparType, ValueType valType, ValueRange valueRange,
		CompressionSettings compressionSettings, MatrixTypology MatrixTypology) {

		this.sparsity = TestConstants.getSparsityValue(sparType);
		this.rows = TestConstants.getNumberOfRows(MatrixTypology);
		this.cols = TestConstants.getNumberOfColumns(MatrixTypology);

		this.max = TestConstants.getMaxRangeValue(valueRange);
		this.min = TestConstants.getMinRangeValue(valueRange);

		try {
			switch(valType) {
				case CONST:
					this.min = this.max;
					// Do not Break, utilize the RAND afterwards.
				case RAND:
					this.input = TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, 7);
					break;
				case RAND_ROUND:
					this.input = TestUtils.round(TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, 7));
					break;
				case OLE_COMPRESSIBLE:
					// Note the Compressible Input generator, generates an already Transposed input
					// normally, therefore last argument is true, to build a non transposed matrix.
					this.input = CompressibleInputGenerator.getInputDoubleMatrix(rows,
						cols,
						CompressionType.OLE,
						(max - min) / 10,
						max,
						min,
						sparsity,
						7,
						true);
					break;
				case RLE_COMPRESSIBLE:
					this.input = CompressibleInputGenerator.getInputDoubleMatrix(rows,
						cols,
						CompressionType.RLE,
						(max - min) / 10,
						max,
						min,
						sparsity,
						7,
						true);
					break;
				default:
					throw new NotImplementedException("Not Implemented Test Value type input generator");
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Error in construction of input Test Base", false);
		}

		this.valRange = valueRange;
		this.valType = valType;
		this.compressionSettings = compressionSettings;

		mb = DataConverter.convertToMatrixBlock(this.input);
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();

		builder.append("args: ");

		builder.append(String.format("%6s%14s", "Vt:", valType));
		builder.append(String.format("%6s%8s", "Vr:", valRange));
		builder.append(String.format("%6s%5s", "Rows:", rows));
		builder.append(String.format("%6s%5s", "Cols:", cols));
		builder.append(String.format("%6s%12s", "Min:", min));
		builder.append(String.format("%6s%12s", "Max:", max));
		builder.append(String.format("%6s%5s", "Spar:", sparsity));
		builder.append(String.format("%6s%8s", "CP:", compressionSettings));

		return builder.toString();
	}
}

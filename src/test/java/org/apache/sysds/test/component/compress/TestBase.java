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

import java.util.Collection;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;

public class TestBase {

	protected final ValueType valType;
	protected final ValueRange valRange;

	protected final int rows;
	protected int cols;
	protected final int min;
	protected final int max;
	protected final int seed = 7;
	protected final double sparsity;

	protected CompressionSettingsBuilder _csb;
	protected CompressionSettings _cs;
	protected OverLapping overlappingType;

	protected Collection<CompressionType> _ct;

	// Input
	protected MatrixBlock mb;

	public TestBase(SparsityType sparType, ValueType valType, ValueRange valueRange,
		CompressionSettingsBuilder compressionSettings, MatrixTypology MatrixTypology, OverLapping ov,
		Collection<CompressionType> ct) {

		this.sparsity = TestConstants.getSparsityValue(sparType);
		this.rows = TestConstants.getNumberOfRows(MatrixTypology);
		this.cols = TestConstants.getNumberOfColumns(MatrixTypology);

		this.max = TestConstants.getMaxRangeValue(valueRange);
		this.min = (valType == ValueType.CONST) ? this.max : TestConstants.getMinRangeValue(valueRange);
		this._ct = ct;
		this.overlappingType = ov;
		this.valRange = valueRange;
		this.valType = valType;
		this._csb = compressionSettings;
		this._cs = compressionSettings != null ? compressionSettings.create() : null;
		try {

			switch(valType) {
				case CONST:
				case RAND:
					mb = TestUtils.generateTestMatrixBlock(rows, cols, min, max, sparsity, seed);
					break;
				case RAND_ROUND:
					mb = TestUtils.round(TestUtils.generateTestMatrixBlock(rows, cols, min, max, sparsity, seed));
					break;
				case OLE_COMPRESSIBLE:
					// Note the Compressible Input generator, generates an already Transposed input
					// normally, therefore last argument is true, to build a non transposed matrix.
					mb = CompressibleInputGenerator.getInputDoubleMatrix(rows, cols, CompressionType.OLE,
						Math.min((max - min), 10), max, min, sparsity, seed, false);
					break;
				case RLE_COMPRESSIBLE:
					mb = CompressibleInputGenerator.getInputDoubleMatrix(rows, cols, CompressionType.RLE,
						Math.min((max - min), 10), max, min, sparsity, seed, false);
					break;
				case UNBALANCED_SPARSE:
					mb = CompressibleInputGenerator.getUnbalancedSparseMatrix(rows, cols, Math.min((max - min), 10), max,
						min, seed);
					cols = mb.getNumColumns();
					break;
				case ONE_HOT:
					mb = CompressibleInputGenerator.getInputOneHotMatrix(rows, cols, seed);
					break;
				default:
					throw new NotImplementedException("Not Implemented Test Value type input generator");
			}

			mb.recomputeNonZeros();
			mb.examSparsity();

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Error in construction of input Test Base", false);
		}
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();

		builder.append("\nargs: ");
		builder.append(String.format("%6s%14s", "Vt:", valType));
		builder.append(String.format("%6s%8s", "Vr:", valRange));
		builder.append(String.format("%6s%5s", "Rows:", rows));
		builder.append(String.format("%6s%5s", "Cols:", cols));
		builder.append(String.format("%6s%12s", "Min:", min));
		builder.append(String.format("%6s%12s", "Max:", max));
		builder.append(String.format("%6s%5s", "Spar:", sparsity));
		builder.append(String.format("%6s%5s", "OV:", overlappingType));
		if(_cs != null)
			builder.append(String.format("%6s\n%8s", "CP:", _cs));
		else
			builder.append(String.format("%8s%5s", "FORCED:", _ct));
		return builder.toString();
	}
}

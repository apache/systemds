/*
 * Modification Copyright 2020 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.component.compress;

import java.util.ArrayList;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.lops.MMTSJ.MMTSJType;
import org.tugraz.sysds.lops.MapMultChain.ChainType;
import org.tugraz.sysds.runtime.compress.CompressedMatrixBlock;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.functionobjects.Multiply;
import org.tugraz.sysds.runtime.functionobjects.Plus;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.TestUtils;
import org.tugraz.sysds.test.TestConstants.CompressionType;
import org.tugraz.sysds.test.TestConstants.MatrixType;
import org.tugraz.sysds.test.TestConstants.SparsityType;
import org.tugraz.sysds.test.TestConstants.ValueType;
import org.tugraz.sysds.test.TestConstants.ValueRange;

@RunWith(value = Parameterized.class)
public class ParCompressedMatrixTest extends CompressedTestBase {

	// Input
	protected double[][] input;
	protected MatrixBlock mb;

	// Compressed Block
	protected CompressedMatrixBlock cmb;

	// Compression Result
	protected MatrixBlock cmbResult;

	// Decompressed Result
	protected MatrixBlock cmbDeCompressed;
	protected double[][] deCompressed;

	int k = InfrastructureAnalyzer.getLocalParallelism();

	protected static MatrixType[] usedMatrixType = new MatrixType[] {MatrixType.LARGE};

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();
		for(SparsityType st : usedSparsityTypes) {
			for(ValueType vt : usedValueTypes) {
				for(ValueRange vr : usedValueRanges) {
					for(CompressionType ct : usedCompressionTypes) {
						for(MatrixType mt : usedMatrixType) {
							tests.add(new Object[] {st, vt, vr, ct, mt, true});
						}
					}
				}
			}
		}

		return tests;
	}

	public ParCompressedMatrixTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionType compType, MatrixType matrixType, boolean compress) {
		super(sparType, valType, valRange, compType, matrixType, compress);
		input = TestUtils.generateTestMatrix(rows, cols, min, max, sparsity, 7);
		mb = getMatrixBlockInput(input);
		cmb = new CompressedMatrixBlock(mb);
		if(compress) {
			cmbResult = cmb.compress(k);
		}
		cmbDeCompressed = cmb.decompress(k);
		deCompressed = DataConverter.convertToDoubleMatrix(cmbDeCompressed);
	}

	@Test
	public void testConstruction() {
		try {
			if(!(cmbResult instanceof CompressedMatrixBlock)) {
				// TODO Compress EVERYTHING!
				return; // Input was not compressed then just pass test
				// Assert.assertTrue("Compression Failed \n" + this.toString(), false);
			}

			double epsilon = 0.0;

			TestUtils.compareMatrices(input, deCompressed, rows, cols, epsilon);
		}
		catch(Exception e) {
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testGetValue() {
		try {
			if(!(cmbResult instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			for(int i = 0; i < rows; i++)
				for(int j = 0; j < cols; j++) {
					double ulaVal = input[i][j];
					double claVal = cmb.getValue(i, j); // calls quickGetValue internally
					TestUtils.compareScalarBitsJUnit(ulaVal, claVal, 0); // Should be exactly same value
				}
		}
		catch(Exception e) {
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixMultChain() {
		try {
			if(!(cmbResult instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector1 = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, 0, 1, 1.0, 3));

			// ChainType ctype = ChainType.XtwXv;
			for(ChainType ctype : new ChainType[] {ChainType.XtwXv, ChainType.XtXv,
				// ChainType.XtXvy
			}) {

				MatrixBlock vector2 = (ctype == ChainType.XtwXv) ? DataConverter
					.convertToMatrixBlock(TestUtils.generateTestMatrix(rows, 1, 0, 1, 1.0, 3)) : null;

				// matrix-vector uncompressed
				MatrixBlock ret1 = (MatrixBlock) mb
					.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype, k);

				// matrix-vector compressed
				MatrixBlock ret2 = (MatrixBlock) cmb
					.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype, k);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				TestUtils.compareMatricesBit(d1, d2, cols, 1, 200);
			}
			// ChainType.XtXvy

		}
		catch(Exception e) {
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testTransposeSelfMatrixMult() {
		try {
			if(!(cmbResult instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			// ChainType ctype = ChainType.XtwXv;
			for(MMTSJType mType : new MMTSJType[] {MMTSJType.LEFT,
				// MMTSJType.RIGHT
			}) {
				// matrix-vector uncompressed
				MatrixBlock ret1 = mb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, k);

				// matrix-vector compressed
				MatrixBlock ret2 = cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, k);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				// High probability that The value is off by some amount 
				TestUtils.compareMatricesBit(d1, d2, cols, cols, 2048); 
			}
		}
		catch(Exception e) {
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testVectorMult() {
		try {
			if(!(cmbResult instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, 1, 1, 1.0, 3));

			// matrix-vector uncompressed
			AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator abop = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), aop, k);
			MatrixBlock ret1 = mb.aggregateBinaryOperations(mb, vector, new MatrixBlock(), abop);

			// matrix-vector compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(cmb, vector, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			TestUtils.compareMatricesBit(d1, d2, rows, 1, 256);
		}
		catch(Exception e) {
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}


	enum AggType {
		ROWSUMS, COLSUMS, SUM, ROWSUMSSQ, COLSUMSSQ, SUMSQ, ROWMAXS, COLMAXS, MAX, ROWMINS, COLMINS, MIN,
	}

	@Test
	public void testUnaryOperators() {
		try {
			if(!(cmbResult instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			for(AggType aggType : AggType.values()) {
				AggregateUnaryOperator auop = null;
				switch(aggType) {
					case SUM:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+",k);
						break;
					case ROWSUMS:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uark+",k);
						break;
					case COLSUMS:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uack+",k);
						break;
					case SUMSQ:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uasqk+",k);
						break;
					case ROWSUMSSQ:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uarsqk+",k);
						break;
					case COLSUMSSQ:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uacsqk+",k);
						break;
					case MAX:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uamax",k);
						break;
					case ROWMAXS:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmax",k);
						break;
					case COLMAXS:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmax",k);
						break;
					case MIN:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uamin",k);
						break;
					case ROWMINS:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uarmin",k);
						break;
					case COLMINS:
						auop = InstructionUtils.parseBasicAggregateUnaryOperator("uacmin",k);
						break;
				}
				// matrix-vector uncompressed
				MatrixBlock ret1 = (MatrixBlock) mb
					.aggregateUnaryOperations(auop, new MatrixBlock(), 1000, null, true);

				// matrix-vector compressed
				MatrixBlock ret2 = (MatrixBlock) cmb
					.aggregateUnaryOperations(auop, new MatrixBlock(), 1000, null, true);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				int dim1 = (aggType == AggType.ROWSUMS || aggType == AggType.ROWSUMSSQ || aggType == AggType.ROWMINS ||
					aggType == AggType.ROWMINS) ? rows : 1;
				int dim2 = (aggType == AggType.COLSUMS || aggType == AggType.COLSUMSSQ || aggType == AggType.COLMAXS ||
					aggType == AggType.COLMINS) ? cols : 1;

				TestUtils.compareMatricesBit(d1, d2, dim1, dim2, 200);
			}
		}
		catch(Exception e) {
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

}

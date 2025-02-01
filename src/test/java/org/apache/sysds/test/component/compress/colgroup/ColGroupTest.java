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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.AMorphingMMColGroup;
import org.apache.sysds.runtime.compress.colgroup.APreAgg;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupRLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCSingleZeros;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.lib.CLALibLeftMultBy;
import org.apache.sysds.runtime.data.DenseBlockFP64;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.lib.CLALibSelectionMultTest;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

/**
 * Basic idea is that we specify a list of compression schemes that a input is allowed to be compressed into. The test
 * verify that these all does the same on a given input. Base on the api for a columnGroup.
 */
@RunWith(value = Parameterized.class)
public class ColGroupTest extends ColGroupBase {
	protected static final Log LOG = LogFactory.getLog(ColGroupTest.class.getName());

	public ColGroupTest(AColGroup base, AColGroup other, int nRow) {
		super(base, other, nRow);
	}

	@Test
	public void getColIndices() {
		assertTrue(base.getColIndices().equals(other.getColIndices()));
	}

	@Test
	public void getNumCols() {
		assertEquals(base.getNumCols(), other.getNumCols());
	}

	@Test
	public void decompressToSparseBlock() {
		MatrixBlock ot = sparseMB(maxCol);
		MatrixBlock bt = sparseMB(maxCol);
		decompressToSparseBlock(ot, bt, 0, nRow);
	}

	@Test
	public void decompressToSparseBlockBigger() {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		decompressToSparseBlock(ot, bt, 0, nRow);
	}

	@Test
	public void decompressToSparseBlockBiggerSubPart() {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		decompressToSparseBlock(ot, bt, nRow / 2, nRow - 5);
	}

	@Test
	public void decompressToSparseBlockBiggerStart() {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		decompressToSparseBlock(ot, bt, 1, 10);
	}

	@Test
	public void decompressToSparseBlockStart() {
		MatrixBlock ot = sparseMB(maxCol);
		MatrixBlock bt = sparseMB(maxCol);
		decompressToSparseBlock(ot, bt, 1, 10);
	}

	@Test
	public void decompressToSparseBlockStartSmall() {
		MatrixBlock ot = sparseMB(maxCol);
		MatrixBlock bt = sparseMB(maxCol);
		decompressToSparseBlock(ot, bt, 1, 3);
	}

	@Test
	public void decompressToSparseBlockStartSingle() {
		MatrixBlock ot = sparseMB(maxCol);
		MatrixBlock bt = sparseMB(maxCol);
		decompressToSparseBlock(ot, bt, 2, 3);
	}

	@Test
	public void decompressToSparseBlockBiggerEnd() {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		decompressToSparseBlock(ot, bt, nRow - 10, nRow - 2);
	}

	@Test
	public void decompressToSparseBlockEnd() {
		MatrixBlock ot = sparseMB(maxCol);
		MatrixBlock bt = sparseMB(maxCol);
		decompressToSparseBlock(ot, bt, nRow - 10, nRow - 2);
	}

	private void decompressToSparseBlock(MatrixBlock ot, MatrixBlock bt, int rl, int ru) {
		decompressToSparseBlock(base, other, ot, bt, rl, ru);
	}

	private void decompressToSparseBlock(AColGroup a, AColGroup b, MatrixBlock ot, MatrixBlock bt, int rl, int ru) {
		try {
			a.decompressToSparseBlock(ot.getSparseBlock(), rl, ru);
			b.decompressToSparseBlock(bt.getSparseBlock(), rl, ru);
			compare(ot, bt);
		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("should never be called")) {
				e.printStackTrace();
				fail("Failed to decompress " + e);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to decompress " + e);
		}
	}

	@Test
	public void decompressToSparseBlockBiggerSubPartOffset() {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		compareDecompressSubPartOffsetSparse(ot, bt, nRow / 2, nRow - 5, 3, 3);
	}

	@Test
	public void decompressToSparseBlockBiggerSubPartOffsetCol() {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		compareDecompressSubPartOffsetSparse(ot, bt, nRow / 2, nRow - 5, 0, 3);
	}

	@Test
	public void decompressToSparseBlockBiggerSubPartOffsetRow() {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		compareDecompressSubPartOffsetSparse(ot, bt, nRow / 2, nRow - 5, 3, 0);
	}

	private void compareDecompressSubPartOffsetSparse(AColGroup a, AColGroup b) {
		MatrixBlock ot = sparseMB(maxCol + 3);
		MatrixBlock bt = sparseMB(maxCol + 3);
		compareDecompressSubPartOffsetSparse(a, b, ot, bt, nRow / 2, nRow - 5, 3, 0);
	}

	private void compareDecompressSubPartOffsetSparse(MatrixBlock ot, MatrixBlock bt, int rl, int ru, int offR,
		int offC) {
		compareDecompressSubPartOffsetSparse(base, other, ot, bt, rl, ru, offR, offC);
	}

	private static void compareDecompressSubPartOffsetSparse(AColGroup a, AColGroup b, MatrixBlock ot, MatrixBlock bt,
		int rl, int ru, int offR, int offC) {
		try {
			a.decompressToSparseBlock(ot.getSparseBlock(), rl, ru, offR, offC);
			b.decompressToSparseBlock(bt.getSparseBlock(), rl, ru, offR, offC);
			compare(ot, bt);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to decompress " + e);
		}
	}

	@Test
	public void decompressToDenseBlock() {
		MatrixBlock ot = denseMB(maxCol);
		MatrixBlock bt = denseMB(maxCol);
		decompressToDenseBlock(ot, bt, 0, nRow);
	}

	@Test
	public void decompressToDenseBlockBigger() {
		MatrixBlock ot = denseMB(maxCol + 3);
		MatrixBlock bt = denseMB(maxCol + 3);
		decompressToDenseBlock(ot, bt, 0, nRow);
	}

	@Test
	public void decompressToDenseBlockBiggerSubPart() {
		MatrixBlock ot = denseMB(maxCol + 3);
		MatrixBlock bt = denseMB(maxCol + 3);
		decompressToDenseBlock(ot, bt, nRow / 2, nRow - 5);
	}

	@Test
	public void decompressToDenseBlockBiggerStart() {
		MatrixBlock ot = denseMB(maxCol + 3);
		MatrixBlock bt = denseMB(maxCol + 3);
		decompressToDenseBlock(ot, bt, 1, 10);
	}

	@Test
	public void decompressToDenseBlockStart() {
		MatrixBlock ot = denseMB(maxCol);
		MatrixBlock bt = denseMB(maxCol);
		decompressToDenseBlock(ot, bt, 1, 10);
	}

	@Test
	public void decompressToDenseBlockStartSmall() {
		MatrixBlock ot = denseMB(maxCol);
		MatrixBlock bt = denseMB(maxCol);
		decompressToDenseBlock(ot, bt, 1, 3);
	}

	@Test
	public void decompressToDenseBlockStartSingle() {
		MatrixBlock ot = denseMB(maxCol);
		MatrixBlock bt = denseMB(maxCol);
		decompressToDenseBlock(ot, bt, 2, 3);
	}

	@Test
	public void decompressToDenseBlockBiggerEnd() {
		MatrixBlock ot = denseMB(maxCol + 3);
		MatrixBlock bt = denseMB(maxCol + 3);
		decompressToDenseBlock(ot, bt, nRow - 5, nRow - 1);
	}

	@Test
	public void decompressToDenseBlockEnd() {
		MatrixBlock ot = denseMB(maxCol);
		MatrixBlock bt = denseMB(maxCol);
		decompressToDenseBlock(ot, bt, nRow - 5, nRow - 1);
	}

	@Test
	public void decompressToMultiBlockDenseBlock() {
		MatrixBlock ot = multiBlockDenseMB(maxCol);
		MatrixBlock bt = multiBlockDenseMB(maxCol);
		decompressToDenseBlock(ot, bt, nRow - 5, nRow - 1);
	}

	private void decompressToDenseBlock(MatrixBlock ot, MatrixBlock bt, int rl, int ru) {
		decompressToDenseBlock(ot, bt, other, base, rl, ru);
	}

	private static void decompressToDenseBlock(MatrixBlock ot, MatrixBlock bt, AColGroup a, AColGroup b, int rl,
		int ru) {
		try {
			a.decompressToDenseBlock(ot.getDenseBlock(), rl, ru);
			b.decompressToDenseBlock(bt.getDenseBlock(), rl, ru);
			compare(ot, bt);
		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("should never be called")) {
				e.printStackTrace();
				fail("Failed to decompress " + e);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to decompress " + e);
		}
	}

	@Test
	public void decompressToDenseBlockBiggerSubPartOffset() {
		try {
			MatrixBlock ot = denseMB(maxCol + 3);
			MatrixBlock bt = denseMB(maxCol + 3);
			base.decompressToDenseBlock(ot.getDenseBlock(), nRow / 2, nRow - 5, 3, 3);
			other.decompressToDenseBlock(bt.getDenseBlock(), nRow / 2, nRow - 5, 3, 3);
			compare(ot, bt);
		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("should never be called")) {
				e.printStackTrace();
				fail("Failed to decompress " + e);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to decompress " + e);
		}
	}

	@Test
	public void decompressToDenseBlockBiggerSubPartOffsetCol() {
		try {
			MatrixBlock ot = denseMB(maxCol + 3);
			MatrixBlock bt = denseMB(maxCol + 3);
			base.decompressToDenseBlock(ot.getDenseBlock(), nRow / 2, nRow - 5, 0, 3);
			other.decompressToDenseBlock(bt.getDenseBlock(), nRow / 2, nRow - 5, 0, 3);
			compare(ot, bt);
		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("should never be called")) {
				e.printStackTrace();
				fail("Failed to decompress " + e);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to decompress " + e);
		}
	}

	@Test
	public void decompressToDenseBlockBiggerSubPartOffsetRow() {
		try {
			MatrixBlock ot = denseMB(maxCol + 3);
			MatrixBlock bt = denseMB(maxCol + 3);
			base.decompressToDenseBlock(ot.getDenseBlock(), nRow / 2, nRow - 5, 3, 0);
			other.decompressToDenseBlock(bt.getDenseBlock(), nRow / 2, nRow - 5, 3, 0);
			compare(ot, bt);
		}
		catch(DMLCompressionException e) {
			if(!e.getMessage().contains("should never be called")) {
				e.printStackTrace();
				fail("Failed to decompress " + e);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to decompress " + e);
		}
	}

	@Test
	public void testSerialization() {
		try {
			final AColGroup baseR = serializeAndBack(base);
			final AColGroup otherB = serializeAndBack(other);
			// Not really a good test but it verify the basics.
			assertEquals(baseR.getMin(), otherB.getMin(), 0.0);
			assertEquals(baseR.getMin(), base.getMin(), 0.0);
			assertEquals(baseR.getExactSizeOnDisk(), base.getExactSizeOnDisk(), 0.0);
			assertEquals(otherB.getExactSizeOnDisk(), other.getExactSizeOnDisk(), 0.0);
			assertEquals(baseR.getCompType(), base.getCompType());
			assertEquals(otherB.getCompType(), other.getCompType());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testNonZeros() {
		final long bNnz = base.getNumberNonZeros(nRow);
		final long oNnz = other.getNumberNonZeros(nRow);
		final long cells = ((long) nRow * maxCol);
		if(!(bNnz == cells || oNnz == cells))
			// if neither is reporting fully dense ... then they should match to exact correct
			if(bNnz != oNnz)
				fail("Fails number of non zero count: " + bNnz + " " + oNnz + "\n\n" + base + " " + other);

	}

	@Test
	public void sliceColumns01() {
		sliceColumns(1, maxCol);
	}

	@Test
	public void sliceColumns02() {
		sliceColumns(0, maxCol - 1);
	}

	@Test
	public void sliceColumns03() {
		sliceColumns(maxCol, maxCol + 3);
	}

	@Test
	public void sliceColumns04() {
		sliceColumns(3, maxCol + 3);
	}

	@Test
	public void sliceColumns05() {
		sliceColumns(4, 6);
	}

	public void sliceColumns(int low, int high) {
		try {
			final AColGroup bs = base.sliceColumns(low, high);
			final AColGroup os = other.sliceColumns(low, high);
			if(bs == os) // if null return
				return;
			compareDecompressSubPartOffsetSparse(bs, os);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sliceColumnsAfterRightMatrixMultiplication() {
		MatrixBlock right = TestUtils.generateTestMatrixBlock(maxCol, 10, 0, 10, 1.0, 2414);
		right = TestUtils.ceil(right);
		try {

			AColGroup bs = base.rightMultByMatrix(right);
			AColGroup os = other.rightMultByMatrix(right);
			if(bs == null || os == null) // if null return
				if(bs != os) {
					fail("both results are not equally null");
					return;
				}
				else
					return;
			bs = bs.sliceColumns(4, 6);
			os = os.sliceColumns(4, 6);
			if(bs == null || os == null) // if null return
				if(bs != os)
					fail("both results are not equally null");
				else
					return;
			MatrixBlock ot = denseMB(10);
			MatrixBlock bt = denseMB(10);
			decompressToDenseBlock(ot, bt, os, bs, 0, nRow);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sliceColumn() {
		final AColGroup bs = base.sliceColumn(maxCol);
		final AColGroup os = other.sliceColumn(maxCol);
		if(bs == os) // if null return
			return;
		compareDecompressSubPartOffsetSparse(bs, os);
	}

	@Test
	public void sliceColumnContained() {
		try {
			final AColGroup bs = base.sliceColumn(base.getColIndices().get(0));
			final AColGroup os = other.sliceColumn(other.getColIndices().get(0));
			if(bs == os) // if null return
				return;
			compareDecompressSubPartOffsetSparse(bs, os);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sliceColumnNotExistingColumn() {
		assertTrue(base.sliceColumn(maxCol + 1) == null);
		assertTrue(other.sliceColumn(maxCol + 1) == null);
	}

	@Test
	public void get() {
		Random r = new Random(134);
		for(int i = 0; i < 1000; i++) {
			final int row = r.nextInt(nRow);
			final int col = r.nextInt(maxCol);
			if(Math.abs(base.get(row, col) - other.get(row, col)) > 0.000000001)
				fail("Not equivalent values at (" + row + "," + col + ") " + base.get(row, col) + " " + other.get(row, col)
					+ "\n" + base.getClass().getSimpleName() + " " + other.getClass().getSimpleName());
		}
	}

	@Test
	public void getNumValues() {
		CompressionType bt = base.getCompType();
		CompressionType ot = other.getCompType();
		if(bt == CompressionType.UNCOMPRESSED || ot == CompressionType.UNCOMPRESSED)
			return;
		int bnv = base.getNumValues();
		int onv = other.getNumValues();

		if(bt == CompressionType.SDC || bt == CompressionType.SDCFOR ||
			(bt == CompressionType.RLE && ((ColGroupRLE) base).containZerosTuples()))
			bnv++;
		if(ot == CompressionType.SDC || ot == CompressionType.SDCFOR ||
			(ot == CompressionType.RLE && ((ColGroupRLE) other).containZerosTuples()))
			onv++;
		if(base instanceof ColGroupSDCSingleZeros)
			return;
		if(other instanceof ColGroupSDCSingleZeros)
			return;

		if(bnv != onv)
			fail("Not equivalent number of values in:  " + bnv + " vs " + onv + "\n" + base + " " + other);

	}

	@Test
	public void colSum() {
		double[] res1 = new double[maxCol];
		double[] res2 = new double[maxCol];
		AColGroup.colSum(Collections.singleton(base), res1, nRow);
		AColGroup.colSum(Collections.singleton(other), res2, nRow);
		TestUtils.compareMatrices(res1, res2, 0.0001);
	}

	@Test
	public void UA_SUM_KAHN() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAKP.toString(), 1), 1);
	}

	@Test
	public void UA_SUM() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAP.toString(), 1), 1);
	}

	@Test
	public void UA_SUM_REP() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAP.toString(), 1), 2);
	}

	@Test
	public void UA_MAX() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString(), 1), 1);
	}

	@Test
	public void UA_MAX_REP() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString(), 1), 2);
	}

	@Test
	public void UA_MIN() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMIN.toString(), 1), 1);
	}

	@Test
	public void UA_MIN_REP() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMIN.toString(), 1), 2);
	}

	@Test
	public void UA_PRODUCT() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAM.toString(), 1), 1);
	}

	@Test
	public void UA_PRODUCT_REP() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAM.toString(), 1), 2);
	}

	@Test
	public void UA_SUMSQ_KAHN() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UASQKP.toString(), 1), 1);
	}

	@Test(expected = DMLRuntimeException.class)
	public void UA_INDEX() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARIMAX.toString(), 1), 1);
	}

	@Test(expected = DMLRuntimeException.class)
	public void UA_VAR() {
		UA_FULL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACVAR.toString(), 1), 1);
	}

	protected void UA_FULL(AggregateUnaryOperator op, int reps) {
		try {

			double[] res1 = new double[1];
			double[] res2 = new double[1];
			if(op.aggOp.increOp.fn instanceof Multiply) {
				Arrays.fill(res1, 1);
				Arrays.fill(res2, 1);
			}
			for(int i = 0; i < reps; i++) {
				base.unaryAggregateOperations(op, res1, nRow, 0, nRow);
				other.unaryAggregateOperations(op, res2, nRow, 0, nRow);
			}
			if(op.aggOp.increOp.fn instanceof Multiply)
				TestUtils.compareMatricesPercentageDistance(res1, res2, 0.90, 0.90, "comp", false);
			else
				TestUtils.compareMatrices(res1, res2, 0.0001);
		}
		catch(DMLRuntimeException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void UA_SUM_KAHN_COL() {
		UA_COL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACKP.toString(), 1));
	}

	@Test
	public void UA_SUM_COL() {
		UA_COL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACP.toString(), 1));
	}

	@Test
	public void UA_MAX_COL() {
		UA_COL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACMAX.toString(), 1));
	}

	@Test
	public void UA_MIN_COL() {
		UA_COL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACMIN.toString(), 1));
	}

	@Test
	public void UA_PRODUCT_COL() {
		UA_COL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACM.toString(), 1));
	}

	@Test
	public void UA_SUMSQ_KAHN_COL() {
		UA_COL(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACSQKP.toString(), 1));
	}

	protected void UA_COL(AggregateUnaryOperator op) {
		try {
			double[] res1 = new double[maxCol];
			double[] res2 = new double[maxCol];
			if(op.aggOp.increOp.fn instanceof Multiply) {
				Arrays.fill(res1, 1);
				Arrays.fill(res2, 1);
			}
			base.unaryAggregateOperations(op, res1, nRow, 0, nRow);
			other.unaryAggregateOperations(op, res2, nRow, 0, nRow);

			if(op.aggOp.increOp.fn instanceof Multiply)
				TestUtils.compareMatricesPercentageDistance(res1, res2, 0.90, 0.90, "comp", false);
			else
				TestUtils.compareMatrices(res1, res2, 0.0001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void UA_SUM_KAHN_ROW() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 1));
	}

	@Test
	public void UA_SUM_ROW() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARP.toString(), 1));
	}

	@Test
	public void UA_MAX_ROW() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMAX.toString(), 1));
	}

	@Test
	public void UA_MIN_ROW() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMIN.toString(), 1));
	}

	@Test
	public void UA_PRODUCT_ROW() {
		try {
			UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARM.toString(), 1));
		}
		catch(AssertionError e) {
			LOG.error(base);
			LOG.error(other);
			throw e;
		}
	}

	@Test
	public void UA_SUMSQ_KAHN_ROW() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARSQKP.toString(), 1));
	}

	protected void UA_ROW(AggregateUnaryOperator op) {
		UA_ROW(op, 0, nRow);
	}

	@Test
	public void UA_SUM_KAHN_ROW_END() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 1), nRow - 4, nRow - 1);
	}

	@Test
	public void UA_SUM_ROW_END() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARP.toString(), 1), nRow - 4, nRow - 1);
	}

	@Test
	public void UA_MAX_ROW_END() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMAX.toString(), 1), nRow - 4, nRow - 1);
	}

	@Test
	public void UA_MIN_ROW_END() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMIN.toString(), 1), nRow - 4, nRow - 1);
	}

	@Test
	public void UA_PRODUCT_ROW_END() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARM.toString(), 1), nRow - 4, nRow - 1);
	}

	@Test
	public void UA_SUMSQ_KAHN_ROW_END() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARSQKP.toString(), 1), nRow - 4, nRow - 1);
	}

	@Test
	public void UA_SUM_KAHN_ROW_START() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 1), 1, 10);
	}

	@Test
	public void UA_SUM_ROW_START() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARP.toString(), 1), 1, 10);
	}

	@Test
	public void UA_MAX_ROW_START() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMAX.toString(), 1), 1, 10);
	}

	@Test
	public void UA_MIN_ROW_START() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMIN.toString(), 1), 1, 10);
	}

	@Test
	public void UA_PRODUCT_ROW_START() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARM.toString(), 1), 1, 10);
	}

	@Test
	public void UA_SUMSQ_KAHN_ROW_START() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARSQKP.toString(), 1), 1, 10);
	}

	@Test
	public void UA_SUM_KAHN_ROW_BEGINNING() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 1), 0, 4);
	}

	@Test
	public void UA_SUM_ROW_BEGINNING() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARP.toString(), 1), 0, 4);
	}

	@Test
	public void UA_MAX_ROW_BEGINNING() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMAX.toString(), 1), 0, 4);
	}

	@Test
	public void UA_MIN_ROW_BEGINNING() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMIN.toString(), 1), 0, 4);
	}

	@Test
	public void UA_PRODUCT_ROW_BEGINNING() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARM.toString(), 1), 0, 4);
	}

	@Test
	public void UA_SUMSQ_KAHN_ROW_BEGINNING() {
		UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARSQKP.toString(), 1), 0, 4);
	}

	protected void UA_ROW(AggregateUnaryOperator op, int rl, int ru) {
		UA_ROW(op, rl, ru, base, other, nRow);
	}

	protected static void UA_ROW(AggregateUnaryOperator op, int rl, int ru, AColGroup a, AColGroup b, int nRow) {
		try {
			double[] res1 = new double[ru];
			double[] res2 = new double[ru];
			if(op.aggOp.increOp.fn instanceof Multiply) {
				Arrays.fill(res1, 1);
				Arrays.fill(res2, 1);
			}

			a.unaryAggregateOperations(op, res1, nRow, rl, ru);
			b.unaryAggregateOperations(op, res2, nRow, rl, ru);
			TestUtils.compareMatrices(res1, res2, 0.0001);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testNoExceptionToString() {
		base.toString();
		other.toString();
	}

	@Test
	public void testNoExceptionEstimateSize() {
		base.estimateInMemorySize();
		other.estimateInMemorySize();
	}

	@Test
	public void sub100() {
		scalarOp(new RightScalarOperator(Plus.getPlusFnObject(), -100));
	}

	@Test
	public void sub2() {
		scalarOp(new RightScalarOperator(Minus.getMinusFnObject(), 2));
	}

	@Test
	public void sub0() {
		// should not do anything but good test to return same object.
		scalarOp(new RightScalarOperator(Minus.getMinusFnObject(), 0));
	}

	@Test
	public void mul2() {
		scalarOp(new RightScalarOperator(Multiply.getMultiplyFnObject(), 2));
	}

	@Test
	public void div2() {
		scalarOp(new RightScalarOperator(Divide.getDivideFnObject(), 2));
	}

	@Test
	public void pow2() {
		scalarOp(new RightScalarOperator(Power.getPowerFnObject(), 2));
	}

	@Test
	public void modulusScalar() {
		scalarOp(new RightScalarOperator(Modulus.getFnObject(), 2));
	}

	protected void scalarOp(ScalarOperator sop) {
		try {
			AColGroup br = base.scalarOperation(sop);
			AColGroup or = other.scalarOperation(sop);
			compare(br, or);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void round() {
		unaryOp(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ROUND)));
	}

	@Test
	public void mod() {
		unaryOp(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ABS)));
	}

	@Test
	public void floor() {
		unaryOp(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.FLOOR)));
	}

	@Test
	public void sin() {
		unaryOp(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.SIN)));
	}

	@Test
	public void cos() {
		unaryOp(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.COS)));
	}

	protected void unaryOp(UnaryOperator uop) {
		try {
			AColGroup br = base.unaryOperation(uop);
			AColGroup or = other.unaryOperation(uop);
			compare(br, or);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void replace100() {
		replaceOp(100, 0);
	}

	@Test
	public void replace0() {
		replaceOp(0, 100);
	}

	protected void replaceOp(double v, double t) {
		try {
			compare(base.replace(v, t), other.replace(v, t));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void multZeroVectorRight() {
		double[] v = new double[maxCol];
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject());
		binaryRowOp(op, v, false);
	}

	@Test
	public void multRandVectorRight() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject());
		binaryRowOp(op, v, false);
	}

	@Test
	public void divRandRight() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Divide.getDivideFnObject());
		binaryRowOp(op, v, false);
	}

	@Test
	public void addRandRight() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
		binaryRowOp(op, v, false);
	}

	@Test
	public void minusRandRight() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		binaryRowOp(op, v, false);
	}

	@Test
	public void modulusRandRight() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextInt(100);
		BinaryOperator op = new BinaryOperator(Modulus.getFnObject());
		binaryRowOp(op, v, false);
	}

	@Test
	public void multZeroVectorLeft() {
		double[] v = new double[maxCol];
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject());
		binaryRowOp(op, v, true);
	}

	@Test
	public void multRandVectorLeft() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject());
		binaryRowOp(op, v, true);
	}

	@Test
	public void divRandLeft() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Divide.getDivideFnObject());
		binaryRowOp(op, v, true);
	}

	@Test
	public void addRandLeft() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject());
		binaryRowOp(op, v, true);
	}

	@Test
	public void minusRandLeft() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextDouble();
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject());
		binaryRowOp(op, v, true);
	}

	@Test
	public void modulusRandLeft() {
		double[] v = new double[maxCol];
		Random r = new Random(134);
		for(int i = 0; i < v.length; i++)
			v[i] = r.nextInt(100);
		BinaryOperator op = new BinaryOperator(Modulus.getFnObject());
		binaryRowOp(op, v, true);
	}

	protected void binaryRowOp(BinaryOperator op, double[] v, boolean left) {
		boolean isRowSafe = op.isRowSafeLeft(v);
		try {
			if(left)
				compare(base.binaryRowOpLeft(op, v, isRowSafe), other.binaryRowOpLeft(op, v, isRowSafe));
			else
				compare(base.binaryRowOpRight(op, v, isRowSafe), other.binaryRowOpRight(op, v, isRowSafe));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	protected void compare(AColGroup a, AColGroup b) {
		final MatrixBlock ot = denseMB(maxCol);
		final MatrixBlock bt = denseMB(maxCol);
		base.decompressToDenseBlock(ot.getDenseBlock(), 0, nRow);
		other.decompressToDenseBlock(bt.getDenseBlock(), 0, nRow);
		compare(ot, bt);
	}

	@Test
	public void getMin() {
		if(Math.abs(base.getMin() - other.getMin()) > 0.000000001)
			fail("Min not Equivalent");
	}

	@Test
	public void getMax() {
		if(Math.abs(base.getMax() - other.getMax()) > 0.000000001)
			fail("Min not Equivalent");
	}

	@Test
	public void tsmm() {
		try {

			final MatrixBlock bt = new MatrixBlock(maxCol, maxCol, false);
			final MatrixBlock ot = new MatrixBlock(maxCol, maxCol, false);
			ot.allocateDenseBlock();
			bt.allocateDenseBlock();
			base.tsmm(bt, nRow);
			other.tsmm(ot, nRow);
			compare(ot, bt);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void isEmpty() {
		assertTrue(base.isEmpty() == other.isEmpty());
	}

	@Test
	public void leftMultNoPreAggDense() {
		leftMultNoPreAgg(3, 0, 1, 0, nRow);
	}

	@Test
	public void leftMultNoPreAggDenseMultiRow() {
		leftMultNoPreAgg(3, 0, 3, 0, nRow);
	}

	@Test
	public void leftMultNoPreAggSparse() {
		leftMultNoPreAgg(3, 0, 1, 0, nRow, 0.1);
	}

	@Test
	public void leftMultNoPreAggSparseMultiRow() {
		leftMultNoPreAgg(3, 0, 3, 0, nRow, 0.1);
	}

	@Test
	public void leftMultNoPreAggDenseColRange() {
		leftMultNoPreAgg(3, 0, 1, 5, nRow - 4);
	}

	@Test
	public void leftMultNoPreAggDenseMultiRowColRange() {
		leftMultNoPreAgg(3, 0, 3, 5, nRow - 4);
	}

	@Test
	public void leftMultNoPreAggSparseColRange() {
		leftMultNoPreAgg(3, 0, 1, 5, nRow - 4, 0.1);
	}

	@Test
	public void leftMultNoPreAggSparseMultiRowColRange() {
		leftMultNoPreAgg(3, 0, 3, 5, nRow - 4, 0.1);
	}

	@Test
	public void leftMultNoPreAggDenseColStartRange() {
		leftMultNoPreAgg(3, 0, 1, 5, 9);
	}

	@Test
	public void leftMultNoPreAggDenseMultiRowColStartRange() {
		leftMultNoPreAgg(3, 0, 3, 5, 9);
	}

	@Test
	public void leftMultNoPreAggSparseColStartRange() {
		leftMultNoPreAgg(3, 0, 1, 5, 9, 0.1);
	}

	@Test
	public void leftMultNoPreAggSparseMultiRowColStartRange() {
		leftMultNoPreAgg(3, 0, 3, 5, 9, 0.1);
	}

	@Test
	public void leftMultNoPreAggDenseColEndRange() {
		leftMultNoPreAgg(3, 0, 1, nRow - 10, nRow - 3);
	}

	@Test
	public void leftMultNoPreAggDenseMultiRowColEndRange() {
		leftMultNoPreAgg(3, 0, 3, nRow - 10, nRow - 3);
	}

	@Test
	public void leftMultNoPreAggSparseColEndRange() {
		leftMultNoPreAgg(3, 0, 1, nRow - 10, nRow - 3, 0.1);
	}

	@Test
	public void leftMultNoPreAggSparseMultiRowColEndRange() {
		leftMultNoPreAgg(3, 0, 3, nRow - 10, nRow - 3, 0.1);
	}

	@Test
	public void leftMultNoPreAggSparseMultiRowColToEnd() {
		leftMultNoPreAgg(3, 0, 3, nRow - 10, nRow, 0.1);
	}

	@Test
	public void leftMultNoPreAggSparseMultiRowColFromStart() {
		leftMultNoPreAgg(3, 0, 3, 0, 4, 0.1);
	}

	public void leftMultNoPreAgg(int nRowLeft, int rl, int ru, int cl, int cu) {
		leftMultNoPreAgg(nRowLeft, rl, ru, cl, cu, 1.0);
	}

	public void leftMultNoPreAgg(int nRowLeft, int rl, int ru, int cl, int cu, double sparsity) {

		final MatrixBlock left = TestUtils
			.round(TestUtils.generateTestMatrixBlock(nRowLeft, nRow, -10, 10, sparsity, 1342));

		leftMultNoPreAgg(nRowLeft, rl, ru, cl, cu, left);
	}

	@Test(expected = NotImplementedException.class)
	public void leftMultNoPreAggWithEmptyRows() {

		MatrixBlock left = TestUtils.round(TestUtils.generateTestMatrixBlock(3, nRow, -10, 10, 0.2, 222));

		left = left.append(new MatrixBlock(3, nRow, true), null, false);
		left.denseToSparse(true);
		left.recomputeNonZeros();
		leftMultNoPreAgg(6, 2, 5, 0, nRow, left);
		throw new NotImplementedException("Make test parse since the check actually says it is correct");
	}

	public void leftMultNoPreAgg(int nRowLeft, int rl, int ru, int cl, int cu, MatrixBlock left) {
		try {

			final MatrixBlock bt = new MatrixBlock(nRowLeft, maxCol, false);
			bt.allocateDenseBlock();

			final MatrixBlock ot = new MatrixBlock(nRowLeft, maxCol, false);
			ot.allocateDenseBlock();

			base.leftMultByMatrixNoPreAgg(left, bt, rl, ru, cl, cu);
			other.leftMultByMatrixNoPreAgg(left, ot, rl, ru, cl, cu);
			bt.recomputeNonZeros();
			ot.recomputeNonZeros();
			compare(bt, ot);
		}
		catch(NotImplementedException e) {
			LOG.error("not implemented: " + base.getClass().getSimpleName() + " or: " + other.getClass().getSimpleName());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void sparseSelection() {
		MatrixBlock mb = CLALibSelectionMultTest.createSelectionMatrix(nRow, 5, false);
		mb = CompressedMatrixBlock.getUncompressed(mb);
		MatrixBlock ret = new MatrixBlock(5, maxCol, true);
		ret.allocateSparseRowsBlock();
		ret.setNonZeros(-1);
		selection(mb, ret);
	}

	@Test
	public void denseSelection() {
		MatrixBlock mb = CLALibSelectionMultTest.createSelectionMatrix(nRow, 5, false);
		mb = CompressedMatrixBlock.getUncompressed(mb);
		MatrixBlock ret = new MatrixBlock(5, maxCol, false);
		ret.allocateDenseBlock();
		ret.setNonZeros(-1);
		assertFalse(ret.isInSparseFormat());
		selection(mb, ret);
	}

	@Test
	public void sparseSelectionEmptyRows() {
		MatrixBlock mb = CLALibSelectionMultTest.createSelectionMatrix(nRow, 50, true);
		mb = CompressedMatrixBlock.getUncompressed(mb);
		MatrixBlock ret = new MatrixBlock(50, maxCol, true);
		ret.allocateSparseRowsBlock();
		ret.setNonZeros(-1);
		selection(mb, ret);
	}

	@Test
	public void denseSelectionEmptyRows() {
		MatrixBlock mb = CLALibSelectionMultTest.createSelectionMatrix(nRow, 50, true);
		mb = CompressedMatrixBlock.getUncompressed(mb);
		MatrixBlock ret = new MatrixBlock(50, maxCol, false);
		ret.allocateDenseBlock();
		ret.setNonZeros(-1);
		assertFalse(ret.isInSparseFormat());
		selection(mb, ret);
	}

	public void selection(MatrixBlock selection, MatrixBlock ret) {
		P[] points = ColGroupUtils.getSortedSelection(selection.getSparseBlock(), 0, selection.getNumRows());
		MatrixBlock ret1 = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), ret.isInSparseFormat());
		ret1.allocateBlock();

		MatrixBlock ret2 = new MatrixBlock(ret.getNumRows(), ret.getNumColumns(), ret.isInSparseFormat());
		ret2.allocateBlock();

		try {

			base.selectionMultiply(selection, points, ret1, 0, selection.getNumRows());
			other.selectionMultiply(selection, points, ret2, 0, selection.getNumRows());

			TestUtils.compareMatricesBitAvgDistance(ret1, ret2, 0, 0,
				base.getClass().getSimpleName() + " vs " + other.getClass().getSimpleName());

		}
		catch(NotImplementedException e) {
			// okay
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void preAggLeftMult() {
		preAggLeftMult(new MatrixBlock(1, nRow, 1.0), 0, 1);
	}

	@Test
	public void preAggLeftMulRand() {
		preAggLeftMult(TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, nRow, -10, 10, 1.0, 32)), 0, 1);
	}

	@Test
	public void preAggLeftMultTwoRows() {
		preAggLeftMult(new MatrixBlock(2, nRow, 1.0), 0, 2);
	}

	@Test
	public void preAggLeftMultTwoRowsRand() {
		preAggLeftMult(TestUtils.ceil(TestUtils.generateTestMatrixBlock(2, nRow, -10, 10, 1.0, 324)), 0, 2);
	}

	@Test
	public void preAggLeftMultSecondRow() {
		preAggLeftMult(new MatrixBlock(2, nRow, 1.0), 1, 2);
	}

	@Test
	public void preAggLeftMultSecondRowRand() {
		preAggLeftMult(TestUtils.ceil(TestUtils.generateTestMatrixBlock(2, nRow, -10, 10, 1.0, 241)), 0, 2);
	}

	@Test
	public void preAggLeftMultSparse() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, nRow, -1, 10, 0.2, 1342));
		mb.denseToSparse(true);
		preAggLeftMult(mb, 0, 1);
	}

	@Test
	public void preAggLeftMultSparseTwoRows() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(2, nRow, -1, 10, 0.2, 1342));
		mb.denseToSparse(true);
		preAggLeftMult(mb, 0, 2);
	}

	@Test
	public void preAggLeftMultSparseFiveRows() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(5, nRow, -1, 10, 0.2, 1342));
		mb.sparseToDense();
		mb.denseToSparse(true);
		preAggLeftMult(mb, 0, 5);
	}

	@Test
	public void preAggLeftMultSparseFiveRowsMCSR() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(5, nRow, -1, 10, 0.2, 1342));
		mb.sparseToDense();
		mb.denseToSparse(false);
		preAggLeftMult(mb, 0, 5);
	}

	@Test
	public void preAggLeftMultSparseSecondRow() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(2, nRow, -1, 10, 0.2, 1342));
		mb.denseToSparse(true);
		preAggLeftMult(mb, 1, 2);
	}

	@Test
	public void preAggLeftMultSparseEmptyRow() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(2, nRow, -1, 10, 0.2, 1342));
		mb = mb.append(new MatrixBlock(2, nRow, false), null, false);
		mb.denseToSparse(true);
		mb.recomputeNonZeros();
		preAggLeftMult(mb, 3, 4);
	}

	@Test
	public void preAggLeftMultSparseSomeEmptyRows() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(3, nRow, -1, 10, 0.2, 1342));
		mb = mb.append(new MatrixBlock(3, nRow, false), null, false);
		mb.denseToSparse(true);
		mb.recomputeNonZeros();
		preAggLeftMult(mb, 2, 4);
	}

	public void preAggLeftMult(MatrixBlock mb, int rl, int ru) {
		try {

			MatrixBlock retB = null;
			MatrixBlock retO = null;

			final double[] rowSum = mb.rowSum().getDenseBlockValues();

			if(base instanceof AMorphingMMColGroup) {
				double[] cb = new double[maxCol];
				AColGroup b = ((AMorphingMMColGroup) base).extractCommon(cb);
				retB = mmPreAgg((APreAgg) b, mb, cb, rowSum, rl, ru);
			}
			else if(base instanceof APreAgg)
				retB = mmPreAgg((APreAgg) base, mb, null, rowSum, rl, ru);

			if(other instanceof AMorphingMMColGroup) {
				double[] cb = new double[maxCol];
				AColGroup b = ((AMorphingMMColGroup) other).extractCommon(cb);
				retO = mmPreAgg((APreAgg) b, mb, cb, rowSum, rl, ru);
			}
			else if(other instanceof APreAgg)
				retO = mmPreAgg((APreAgg) other, mb, null, rowSum, rl, ru);

			if(retB == null) {
				retB = new MatrixBlock(ru, maxCol, false);
				retB.allocateDenseBlock();
				base.leftMultByMatrixNoPreAgg(mb, retB, rl, ru, 0, nRow);
			}

			if(retO == null) {
				retO = new MatrixBlock(ru, maxCol, false);
				retO.allocateDenseBlock();
				other.leftMultByMatrixNoPreAgg(mb, retO, rl, ru, 0, nRow);
			}

			compare(retB, retO);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	private MatrixBlock mmPreAgg(APreAgg g, MatrixBlock mb, double[] cv, double[] rowSum, int rl, int ru) {
		final MatrixBlock retB = new MatrixBlock(ru, maxCol, false);
		retB.allocateDenseBlock();

		double[] preB = new double[g.getPreAggregateSize() * (ru - rl)];

		g.preAggregate(mb, preB, rl, ru);

		MatrixBlock preAggB = new MatrixBlock(ru - rl, g.getPreAggregateSize(), preB);
		MatrixBlock tmpRes = new MatrixBlock(1, retB.getNumColumns(), false);

		g.mmWithDictionary(preAggB, tmpRes, retB, 1, rl, ru);

		if(rowSum != null) {
			if(cv != null) {
				if(retB.isEmpty())
					retB.allocateDenseBlock();
				else
					retB.sparseToDense();

				outerProduct(rowSum, cv, retB.getDenseBlockValues(), rl, ru);
			}
		}
		return retB;
	}

	@Test
	public void preAggLeftMultDenseSub() {
		preAggLeftMultDense(new MatrixBlock(1, nRow, 2.0), 0, 1, 3, nRow - 3);
	}

	@Test
	public void preAggLeftMultTwoRowsDenseStart() {
		preAggLeftMultDense(new MatrixBlock(2, nRow, 2.0), 0, 2, 3, 5);
	}

	@Test
	public void preAggLeftMultSecondRowDenseEnd() {
		preAggLeftMultDense(new MatrixBlock(2, nRow, 2.0), 1, 2, nRow - 10, nRow - 3);
	}

	@Test
	public void preAggLeftMultDenseNonContiguous() {
		try {

			MatrixBlock mb = new MatrixBlock(1, nRow, 2.0);
			double[] vals = mb.getDenseBlockValues();
			DenseBlockFP64Mock mock = new DenseBlockFP64Mock(new int[] {1, nRow}, vals);
			preAggLeftMultDense(new MatrixBlock(1, nRow, mock), 0, 1, 3, nRow - 3);
		}
		catch(NotImplementedException e) {
			// valid
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static class DenseBlockFP64Mock extends DenseBlockFP64 {
		private static final long serialVersionUID = -3601232958390554672L;

		public DenseBlockFP64Mock(int[] dims, double[] data) {
			super(dims, data);
		}

		@Override
		public boolean isContiguous() {
			return false;
		}

		@Override
		public int numBlocks() {
			return 2;
		}
	}

	public void preAggLeftMultDense(MatrixBlock mb, int rl, int ru, int cl, int cu) {
		final double[] rowSum = CLALibLeftMultBy.rowSum(mb, rl, ru, cl, cu);

		final MatrixBlock retB = morphingLLM(base, mb, rl, ru, cl, cu, rowSum);
		final MatrixBlock retO = morphingLLM(other, mb, rl, ru, cl, cu, rowSum);

		retB.recomputeNonZeros();
		retO.recomputeNonZeros();

		compare(retB, retO);

	}

	private MatrixBlock lmmNoAgg(AColGroup g, MatrixBlock mb, int rl, int ru, int cl, int cu) {
		MatrixBlock tmpB = new MatrixBlock(ru, maxCol, false);
		tmpB.allocateDenseBlock();
		g.leftMultByMatrixNoPreAgg(mb, tmpB, rl, ru, cl, cu);
		return tmpB;
	}

	private MatrixBlock morphingLLM(AColGroup g, MatrixBlock mb, int rl, int ru, int cl, int cu, final double[] rowSum) {
		final MatrixBlock retB;
		if(g instanceof AMorphingMMColGroup) {
			double[] cb = new double[maxCol];
			AColGroup b = ((AMorphingMMColGroup) g).extractCommon(cb);
			retB = mmPreAggDense((APreAgg) b, mb, cb, rowSum, rl, ru, cl, cu);
		}
		else if(g instanceof APreAgg)
			retB = mmPreAggDense((APreAgg) g, mb, null, rowSum, rl, ru, cl, cu);
		else if(g instanceof ColGroupConst) {
			double[] cb = new double[maxCol];
			((ColGroupConst) g).addToCommon(cb);
			retB = mmRowSum(cb, rowSum, rl, ru, cl, cu);
		}
		else
			retB = lmmNoAgg(g, mb, rl, ru, cl, cu);

		return retB;
	}

	private MatrixBlock mmPreAggDense(APreAgg g, MatrixBlock mb, double[] cv, double[] rowSum, int rl, int ru, int cl,
		int cu) {
		try {
			final MatrixBlock retB = new MatrixBlock(ru, maxCol, false);
			retB.allocateDenseBlock();
			double[] preB = new double[g.getPreAggregateSize() * (ru - rl)];
			g.preAggregateDense(mb, preB, rl, ru, cl, cu);
			MatrixBlock preAggB = new MatrixBlock(ru - rl, g.getPreAggregateSize(), preB);
			MatrixBlock tmpRes = new MatrixBlock(1, retB.getNumColumns(), false);
			tmpRes.allocateBlock();
			g.mmWithDictionary(preAggB, tmpRes, retB, 1, rl, ru);
			mmRowSum(retB, cv, rowSum, rl, ru);
			return retB;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
			return null;
		}
	}

	private MatrixBlock mmRowSum(double[] cv, double[] rowSum, int rl, int ru, int cl, int cu) {
		final MatrixBlock retB = new MatrixBlock(ru, maxCol, false);
		retB.allocateDenseBlock();
		mmRowSum(retB, cv, rowSum, rl, ru);
		return retB;
	}

	private void mmRowSum(MatrixBlock retB, double[] cv, double[] rowSum, int rl, int ru) {
		if(rowSum != null) {
			if(cv != null) {
				if(retB.isEmpty())
					retB.allocateDenseBlock();
				else
					retB.sparseToDense();

				outerProduct(rowSum, cv, retB.getDenseBlockValues(), rl, ru);
			}
		}
	}

	private static void outerProduct(final double[] leftRowSum, final double[] rightColumnSum, final double[] result,
		int rl, int ru) {
		for(int row = rl; row < ru; row++) {
			final int offOut = rightColumnSum.length * row;
			final double vLeft = leftRowSum[row];
			for(int col = 0; col < rightColumnSum.length; col++)
				result[offOut + col] += vLeft * rightColumnSum[col];
		}
	}

	@Test
	public void contains0() {
		containsValue(0);
	}

	@Test
	public void contains2() {
		containsValue(2);
	}

	@Test
	public void contains100() {
		containsValue(100);
	}

	@Test
	public void containsNaN() {
		containsValue(Double.NaN);
	}

	@Test
	public void containsInf() {
		containsValue(Double.POSITIVE_INFINITY);
	}

	@Test
	public void containsInfNeg() {
		containsValue(Double.NEGATIVE_INFINITY);
	}

	public void containsValue(double v) {
		boolean baseContains = base.containsValue(v);
		boolean otherContains = other.containsValue(v);
		if(baseContains != otherContains) {
			fail("not containing value " + v + " -- " + baseContains + ":" + otherContains + "\n\n" + base + " " + other);
		}
	}

	@Test
	public void rexpandColsMax() {
		int m = (int) base.getMax();
		if(m > 0)
			rexpandCols(m, true, true);
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMaxNoIgnore() {
		int m = (int) base.getMax();
		if(m > 0)
			rexpandCols(m, false, true);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test
	public void rexpandColsMaxNoCast() {
		try {
			int m = (int) base.getMax();
			if(m > 0)
				rexpandCols(m, true, false);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMaxNoCast1() {
		rexpandCols(1, true, false);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMaxNoCast100() {
		rexpandCols(100, true, false);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMaxNoBoth() {
		int m = (int) base.getMax();
		if(m > 0)
			rexpandCols(m, false, false);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMax1NoBoth() {
		rexpandCols(1, false, false);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMax100NoBoth() {
		rexpandCols(100, false, false);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMax1() {
		rexpandCols(1, true, true);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMax5() {
		rexpandCols(5, true, true);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	@Test(expected = DMLRuntimeException.class)
	public void rexpandColsMax100() {
		rexpandCols(100, true, true);
		throw new DMLRuntimeException("To make test parse if it correctly evaluated before");
	}

	public void rexpandCols(int m, boolean ignore, boolean cast) {
		try {
			if(maxCol == 1) {
				AColGroup b = base.rexpandCols(m, ignore, cast, nRow);
				AColGroup o = other.rexpandCols(m, ignore, cast, nRow);
				if(!(b == null && o == null))
					compare(b, o);
			}
		}
		catch(DMLRuntimeException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void rightMultDenseVector() {
		rightMult(new MatrixBlock(maxCol, 1, 1.0));
	}

	@Test
	public void rightMultEmptyVector() {
		rightMult(new MatrixBlock(maxCol, 1, false));
	}

	@Test
	public void rightMultSparseVector() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(maxCol, 1, 1, 1, 0.01, 1342);
		mb.denseToSparse(true);
		rightMult(mb);
	}

	@Test
	public void rightMultDenseMatrix() {
		rightMult(new MatrixBlock(maxCol, 10, 1.0));
	}

	@Test
	public void rightMultDenseMatrixSomewhatSparse() {
		rightMultWithAllCols(TestUtils.generateTestMatrixBlock(maxCol, 10, 1, 1, 0.6, 1342));
	}

	@Test
	public void rightMultDenseMatrixSomewhatSparseManyColumns() {
		rightMultWithAllCols(TestUtils.generateTestMatrixBlock(maxCol, 201, 1, 1, 0.6, 1342));
	}

	@Test
	public void rightMultEmptyMatrix() {
		rightMult(new MatrixBlock(maxCol, 10, false));
	}

	@Test
	public void rightMultSparseMatrix() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(maxCol, 10, 1, 1, 0.01, 1342);
		mb.denseToSparse(true);
		rightMult(mb);
	}

	@Test
	public void rightMultMatrixNotContainingValuesInColumns() {
		MatrixBlock mb = new MatrixBlock(maxCol + 4, 10, false);
		mb.allocateDenseBlock();
		mb.set(maxCol + 1, 3, 2.0);
		mb.set(maxCol + 3, 6, 2.0);
		rightMult(mb);
	}

	@Test
	public void rightMultMatrixNotContainingValuesInColumnsSparse() {
		MatrixBlock mb = new MatrixBlock(maxCol + 4, 10, false);
		mb.allocateDenseBlock();
		mb.set(maxCol + 1, 3, 2.0);
		mb.set(maxCol + 3, 6, 2.0);
		mb.denseToSparse(true);
		rightMult(mb);
	}

	@Test
	public void rightMultMatrixSingleValue() {
		MatrixBlock mb = new MatrixBlock(maxCol, 10, false);
		mb.allocateDenseBlock();
		mb.set(maxCol - 1, 3, 2.0);
		rightMult(mb);
	}

	@Test
	public void rightMultMatrixDiagonalSparse() {
		MatrixBlock mb = new MatrixBlock(maxCol, 10, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < maxCol; i++) {
			mb.set(i, i % 10, i);
		}
		mb.denseToSparse(true);
		rightMult(mb);
	}

	@Test
	public void rightMultMatrixDiagonalSparseWithCols() {
		MatrixBlock mb = new MatrixBlock(maxCol, 10, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < maxCol; i++) {
			mb.set(i, i % 10, i);
		}
		mb.denseToSparse(true);
		rightMultWithAllCols(mb);
	}

	public void rightMultWithAllCols(MatrixBlock right) {
		try {
			final IColIndex cols = ColIndexFactory.create(right.getNumColumns());
			AColGroup b = base.rightMultByMatrix(right, cols, 1);
			AColGroup o = other.rightMultByMatrix(right, cols, 1);
			if(!(b == null && o == null))
				compare(b, o);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	public void rightMult(MatrixBlock right) {
		try {
			AColGroup b = base.rightMultByMatrix(right);
			AColGroup o = other.rightMultByMatrix(right);
			if(!(b == null && o == null))
				compare(b, o);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void lmmUncompressedColGroup() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRow, 3, 1, 1, 0.5, 134);
		lmmColGroup(ColGroupUncompressed.create(mb));
	}

	@Test
	public void lmmSDCSingleZeroColGroup() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1, nRow, 1, 1, 0.3, 134);
		lmmColGroup(getSDCGroup(mb));
	}

	@Test
	public void lmmSDCZeroSingleColGroup() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, nRow, 1, 3, 0.3, 134));
		lmmColGroup(getSDCGroup(mb));
	}

	@Test
	public void lmmSDCSingleZeroMultiColColGroup() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1, nRow, 1, 1, 0.3, 134);
		mb = mb.append(mb, false).append(mb, false).append(mb, false);
		lmmColGroup(getSDCGroup(mb));
	}

	@Test
	public void lmmSDCZeroMultiColColGroup() {
		MatrixBlock a = TestUtils.ceil(TestUtils.generateTestMatrixBlock(2, nRow / 2, 1, 2, 0.5, 134));
		MatrixBlock b = new MatrixBlock(2, nRow - a.getNumColumns(), false);
		MatrixBlock mb = a.append(b);

		mb = mb.append(mb, false).append(new MatrixBlock(maxCol * 2, nRow, false), false);
		AColGroup g = getSDCGroup(mb);
		lmmColGroup(g);
	}

	@Test
	public void lmmSDCZeroMultiColColGroupSmallCost() {
		MatrixBlock mb = new MatrixBlock(2, 2, new double[] {1, 2, 3, 4});
		mb = mb.append(new MatrixBlock(2, nRow - 2, false));
		AColGroup g = getSDCGroup(mb);
		lmmColGroup(g);
	}

	@Test
	public void lmmSDCZeroMultiColColGroupSmallCostInverted() {
		MatrixBlock mb = new MatrixBlock(2, 2, new double[] {1, 2, 3, 4});
		mb = new MatrixBlock(2, nRow - 2, false).append(mb);
		AColGroup g = getSDCGroup(mb);
		lmmColGroup(g);
	}

	@Test
	public void lmmSDCZeroMultiColColGroupSmallCostMiddle() {
		MatrixBlock mb = new MatrixBlock(2, 2, new double[] {1, 2, 3, 4});
		mb = new MatrixBlock(2, nRow / 2, false).append(mb).append(new MatrixBlock(2, nRow / 2 - 2, false));
		AColGroup g = getSDCGroup(mb);
		lmmColGroup(g);
	}

	@Test
	public void lmmDDCColGroupSingleCol() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, nRow, 1, 3, 0.3, 134));
		lmmColGroup(getDDCGroup(mb));
	}

	@Test
	public void lmmDDCColGroupMultiCol() {
		MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(5, nRow, 1, 3, 0.3, 134));
		lmmColGroup(getDDCGroup(mb));
	}

	private AColGroup getRLEGroup(MatrixBlock mbt) {
		return getColGroup(mbt, CompressionType.RLE);
	}

	private AColGroup getSDCGroup(MatrixBlock mbt) {
		return getColGroup(mbt, CompressionType.SDC);
	}

	private AColGroup getDDCGroup(MatrixBlock mbt) {
		return getColGroup(mbt, CompressionType.DDC);
	}

	private AColGroup getColGroup(MatrixBlock mbt, CompressionType ct) {
		return getColGroup(mbt, ct, nRow);
	}

	protected static AColGroup getColGroup(MatrixBlock mbt, CompressionType ct, int nRow) {
		try {

			final IColIndex cols = ColIndexFactory.create(mbt.getNumRows());
			final List<CompressedSizeInfoColGroup> es = new ArrayList<>();
			final EstimationFactors f = new EstimationFactors(nRow, nRow, mbt.getSparsity());
			es.add(new CompressedSizeInfoColGroup(cols, f, 321452, ct));
			final CompressedSizeInfo csi = new CompressedSizeInfo(es);

			final CompressionSettings cs = new CompressionSettingsBuilder().create();
			cs.transposed = true;

			final List<AColGroup> comp = ColGroupFactory.compressColGroups(mbt, csi, cs);

			return comp.get(0);
		}
		catch(Exception e) {
			fail("Failed construction compression : " + ct + "\n" + mbt);
			return null;
		}
	}

	@Test(expected = DMLCompressionException.class)
	public void lmmSelfBase() {
		lmmColGroup(base);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	@Test(expected = DMLCompressionException.class)
	public void lmmSelfOther() {
		lmmColGroup(other);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	public void lmmColGroup(AColGroup g) {
		try {
			MatrixBlock outB = allocateLMMOut(g);
			MatrixBlock outO = allocateLMMOut(g);

			base.leftMultByAColGroup(g, outB, nRow);
			other.leftMultByAColGroup(g, outO, nRow);
			compare(outB, outO);
		}
		catch(DMLCompressionException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	protected MatrixBlock allocateLMMOut(AColGroup g) {
		final IColIndex gci = g.getColIndices();
		final int maxLeft = gci.get(gci.size() - 1) + 1;

		MatrixBlock ret = new MatrixBlock(maxLeft, maxCol, false);
		ret.allocateDenseBlock();
		return ret;
	}

	@Test(expected = DMLCompressionException.class)
	public void tsmmSelfBase() {
		tsmmColGroup(base);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	@Test(expected = DMLCompressionException.class)
	public void tsmmSelfOther() {
		tsmmColGroup(other);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	@Test(expected = DMLCompressionException.class)
	public void tsmmEmpty() {
		tsmmColGroup(new ColGroupEmpty(ColIndexFactory.create(new int[] {1, 3, 10})));
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	@Test(expected = DMLCompressionException.class)
	public void tsmmSDCZeroMultiColColGroupSmallCost() {
		MatrixBlock mb = new MatrixBlock(2, 2, new double[] {1, 2, 3, 4});
		mb = mb.append(new MatrixBlock(2, nRow - 2, false));
		AColGroup g = getSDCGroup(mb);
		tsmmColGroup(g);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	@Test(expected = DMLCompressionException.class)
	public void tsmmSDCZeroMultiColColGroupSmallCostInverted() {
		MatrixBlock mb = new MatrixBlock(2, 2, new double[] {1, 2, 3, 4});
		mb = new MatrixBlock(2, nRow - 2, false).append(mb);
		AColGroup g = getSDCGroup(mb);
		tsmmColGroup(g);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");

	}

	@Test(expected = DMLCompressionException.class)
	public void tsmmSDCZeroMultiColColGroupSmallCostMiddle() {
		MatrixBlock a = new MatrixBlock(2, 2, new double[] {1, 2, 3, 4});
		MatrixBlock b = new MatrixBlock(2, nRow / 2, false);
		MatrixBlock c = new MatrixBlock(2, nRow - a.getNumColumns() - b.getNumColumns(), false);

		AColGroup g = getSDCGroup(a.append(b).append(c));
		tsmmColGroup(g);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	@Test(expected = DMLCompressionException.class)
	// @Test
	public void tsmmRLE() {
		MatrixBlock mb = new MatrixBlock(2, 4, new double[] {1, 2, 0, 2, 0, 2, 1, 2});
		mb = mb.append(new MatrixBlock(2, nRow - 4, false));
		AColGroup g = getRLEGroup(mb);
		tsmmColGroup(g);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	@Test(expected = DMLCompressionException.class)
	// @Test
	public void tsmmRLEFull() {
		MatrixBlock a = new MatrixBlock(2, nRow / 2, 2.0);
		MatrixBlock b = new MatrixBlock(2, nRow - a.getNumColumns(), 3.0);
		AColGroup g = getRLEGroup(a.append(b));
		tsmmColGroup(g);
		throw new DMLCompressionException("The output is verified correct just ignore not implemented");
	}

	public void tsmmColGroup(AColGroup g) {
		try {
			MatrixBlock outB = allocateTSMMOut(g);
			MatrixBlock outO = allocateTSMMOut(g);
			base.tsmmAColGroup(g, outB);
			other.tsmmAColGroup(g, outO);
			compare(outB, outO);
		}
		catch(DMLCompressionException e) {
			throw e;
		}
		catch(AssertionError e) {
			e.printStackTrace();
			fail(e.getMessage() + this);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage() + this);
		}

	}

	protected MatrixBlock allocateTSMMOut(AColGroup g) {
		final IColIndex gci = g.getColIndices();
		final int max = Math.max(gci.get(gci.size() - 1) + 1, maxCol);

		MatrixBlock ret = new MatrixBlock(max, max, false);
		ret.allocateDenseBlock();
		return ret;
	}

	@Test
	public void centralMoment() {
		if(maxCol == 1) {
			AggregateOperationTypes opType = CMOperator.getCMAggOpType(2);
			CMOperator cmb = new CMOperator(CM.getCMFnObject(opType), opType);
			CMOperator cmo = new CMOperator(CM.getCMFnObject(opType), opType);
			double b = base.centralMoment(cmb, nRow).getRequiredResult(opType);
			double o = other.centralMoment(cmo, nRow).getRequiredResult(opType);
			if(Math.abs(b - o) > 0.000000001)
				fail("centralMomentNotEquivalent base: " + b + " other: " + o + "\n\n" + base + " " + other);
		}
	}

	@Test
	public void getCost() {
		final ComputationCostEstimator cheap = new ComputationCostEstimator(1, 1, 1, 1, 1, 1, 1, 1, false);
		final ComputationCostEstimator expensive = new ComputationCostEstimator(100, 100, 100, 100, 100, 100, 100, 100,
			true);
		double cb = base.getCost(cheap, nRow);
		double eb = base.getCost(expensive, nRow);
		double co = other.getCost(cheap, nRow);
		double eo = other.getCost(expensive, nRow);

		assertTrue(cb < eb);
		assertTrue(co < eo);
	}

	@Test
	public void sliceRowsBeforeEnd() {
		if(nRow > 10)
			sliceRows(0, nRow - 1);
	}

	@Test
	public void sliceRowsFull() {
		if(nRow > 10)
			sliceRows(0, nRow);
	}

	@Test
	public void sliceRowsAfterStart() {
		if(nRow > 10)
			sliceRows(3, nRow);
	}

	@Test
	public void sliceRowsMiddle() {
		if(nRow > 10)
			sliceRows(5, nRow - 3);
	}

	@Test
	public void sliceRowsEnd() {
		if(nRow > 10)
			sliceRows(nRow - 7, nRow - 4);
	}

	@Test
	public void sliceRowsStart() {
		if(nRow > 10)
			sliceRows(2, 7);
	}

	public void sliceRows(int rl, int ru) {
		try {
			if(base instanceof ColGroupRLE || other instanceof ColGroupRLE) {
				expectNotImplementedSlice(rl, ru);
				return;
			}

			AColGroup a = base.sliceRows(rl, ru);
			AColGroup b = other.sliceRows(rl, ru);

			final int newNRow = ru - rl;

			if(a == null || b == null)
				// one side is concluded empty
				// We do not enforce that empty is returned if it is empty, since some column groups
				// are to expensive to analyze if empty.
				return;
			assertTrue(a.getColIndices() == base.getColIndices());
			assertTrue(b.getColIndices() == other.getColIndices());

			int nRow = ru - rl;
			MatrixBlock ot = sparseMB(ru - rl, maxCol);
			MatrixBlock bt = sparseMB(ru - rl, maxCol);
			decompressToSparseBlock(a, b, ot, bt, 0, nRow);

			MatrixBlock otd = denseMB(ru - rl, maxCol);
			MatrixBlock btd = denseMB(ru - rl, maxCol);
			decompressToDenseBlock(otd, btd, a, b, 0, nRow);

			UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARP.toString(), 1), 0, newNRow, a, b, newNRow);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	private void expectNotImplementedSlice(int rl, int ru) {
		boolean exception = false;
		try {
			base.sliceRows(rl, ru);
			other.sliceRows(rl, ru);
		}
		catch(NotImplementedException nie) {
			// good
			exception = true;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
		assertTrue(exception);
	}

	@Test
	public void getScheme() {
		try {
			// create scheme and check if it compress the same matrix input in same way.
			compare(base, other);
			checkScheme(base.getCompressionScheme(), base, nRow, maxCol);
			checkScheme(other.getCompressionScheme(), other, nRow, maxCol);
		}
		catch(NotImplementedException e) {
			// allow it to be not implemented
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static void checkScheme(ICLAScheme ia, AColGroup a, int nRow, int nCol) {
		try {
			if(ia != null) {
				// if whatever is returned is not null,
				// then it should be able to compress the matrix block again
				// in same scheme
				MatrixBlock ot = sparseMB(nRow, nCol);
				a.decompressToSparseBlock(ot.getSparseBlock(), 0, nRow);
				ot.recomputeNonZeros();

				AColGroup g = ia.encode(ot);
				if(g == null)
					fail("Should not be possible to return null on equivalent matrix:\nGroup\n" + a + "\nScheme:\n" + ia);

			}
		}
		catch(NotImplementedException e) {
			// allow it to be not implemented

		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testAppendSelf() {
		appendSelfVerification(base);
		appendSelfVerification(other);
	}

	@Test
	public void testAppendSomethingElse() {
		// This is under the assumption that if one is appending
		// to the other then other should append to this.
		// If this property does not hold it is because some cases are missing in the append logic.
		try {

			AColGroup g2 = base.append(other);
			AColGroup g2n = other.append(base);
			// both should be null, or both should not be.
			if(g2 == null)
				assertTrue(g2n == null);
			else
				assertTrue(g2n != null);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private void appendSelfVerification(AColGroup g) {
		try {

			AColGroup g2 = g.append(g);
			AColGroup g2n = AColGroup.appendN(new AColGroup[] {g, g}, nRow, nRow * 2);
			if(g2 != null && g2n != null) {
				double s2 = g2.getSum(nRow * 2);
				double s = g.getSum(nRow) * 2;
				double s2n = g2n.getSum(nRow * 2);
				assertEquals(s2, s, 0.0001);
				assertEquals(s2n, s, 0.0001);

				UA_ROW(InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARP.toString(), 1), 0, nRow * 2, g2, g2n, nRow * 2);
			}
		}
		catch(NotImplementedException e) {
			// okay
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

}

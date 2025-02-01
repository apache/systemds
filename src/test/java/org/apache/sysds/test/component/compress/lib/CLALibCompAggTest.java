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

package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.CompressibleInputGenerator;
import org.junit.Test;

public class CLALibCompAggTest {

	MatrixBlock mb = CompressibleInputGenerator.getInput(250, 10, CompressionType.RLE, 10, 0.9, 2341);

	CompressedMatrixBlock cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();

	@Test
	public void uavar() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAVAR.toString(), 1);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "variance");
	}

	@Test
	public void uamult() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAM.toString(), 1);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "product");
	}

	@Test
	public void uarmult() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARM.toString(), 1);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "product");
	}

	@Test
	public void uacmult() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACM.toString(), 1);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "product");
	}

	@Test
	public void uarimax() {
		try {
			AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARIMAX.toString(), 1);
			MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
			MatrixBlock uRet = mb.aggregateUnaryOperations(op);
			TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "maxindexs");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = Exception.class)
	public void custom_invalid_aggregation() {
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject(), CorrectionLocationType.LASTFOURCOLUMNS);
		AggregateUnaryOperator op = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), 1);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op, null, 1000, null, false);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op, null, 1000, null, false);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, Opcodes.UAMAX.toString());
	}

	@Test
	public void uamultOverlapping() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAM.toString(), 1);
		cmb.setOverlapping(true);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "product");
	}

	@Test
	public void uamultOverlapping_noCache() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAM.toString(), 1);
		cmb.setOverlapping(true);
		CompressedMatrixBlock spy = spy(cmb);
		when(spy.getCachedDecompressed()).thenReturn(null);

		MatrixBlock cRet = spy.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "product");
	}

	@Test
	public void uamaxOverlapping() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString(), 1);
		cmb.setOverlapping(true);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "max");
	}

	@Test
	public void uamaxOverlapping_noCache() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString(), 1);
		cmb.setOverlapping(true);
		CompressedMatrixBlock spy = spy(cmb);
		when(spy.getCachedDecompressed()).thenReturn(null);

		MatrixBlock cRet = spy.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "max");
	}

	@Test
	public void uamaxPrefilterSingleThread() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString(), 1);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 10);
		CompressedMatrixBlock cmbt = (CompressedMatrixBlock) cmb.append(c);
		cmbt.setOverlapping(true);
		MatrixBlock tmb = mb.append(new MatrixBlock(cmb.getNumRows(), 1, 10.0));

		MatrixBlock cRet = cmbt.aggregateUnaryOperations(op);
		MatrixBlock uRet = tmb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "max");
	}

	@Test
	public void uamaxPrefilterParallel() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString(), 10);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 10);
		CompressedMatrixBlock cmbt = (CompressedMatrixBlock) cmb.append(c);
		cmbt.setOverlapping(true);
		MatrixBlock tmb = mb.append(new MatrixBlock(cmb.getNumRows(), 1, 10.0));

		MatrixBlock cRet = cmbt.aggregateUnaryOperations(op);
		MatrixBlock uRet = tmb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "max");
	}

	@Test
	public void uarmaxPrefilterParallel() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMAX.toString(), 10);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 10);
		CompressedMatrixBlock cmbt = (CompressedMatrixBlock) cmb.append(c);
		cmbt.setOverlapping(true);
		MatrixBlock tmb = mb.append(new MatrixBlock(cmb.getNumRows(), 1, 10.0));

		MatrixBlock cRet = cmbt.aggregateUnaryOperations(op);
		MatrixBlock uRet = tmb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "max");
	}

	@Test
	public void uacmaxPrefilterParallel() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UACMAX.toString(), 10);
		CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 10);
		CompressedMatrixBlock cmbt = (CompressedMatrixBlock) cmb.append(c);
		cmbt.setOverlapping(true);
		MatrixBlock tmb = mb.append(new MatrixBlock(cmb.getNumRows(), 1, 10.0));

		MatrixBlock cRet = cmbt.aggregateUnaryOperations(op);
		MatrixBlock uRet = tmb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "max");
	}

	@Test
	public void rowsum_compressedReturn() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 10);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowsum");
		assertTrue(cRet instanceof CompressedMatrixBlock);
	}

	@Test
	public void rowmean() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMEAN.toString(), 10);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowmean");
	}

	@Test
	public void rowmeanDecompressing() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMEAN.toString(), 10);
		CompressedMatrixBlock spy = spy(cmb);
		when(spy.isOverlapping()).thenReturn(true);
		when(spy.getCachedDecompressed()).thenReturn(null);
		MatrixBlock cRet = spy.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowmean");
	}

	@Test
	public void rowSquareSumDecompressing() {
		try{
			AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARSQKP.toString(), 10);
			CompressedMatrixBlock spy = spy(cmb);
			when(spy.isOverlapping()).thenReturn(true);
			when(spy.getCachedDecompressed()).thenReturn(null);
			MatrixBlock cRet = spy.aggregateUnaryOperations(op);
			MatrixBlock uRet = mb.aggregateUnaryOperations(op);
			TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowmean");
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}
	}


	@Test
	public void rowMin() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMIN.toString(), 10);
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowmin");
	}

	@Test
	public void rowMinSparseLotsOfZero() {
		AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARMIN.toString(), 10);

		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 1000, 1, 1, 0.01, 2341);

		CompressedMatrixBlock cmb = (CompressedMatrixBlock) CompressedMatrixBlockFactory.compress(mb, 1).getLeft();
		cmb.setOverlapping(true);
		cmb.clearSoftReferenceToDecompressed();
		MatrixBlock cRet = cmb.aggregateUnaryOperations(op);
		MatrixBlock uRet = mb.aggregateUnaryOperations(op);
		TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowmin");
	}



	@Test
	public void rowsum_compressedReturn2() {
		try {
			AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 10);

			CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 0);
			CompressedMatrixBlock cmbt = (CompressedMatrixBlock) cmb.append(c);
			cmbt.setOverlapping(true);
			MatrixBlock tmb = mb.append(new MatrixBlock(cmb.getNumRows(), 1, 0.0));

			MatrixBlock cRet = cmbt.aggregateUnaryOperations(op);
			MatrixBlock uRet = tmb.aggregateUnaryOperations(op);
			TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowsum");
			assertTrue(cRet instanceof CompressedMatrixBlock);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void rowsum_compressedReturn3() {
		try {
			AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 10);

			CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, -1);
			CompressedMatrixBlock c2 = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 1);
			MatrixBlock ctmp = c.append(c2);

			MatrixBlock mb1 = new MatrixBlock(cmb.getNumRows(), 1, -1.0);
			MatrixBlock mb2 = new MatrixBlock(cmb.getNumRows(), 1, 1.0);
			MatrixBlock tmb = mb1.append(mb2);

			MatrixBlock cRet = ctmp.aggregateUnaryOperations(op);
			MatrixBlock uRet = tmb.aggregateUnaryOperations(op);

			TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowsum");
			assertTrue(cRet.isEmpty());
			assertTrue(cRet instanceof MatrixBlock);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void rowsum_compressedReturn4() {
		try {
			AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 10);

			CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 12);
			CompressedMatrixBlock c2 = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 1);
			MatrixBlock ctmp = c.append(c2);

			MatrixBlock mb1 = new MatrixBlock(cmb.getNumRows(), 1, 12.0);
			MatrixBlock mb2 = new MatrixBlock(cmb.getNumRows(), 1, 1.0);
			MatrixBlock tmb = mb1.append(mb2);

			MatrixBlock cRet = ctmp.aggregateUnaryOperations(op);
			MatrixBlock uRet = tmb.aggregateUnaryOperations(op);

			TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowsum");
			assertTrue(cRet instanceof CompressedMatrixBlock);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void rowsum_compressedReturn5() {
		try {
			AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARKP.toString(), 10);

			op = new AggregateUnaryOperator(new AggregateOperator(0, Plus.getPlusFnObject(), CorrectionLocationType.NONE),
				op.indexFn, op.getNumThreads());
			CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 12);
			CompressedMatrixBlock c2 = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 1);
			MatrixBlock ctmp = c.append(c2);

			MatrixBlock mb1 = new MatrixBlock(cmb.getNumRows(), 1, 12.0);
			MatrixBlock mb2 = new MatrixBlock(cmb.getNumRows(), 1, 1.0);
			MatrixBlock tmb = mb1.append(mb2);

			MatrixBlock cRet = ctmp.aggregateUnaryOperations(op);
			MatrixBlock uRet = tmb.aggregateUnaryOperations(op);

			TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowsum");
			assertTrue(cRet instanceof CompressedMatrixBlock);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}


	@Test
	public void notRowSumThereforeNotCompressed() {
		try {
			AggregateUnaryOperator op = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UARSQKP.toString(), 10);
			CompressedMatrixBlock c = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 12);
			CompressedMatrixBlock c2 = CompressedMatrixBlockFactory.createConstant(cmb.getNumRows(), 1, 1);
			MatrixBlock ctmp = c.append(c2);

			MatrixBlock mb1 = new MatrixBlock(cmb.getNumRows(), 1, 12.0);
			MatrixBlock mb2 = new MatrixBlock(cmb.getNumRows(), 1, 1.0);
			MatrixBlock tmb = mb1.append(mb2);

			MatrixBlock cRet = ctmp.aggregateUnaryOperations(op);
			MatrixBlock uRet = tmb.aggregateUnaryOperations(op);

			TestUtils.compareMatricesPercentageDistance(uRet, cRet, 0, 0, "rowsum");
			assertTrue(cRet instanceof MatrixBlock);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}

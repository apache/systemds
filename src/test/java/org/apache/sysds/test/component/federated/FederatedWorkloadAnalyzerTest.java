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

package org.apache.sysds.test.component.federated;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedWorkloadAnalyzer;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FederatedWorkloadAnalyzerTest {
	protected static final Log LOG = LogFactory.getLog(FederatedWorkloadAnalyzerTest.class.getName());

	/** Async compression triggered by compressRun runs on a worker thread, so poll instead of sleeping. */
	private static final int COMPRESS_TIMEOUT_MS = 10000;

	private final FederatedWorkloadAnalyzer analyzer = new FederatedWorkloadAnalyzer();

	// --------------------------------------------------------------------------------------------
	// AggregateBinary (matrix multiply)
	// --------------------------------------------------------------------------------------------

	@Test
	public void aggregateBinaryBothSidesCounted() {
		// left 100x100 (valid), right 100x50 (valid)
		ExecutionContext ec = ec("1", mo(100, 100), "2", mo(100, 50));
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(ec, mm, mm("1", "2"));

		// left side: RMM with the right-hand column count, plus overlapping decompress sized by c2
		InstructionTypeCounter left = mm.get(1L);
		assertEquals(50, left.getRightMultiplications());
		assertEquals(50, left.getOverlappingDecompressions());
		// right side: LMM with the left-hand row count
		InstructionTypeCounter right = mm.get(2L);
		assertEquals(100, right.getLeftMultiplications());
	}

	@Test
	public void aggregateBinaryOnlyLeftCountedWhenRightTooSmall() {
		// left 100x10 (valid), right 10x5 (too few rows -> invalid)
		ExecutionContext ec = ec("1", mo(100, 10), "2", mo(10, 5));
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(ec, mm, mm("1", "2"));

		InstructionTypeCounter left = mm.get(1L);
		assertEquals(5, left.getRightMultiplications());
		assertEquals(5, left.getOverlappingDecompressions());
		// right side never tracked because it does not pass validSize
		assertFalse(mm.containsKey(2L));
	}

	@Test
	public void aggregateBinaryNeitherCountedWhenBothTooSmall() {
		ExecutionContext ec = ec("1", mo(10, 10), "2", mo(10, 10));
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(ec, mm, mm("1", "2"));

		assertTrue(mm.isEmpty());
	}

	@Test
	public void aggregateBinaryWideOperandNotCounted() {
		// 100x200: enough rows (>90) but more columns than rows -> validSize false on the second clause
		ExecutionContext ec = ec("1", mo(100, 200), "2", mo(10, 5));
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(ec, mm, mm("1", "2"));

		assertTrue(mm.isEmpty());
	}

	// --------------------------------------------------------------------------------------------
	// MMChain
	// --------------------------------------------------------------------------------------------

	@Test
	public void mmChainCountsOneLeftAndOneRight() {
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(null, mm, mmchain("1"));

		InstructionTypeCounter c = mm.get(1L);
		assertEquals(1, c.getRightMultiplications());
		assertEquals(1, c.getLeftMultiplications());
	}

	// --------------------------------------------------------------------------------------------
	// AggregateUnary
	// --------------------------------------------------------------------------------------------

	@Test
	public void aggregateUnaryColSumsIsDictOp() {
		// colSums -> ReduceRow -> compression friendly (2 dict ops, no decompress)
		assertDictOpsAndDecompress(Opcodes.UACKP.toString(), 2, 0);
	}

	@Test
	public void aggregateUnaryFullSumIsDictOp() {
		// sum -> ReduceAll -> compression friendly (2 dict ops, no decompress)
		assertDictOpsAndDecompress(Opcodes.UAKP.toString(), 2, 0);
	}

	@Test
	public void aggregateUnaryRowSumsIsDictOp() {
		// rowSums -> ReduceCol with KahanPlus -> compression friendly (2 dict ops, no decompress)
		assertDictOpsAndDecompress(Opcodes.UARKP.toString(), 2, 0);
	}

	@Test
	public void aggregateUnaryRowMeansIsDictOp() {
		// rowMeans -> ReduceCol with Mean -> compression friendly (2 dict ops, no decompress)
		assertDictOpsAndDecompress(Opcodes.UARMEAN.toString(), 2, 0);
	}

	@Test
	public void aggregateUnaryRowProductsForcesDecompress() {
		// rowProds -> ReduceCol with Multiply -> not friendly (1 dict op + 1 decompress)
		assertDictOpsAndDecompress(Opcodes.UARM.toString(), 1, 1);
	}

	@Test
	public void aggregateUnaryRowSumsPlusIsDictOp() {
		// rowSums (plain Plus, no Kahan) -> ReduceCol with Plus -> compression friendly (2 dict ops)
		assertDictOpsAndDecompress(Opcodes.UARP.toString(), 2, 0);
	}

	@Test
	public void aggregateUnaryRowMaxForcesDecompress() {
		// rowMax -> ReduceCol with Builtin max -> not friendly (1 dict op + 1 decompress)
		assertDictOpsAndDecompress(Opcodes.UARMAX.toString(), 1, 1);
	}

	@Test
	public void aggregateUnaryNonAggregateOperatorIgnored() {
		// nrow uses a SimpleOperator (not an AggregateUnaryOperator) so nothing is tracked
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(null, mm, uagg(Opcodes.NROW.toString(), "1"));

		assertTrue(mm.isEmpty());
	}

	private void assertDictOpsAndDecompress(String opcode, int expectedDictOps, int expectedDecompress) {
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(null, mm, uagg(opcode, "1"));

		InstructionTypeCounter c = mm.get(1L);
		assertEquals("Unexpected dict-ops for " + opcode, expectedDictOps, c.getDictionaryOps());
		assertEquals("Unexpected decompressions for " + opcode, expectedDecompress, c.getDecompressions());
	}

	// --------------------------------------------------------------------------------------------
	// Instance level dispatch + async compress trigger
	// --------------------------------------------------------------------------------------------

	@Test
	public void compressRunCompressesAfterEnoughWorkload() {
		final long tid = 1;
		final int dim = 100, iter = 10;
		// Right operand is left-multiplied each matmul, accumulating LMM = leftRows (=dim) per
		// invocation, so iter=10 yields LMM=1000 on a 100x100 rounded block. This mirrors the shape
		// and counter that FedWorkerMatrixMultiplyWorkload relies on to trigger compression.
		MatrixBlock rightBlock = TestUtils.round(TestUtils.generateTestMatrixBlock(dim, dim, 0.5, 2.5, 1.0, 222));
		MatrixBlock probeBlock = new MatrixBlock();
		probeBlock.copy(rightBlock);

		MatrixObject left = compressibleMO(dim, dim, 7);
		MatrixObject right = wrap(rightBlock);
		ExecutionContext ec = ec("1", left, "2", right);

		// each matmul with two valid sides increments the counter twice; reaching the
		// compressRunFrequency threshold of 10 schedules an async compression pass
		ComputationCPInstruction ins = mm("1", "2");
		for(int i = 0; i < iter; i++)
			analyzer.incrementWorkload(ec, tid, ins);

		analyzer.compressRun(ec, tid);

		// Only assert the async compression materialized if the cost model would compress this shape
		// locally; otherwise the workload pass legitimately leaves it uncompressed (matches the skip
		// pattern in FedWorkerMatrixMultiplyWorkload).
		InstructionTypeCounter probe = new InstructionTypeCounter(0, 0, 0, dim * iter, 0, 0, 0, 0, false);
		boolean locallyCompressible = CompressedMatrixBlockFactory.compress(probeBlock, probe)
			.getLeft() instanceof CompressedMatrixBlock;
		if(locallyCompressible)
			assertCompressedWithinTimeout(right);
	}

	@Test
	public void compressRunNoOpBelowThreshold() {
		final long tid = 2;
		MatrixObject left = compressibleMO(500, 10, 7);
		MatrixObject right = compressibleMO(500, 10, 13);
		ExecutionContext ec = ec("1", left, "2", right);

		// only two invocations -> counter = 4, below threshold, so nothing compresses
		ComputationCPInstruction ins = mm("1", "2");
		analyzer.incrementWorkload(ec, tid, ins);
		analyzer.incrementWorkload(ec, tid, ins);

		analyzer.compressRun(ec, tid);

		assertFalse(left.acquireReadAndRelease() instanceof CompressedMatrixBlock);
		assertFalse(right.acquireReadAndRelease() instanceof CompressedMatrixBlock);
	}

	@Test
	public void nonComputationInstructionIgnored() {
		// the public entry point silently ignores non-CP / non-computation instructions
		analyzer.incrementWorkload(null, 99, (Instruction) null);
		analyzer.compressRun(null, 99);
	}

	@Test
	public void unhandledComputationInstructionIgnored() {
		// a transpose is a ComputationCPInstruction but none of the tracked shapes -> no counters
		ConcurrentHashMap<Long, InstructionTypeCounter> mm = new ConcurrentHashMap<>();

		analyzer.incrementWorkload(null, mm, reorg("1"));

		assertTrue(mm.isEmpty());
	}

	@Test
	public void toStringReportsState() {
		String s = analyzer.toString();
		assertTrue(s.contains(FederatedWorkloadAnalyzer.class.getSimpleName()));
		assertTrue(s.contains("Counter"));
	}

	// --------------------------------------------------------------------------------------------
	// helpers
	// --------------------------------------------------------------------------------------------

	private static void assertCompressedWithinTimeout(MatrixObject mo) {
		final long deadline = System.currentTimeMillis() + COMPRESS_TIMEOUT_MS;
		while(System.currentTimeMillis() < deadline) {
			if(mo.acquireReadAndRelease() instanceof CompressedMatrixBlock)
				return;
			try {
				Thread.sleep(50);
			}
			catch(InterruptedException e) {
				Thread.currentThread().interrupt();
				fail("Interrupted while waiting for async compression");
			}
		}
		fail("Matrix was not compressed by the workload analyzer within " + COMPRESS_TIMEOUT_MS + "ms");
	}

	private static ExecutionContext ec(String n1, MatrixObject m1, String n2, MatrixObject m2) {
		LocalVariableMap vars = new LocalVariableMap();
		ExecutionContext ec = new ExecutionContext(vars);
		ec.setVariable(n1, m1);
		ec.setVariable(n2, m2);
		return ec;
	}

	/** Build a MatrixObject of the requested shape (data content irrelevant for the counters). */
	private static MatrixObject mo(int rows, int cols) {
		return wrap(new MatrixBlock(rows, cols, 0.0));
	}

	private static MatrixObject compressibleMO(int rows, int cols, int seed) {
		return wrap(TestUtils.round(TestUtils.generateTestMatrixBlock(rows, cols, 0, 3, 1.0, seed)));
	}

	private static MatrixObject wrap(MatrixBlock mb) {
		MatrixCharacteristics mc = new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), -1, mb.getNonZeros());
		MetaDataFormat md = new MetaDataFormat(mc, FileFormat.BINARY);
		MatrixObject mo = new MatrixObject(ValueType.FP64, "/dev/null", md, mb);
		mo.getDataCharacteristics().setDimension(mb.getNumRows(), mb.getNumColumns());
		return mo;
	}

	private static ComputationCPInstruction mm(String in1, String in2) {
		String str = InstructionUtils.concatOperands("CP", Opcodes.MMULT.toString(), in1, in2, "3", "16");
		return AggregateBinaryCPInstruction.parseInstruction(str);
	}

	private static ComputationCPInstruction mmchain(String in1) {
		String str = InstructionUtils.concatOperands("CP", Opcodes.MMCHAIN.toString(), in1, "2", "3", "XtXv", "16");
		return MMChainCPInstruction.parseInstruction(str);
	}

	private static ComputationCPInstruction uagg(String opcode, String in1) {
		String str = InstructionUtils.concatOperands("CP", opcode, in1, "2", "16");
		return AggregateUnaryCPInstruction.parseInstruction(str);
	}

	private static ComputationCPInstruction reorg(String in1) {
		String str = InstructionUtils.concatOperands("CP", Opcodes.TRANSPOSE.toString(), in1, "2", "16");
		return ReorgCPInstruction.parseInstruction(str);
	}
}

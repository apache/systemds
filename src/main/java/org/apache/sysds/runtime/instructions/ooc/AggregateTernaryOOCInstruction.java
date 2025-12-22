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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

public class AggregateTernaryOOCInstruction extends ComputationOOCInstruction {

	private static final Log LOG = LogFactory.getLog(AggregateTernaryOOCInstruction.class.getName());

	private AggregateTernaryOOCInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String istr) {
		super(OOCInstruction.OOCType.AggregateTernary, op, in1, in2, in3, out, opcode, istr);
	}

	public static AggregateTernaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase(Opcodes.TAKPM.toString()) || opcode.equalsIgnoreCase(Opcodes.TACKPM.toString())) {
			InstructionUtils.checkNumFields(parts , 4, 5);

			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			//int numThreads = parts.length == 6 ? Integer.parseInt(parts[5]) : 1;

			AggregateTernaryOperator op = InstructionUtils.parseAggregateTernaryOperator(opcode, 1);
			return new AggregateTernaryOOCInstruction(op, in1, in2, in3, out, opcode, str);
		}
		throw new DMLRuntimeException("AggregateTernaryOOCInstruction.parseInstruction():: Unknown opcode " + opcode);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject m1 = ec.getMatrixObject(input1);
		MatrixObject m2 = ec.getMatrixObject(input2);
		MatrixObject m3 = input3.isLiteral() ? null : ec.getMatrixObject(input3);

		AggregateTernaryOperator abOp = (AggregateTernaryOperator) _optr;
		validateInput(m1, m2, m3, abOp, input1.getName(), input2.getName(), input3.getName());

		boolean isReduceAll = abOp.indexFn instanceof ReduceAll;

		OOCStream<IndexedMatrixValue> qIn1 = m1.getStreamHandle();
		OOCStream<IndexedMatrixValue> qIn2 = m2.getStreamHandle();
		OOCStream<IndexedMatrixValue> qIn3 = m3 == null ? null : m3.getStreamHandle();

		if(isReduceAll)
			processReduceAll(ec, abOp, qIn1, qIn2, qIn3);
		else
			processReduceRow(ec, abOp, qIn1, qIn2, qIn3, m1.getDataCharacteristics());
	}

	private void processReduceAll(ExecutionContext ec, AggregateTernaryOperator abOp,
		OOCStream<IndexedMatrixValue> qIn1, OOCStream<IndexedMatrixValue> qIn2, OOCStream<IndexedMatrixValue> qIn3) {

		final int extra = abOp.aggOp.correction.getNumRemovedRowsColumns();
		final MatrixBlock agg = new MatrixBlock(1, 1 + extra, false);
		final MatrixBlock corr = new MatrixBlock(1, 1 + extra, false);

		OOCStream<IndexedMatrixValue> qMid = createWritableStream();

		List<OOCStream<IndexedMatrixValue>> streams = new ArrayList<>();
		streams.add(qIn1);
		streams.add(qIn2);
		if(qIn3 != null)
			streams.add(qIn3);

		CompletableFuture<Void> fut = joinOOC(streams, qMid, blocks -> {
			MatrixBlock b1 = (MatrixBlock) blocks.get(0).getValue();
			MatrixBlock b2 = (MatrixBlock) blocks.get(1).getValue();
			MatrixBlock b3 = blocks.size() == 3 ? (MatrixBlock) blocks.get(2).getValue() : null;
			MatrixBlock partial = MatrixBlock.aggregateTernaryOperations(b1, b2, b3, new MatrixBlock(), abOp, false);
			return new IndexedMatrixValue(blocks.get(0).getIndexes(), partial);
		}, IndexedMatrixValue::getIndexes);

		try {
			IndexedMatrixValue imv;
			while((imv = qMid.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
				MatrixBlock partial = (MatrixBlock) imv.getValue();
				OperationsOnMatrixValues.incrementalAggregation(agg,
					abOp.aggOp.existsCorrection() ? corr : null, partial, abOp.aggOp, true);
			}
			fut.join();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}

		agg.dropLastRowsOrColumns(abOp.aggOp.correction);
		ec.setScalarOutput(output.getName(), new DoubleObject(agg.get(0, 0)));
	}

	private void processReduceRow(ExecutionContext ec, AggregateTernaryOperator abOp,
		OOCStream<IndexedMatrixValue> qIn1, OOCStream<IndexedMatrixValue> qIn2, OOCStream<IndexedMatrixValue> qIn3,
		DataCharacteristics dc) {

		long emitThreshold = dc.getNumRowBlocks();
		if(emitThreshold <= 0)
			throw new DMLRuntimeException("Unknown number of row blocks for out-of-core aggregate ternary.");

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		OOCStream<IndexedMatrixValue> qMid = createWritableStream();

		List<OOCStream<IndexedMatrixValue>> streams = new ArrayList<>();
		streams.add(qIn1);
		streams.add(qIn2);
		if(qIn3 != null)
			streams.add(qIn3);

		for (OOCStream<IndexedMatrixValue> stream : streams)
			stream.setDownstreamMessageRelay(qOut::messageDownstream);

		qOut.setUpstreamMessageRelay(msg ->
			streams.forEach(stream -> stream.messageUpstream(streams.size() > 1 ? msg.split() : msg)));

		qOut.setIXTransform((downstream, range) -> {
			if (downstream)
				return new IndexRange(1, 1, range.colStart, range.colEnd);
			else
				return new IndexRange(1, dc.getRows(),  range.colStart, range.colEnd);
		});

		CompletableFuture<Void> fut = joinOOC(streams, qMid, blocks -> {
			MatrixBlock b1 = (MatrixBlock) blocks.get(0).getValue();
			MatrixBlock b2 = (MatrixBlock) blocks.get(1).getValue();
			MatrixBlock b3 = blocks.size() == 3 ? (MatrixBlock) blocks.get(2).getValue() : null;
			MatrixBlock partial = MatrixBlock.aggregateTernaryOperations(b1, b2, b3, new MatrixBlock(), abOp, false);
			return new IndexedMatrixValue(blocks.get(0).getIndexes(), partial);
		}, IndexedMatrixValue::getIndexes);

		final Map<Long, MatrixBlock> aggMap = new HashMap<>();
		final Map<Long, MatrixBlock> corrMap = new HashMap<>();
		final Map<Long, Integer> cntMap = new HashMap<>();

		try {
			IndexedMatrixValue imv;
			while((imv = qMid.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
				MatrixIndexes idx = imv.getIndexes();
				long colIx = idx.getColumnIndex();
				MatrixBlock partial = (MatrixBlock) imv.getValue();

				MatrixBlock curAgg = aggMap.get(colIx);
				MatrixBlock curCorr = corrMap.get(colIx);
				if(curAgg == null) {
					aggMap.put(colIx, partial);
					curCorr = new MatrixBlock(partial.getNumRows(), partial.getNumColumns(), false);
					corrMap.put(colIx, curCorr);
					cntMap.put(colIx, 1);
				}
				else {
					OperationsOnMatrixValues.incrementalAggregation(curAgg, abOp.aggOp.existsCorrection() ? curCorr : null,
						partial, abOp.aggOp, true);
					cntMap.put(colIx, cntMap.get(colIx) + 1);
				}

				if(cntMap.get(colIx) >= emitThreshold) {
					MatrixBlock finalAgg = aggMap.remove(colIx);
					corrMap.remove(colIx);
					cntMap.remove(colIx);

					finalAgg.dropLastRowsOrColumns(abOp.aggOp.correction);
					MatrixIndexes outIdx = new MatrixIndexes(1, colIx);
					qOut.enqueue(new IndexedMatrixValue(outIdx, finalAgg));
				}
			}
			fut.join();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally {
			qOut.closeInput();
		}
	}

	private static void validateInput(MatrixObject m1, MatrixObject m2, MatrixObject m3, AggregateTernaryOperator op,
		String name1, String name2, String name3) {

		DataCharacteristics c1 = m1.getDataCharacteristics();
		DataCharacteristics c2 = m2.getDataCharacteristics();
		DataCharacteristics c3 = m3 == null ? c2 : m3.getDataCharacteristics();

		long m1r = c1.getRows();
		long m2r = c2.getRows();
		long m3r = c3.getRows();
		long m1c = c1.getCols();
		long m2c = c2.getCols();
		long m3c = c3.getCols();

		if(m1r <= 0 || m2r <= 0 || m3r <= 0 || m1c <= 0 || m2c <= 0 || m3c <= 0)
			throw new DMLRuntimeException("Unknown dimensions for aggregate ternary inputs.");

		if(m1r != m2r || m1c != m2c || m2r != m3r || m2c != m3c){
			if(LOG.isTraceEnabled()){
				LOG.trace("matBlock1:" + name1 + " (" + m1r + "x" + m1c + ")");
				LOG.trace("matBlock2:" + name2 + " (" + m2r + "x" + m2c + ")");
				LOG.trace("matBlock3:" + name3 + " (" + m3r + "x" + m3c + ")");
			}
			throw new DMLRuntimeException("Invalid dimensions for aggregate ternary (" + m1r + "x" + m1c + ", "
				+ m2r + "x" + m2c + ", " + m3r + "x" + m3c + ").");
		}

		if(!(op.aggOp.increOp.fn instanceof KahanPlus && op.binaryFn instanceof Multiply))
			throw new DMLRuntimeException("Unsupported operator for aggregate ternary operations.");

	}
}

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

import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class UnaryOOCInstruction extends ComputationOOCInstruction {
	private UnaryOperator _uop = null;

	protected UnaryOOCInstruction(OOCType type, UnaryOperator op, CPOperand in1, CPOperand out, String opcode, String istr) {
		super(type, op, in1, out, opcode, istr);

		_uop = op;
	}

	protected UnaryOOCInstruction(OOCType type, UnaryOperator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);

		_uop = op;
	}

	public static UnaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 2);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);

		UnaryOperator uopcode = InstructionUtils.parseUnaryOperator(opcode);
		return new UnaryOOCInstruction(OOCType.Unary, uopcode, in1, out, opcode, str);
	}

	public void processInstruction( ExecutionContext ec ) {
		UnaryOperator uop = (UnaryOperator) _uop;
		// Create thread and process the unary operation
		MatrixObject min = ec.getMatrixObject(input1);
		boolean cumSumProd = Builtin.isBuiltinCode(uop.fn, BuiltinCode.CUMSUMPROD);
		ec.getDataCharacteristics(output.getName()).set(min.getNumRows(), cumSumProd ? 1 : min.getNumColumns(),
			min.getBlocksize(), -1);
		OOCStream<IndexedMatrixValue> qIn = min.getStreamHandle();
		OOCStream<IndexedMatrixValue> qOut;
		boolean cumulative = isCumulativeUnary(uop);

		if(cumulative) {
			qOut = processCumulativeUnaryInstruction(ec, uop, qIn);
		}
		else {
			qOut = createWritableStream();
			mapOOC(qIn, qOut, tmp -> {
				IndexedMatrixValue tmpOut = new IndexedMatrixValue();
				tmpOut.set(tmp.getIndexes(), tmp.getValue().unaryOperations(uop, new MatrixBlock()));
				return tmpOut;
			});
		}

		ec.getMatrixObject(output).setStreamHandle(qOut);
		if(!cumulative) {
			qIn.setDownstreamMessageRelay(qOut::messageDownstream);
			qOut.setUpstreamMessageRelay(qIn::messageUpstream);
		}
	}

	private OOCStream<IndexedMatrixValue> processCumulativeUnaryInstruction(ExecutionContext ec, UnaryOperator uop,
		OOCStream<IndexedMatrixValue> qIn) {
		DataCharacteristics dc = ec.getDataCharacteristics(input1.getName());
		if(!dc.dimsKnown())
			throw new DMLRuntimeException(
				"OOC cumulative unary operations require known dimensions for deterministic block ordering.");

		BuiltinCode bcode = ((Builtin)uop.fn).getBuiltinCode();
		long rowBlocks = dc.getNumRowBlocks();
		long colBlocks = dc.getNumColBlocks();
		boolean rowCum = (bcode == BuiltinCode.ROWCUMSUM);
		boolean sumProd = (bcode == BuiltinCode.CUMSUMPROD);
		if(sumProd && colBlocks != 1)
			throw new DMLRuntimeException(
				"Unsupported OOC cumulative sum-product with more than one column block: " + colBlocks);

		long outerSize = rowCum ? rowBlocks : colBlocks;
		long innerSize = rowCum ? colBlocks : rowBlocks;
		if(outerSize > Integer.MAX_VALUE)
			throw new DMLRuntimeException(
				"Unsupported number of cumulative partitions: " + outerSize + " (max " + Integer.MAX_VALUE + ").");

		int partitions = Math.toIntExact(outerSize);
		List<OOCStream<IndexedMatrixValue>> splitInputs = splitOOCStream(qIn, imv -> {
			long outerIx = rowCum ? imv.getIndexes().getRowIndex() : imv.getIndexes().getColumnIndex();
			return (int) (outerIx - 1);
		}, partitions);

		List<OOCStream<IndexedMatrixValue>> splitOutputs = new ArrayList<>(partitions);

		for(int i = 0; i < partitions; i++) {
			OOCStream<IndexedMatrixValue> partOut = createWritableStream();
			splitOutputs.add(partOut);

			this.<IndexedMatrixValue, double[]>scanOOC(splitInputs.get(i), partOut,
				imv -> rowCum ? imv.getIndexes().getColumnIndex() : imv.getIndexes().getRowIndex(), (imv, agg) -> {
					MatrixBlock inBlk = (MatrixBlock) imv.getValue();
					int outRows = inBlk.getNumRows();
					int outCols = sumProd ? 1 : inBlk.getNumColumns();
					MatrixBlock outBlk = LibMatrixAgg.cumaggregateUnaryMatrix(inBlk,
						new MatrixBlock(outRows, outCols, false), uop, agg);
					MatrixIndexes idx = imv.getIndexes();
					IndexedMatrixValue out = new IndexedMatrixValue(new MatrixIndexes(idx.getRowIndex(), idx.getColumnIndex()),
						outBlk);
					double[] nextCarry = rowCum ? extractLastColumn(outBlk) : extractLastRow(outBlk);
					return new ScanStep<>(out, nextCarry);
				}, innerSize).exceptionally(t -> {
				partOut.propagateFailure(DMLRuntimeException.of(t));
				return null;
			});
		}

		return mergeOOCStreams(splitOutputs);
	}

	private static double[] extractLastRow(MatrixBlock blk) {
		int rows = blk.getNumRows();
		int cols = blk.getNumColumns();
		double[] ret = new double[cols];
		if(rows == 0 || cols == 0)
			return ret;
		int lr = rows - 1;
		for(int j = 0; j < cols; j++)
			ret[j] = blk.get(lr, j);
		return ret;
	}

	private static double[] extractLastColumn(MatrixBlock blk) {
		int rows = blk.getNumRows();
		int cols = blk.getNumColumns();
		double[] ret = new double[rows];
		if(rows == 0 || cols == 0)
			return ret;
		int lc = cols - 1;
		for(int i = 0; i < rows; i++)
			ret[i] = blk.get(i, lc);
		return ret;
	}

	private static boolean isCumulativeUnary(UnaryOperator uop) {
		return Builtin.isBuiltinCode(uop.fn, BuiltinCode.CUMSUM, BuiltinCode.ROWCUMSUM, BuiltinCode.CUMPROD,
			BuiltinCode.CUMMIN, BuiltinCode.CUMMAX, BuiltinCode.CUMSUMPROD);
	}
}

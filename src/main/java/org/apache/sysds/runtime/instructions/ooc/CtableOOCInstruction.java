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

import java.util.HashMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.Ctable;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.CTableMap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.LongLongDoubleHashMap;

public class CtableOOCInstruction extends ComputationOOCInstruction {
	private final CPOperand _outDim1;
	private final CPOperand _outDim2;
	private final boolean _ignoreZeros;

	protected CtableOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, CPOperand outDim1, CPOperand outDim2, boolean ignoreZeros, String opcode, String istr) {
		super(type, op, in1, in2, in3, out, opcode, istr);
		_ignoreZeros = ignoreZeros;
		_outDim1 = outDim1;
		_outDim2 = outDim2;
	}

	public static CtableOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 8);

		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[6]);

		String[] dim1Fields = parts[4].split(Instruction.LITERAL_PREFIX);
		String[] dim2Fields = parts[5].split(Instruction.LITERAL_PREFIX);
		CPOperand outDim1 = new CPOperand(dim1Fields[0], Types.ValueType.FP64, Types.DataType.SCALAR,  Boolean.parseBoolean(dim1Fields[1]));
		CPOperand outDim2 = new CPOperand(dim2Fields[0], Types.ValueType.FP64, Types.DataType.SCALAR,  Boolean.parseBoolean(dim2Fields[1]));

		boolean ignoreZeros = Boolean.parseBoolean(parts[7]);

		// does not require any op
		return new CtableOOCInstruction(OOCType.Ctable, null, in1, in2, in3, out, outDim1, outDim2, ignoreZeros, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {

		MatrixObject in1 = ec.getMatrixObject(input1); // stream
		LocalTaskQueue<IndexedMatrixValue> qIn1 = in1.getStreamHandle();
		IndexedMatrixValue tmp1 = null;

		long outputDim1 = ec.getScalarInput(_outDim1).getLongValue();
		long outputDim2 = ec.getScalarInput(_outDim2).getLongValue();

		long cols = in1.getDataCharacteristics().getNumColBlocks();
		CTableMap map = new CTableMap(LongLongDoubleHashMap.EntryType.INT);

		Ctable.OperationTypes ctableOp = findCtableOperation();
		MatrixObject in2 = null, in3 = null;
		LocalTaskQueue<IndexedMatrixValue> qIn2 = null, qIn3 = null;
		double cst2 = 0, cst3 = 0;

		// init vars based on ctableOp
		if (ctableOp.hasSecondInput()){
			in2 = ec.getMatrixObject(input2); // stream
			qIn2 = in2.getStreamHandle();
		} else
			cst2 = ec.getScalarInput(input2).getDoubleValue();

		if (ctableOp.hasThirdInput()){
			in3 = ec.getMatrixObject(input3); // stream
			qIn3 = in3.getStreamHandle();
		} else
			cst3 = ec.getScalarInput(input3).getDoubleValue();

		HashMap<Long, MatrixBlock> blocksIn2 = new HashMap<>(), blocksIn3 = new HashMap<>();
		MatrixBlock block2, block3;

		// only init result block if output dims known and dense
		MatrixBlock result = null;
		boolean outputDimsKnown = (outputDim1 != -1 && outputDim2 != -1);
		if (outputDimsKnown){
			long totalRows = in1.getDataCharacteristics().getRows();
			long totalCols = in1.getDataCharacteristics().getCols();
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(outputDim1, outputDim2, totalRows*totalCols);
			if(!sparse)
				result = new MatrixBlock((int)outputDim1, (int)outputDim2, false);
		}

		try {
			while((tmp1 = qIn1.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {

				MatrixBlock block1 = (MatrixBlock) tmp1.getValue();
				long r = tmp1.getIndexes().getRowIndex();
				long c = tmp1.getIndexes().getColumnIndex();
				long key = (r-1) * cols + (c-1);

				switch(ctableOp) {
					case CTABLE_TRANSFORM:
						// ctable(A,B,W)
						block2 = getOrDequeueBlock(key, cols, blocksIn2, qIn2);
						block3 = getOrDequeueBlock(key, cols, blocksIn3, qIn3);
						block1.ctableOperations(_optr, block2, block3, map, result);
						break;
					case CTABLE_TRANSFORM_SCALAR_WEIGHT:
						// ctable(A,B) or ctable(A,B,1)
						block2 = getOrDequeueBlock(key, cols, blocksIn2, qIn2);
						block1.ctableOperations(_optr, block2, cst3, _ignoreZeros, map, result);
						break;
					case CTABLE_TRANSFORM_HISTOGRAM:
						// ctable(A,1) or ctable(A,1,1)
						block1.ctableOperations(_optr, cst2, cst3, map, result);
						break;
					case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM:
						// ctable(A,1,W)
						block3 = getOrDequeueBlock(key, cols, blocksIn3, qIn3);
						block1.ctableOperations(_optr, cst2, block3, map, result);
						break;

					default:
						throw new DMLRuntimeException("Encountered an invalid OOC ctable operation ("+ctableOp+") while executing instruction: " +
							this);
				}
			}
			if (result == null){
				if(outputDimsKnown)
					result = DataConverter.convertToMatrixBlock(map, (int)outputDim1, (int)outputDim2);
				else
					result = DataConverter.convertToMatrixBlock(map);
			}
			else
				result.examSparsity();

			ec.setMatrixOutput(output.getName(), result);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private MatrixBlock getOrDequeueBlock(long key, long cols, HashMap<Long, MatrixBlock> blocks, LocalTaskQueue<IndexedMatrixValue> queue)
		throws InterruptedException {
		MatrixBlock block = blocks.get(key);
		if (block == null) {
			IndexedMatrixValue tmp;
			// corresponding block still in queue, dequeue until found
			while ((tmp = queue.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
				block = (MatrixBlock) tmp.getValue();
				long r = tmp.getIndexes().getRowIndex();
				long c = tmp.getIndexes().getColumnIndex();
				long tmpKey = (r-1) * cols + (c-1);
				// found corresponding block
				if (tmpKey == key) break;
				// store all dequeued blocks in cache that we don't need yet
				blocks.put(tmpKey, block);
			}
		}
		else
			blocks.remove(key); // needed only once

		return block;
	}

	private Ctable.OperationTypes findCtableOperation() {
		Types.DataType dt1 = input1.getDataType();
		Types.DataType dt2 = input2.getDataType();
		Types.DataType dt3 = input3.getDataType();
		return Ctable.findCtableOperationByInputDataTypes(dt1, dt2, dt3);
	}

}

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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class MatrixVectorBinaryOOCInstruction extends ComputationOOCInstruction {


	protected MatrixVectorBinaryOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}

	public static MatrixVectorBinaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 4);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]); // the larget matrix (streamed)
		CPOperand in2 = new CPOperand(parts[2]); // vector operand (may be OOC)
		CPOperand out = new CPOperand(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new MatrixVectorBinaryOOCInstruction(OOCType.MAPMM, ba, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {
		// Fetch both inputs without assuming which one fits in memory
		MatrixObject min = ec.getMatrixObject(input1); 
		MatrixObject vin = ec.getMatrixObject(input2);

		// number of colBlocks for early block output
		long emitThreshold = min.getDataCharacteristics().getNumColBlocks();
		OOCMatrixBlockTracker aggTracker = new OOCMatrixBlockTracker(emitThreshold);

		OOCStream<IndexedMatrixValue> qIn1 = min.getStreamHandle();
		OOCStream<IndexedMatrixValue> qIn2 = vin.getStreamHandle(); // Stream handles for matrix and vector (both may be OOC)
		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());
		ec.getMatrixObject(output).setStreamHandle(qOut);

		submitOOCTask(() -> {

				try {
					// Cache vector blocks indexed by their block row id
					// This removes the assumption that the vector is fully in-memory
					HashMap<Long, MatrixBlock> vectorCache = new HashMap<>();

					// Consume the entire vector stream and cache it block-wise
					IndexedMatrixValue vecVal;
					while ((vecVal = qIn2.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
						vectorCache.put(
							vecVal.getIndexes().getRowIndex(),
							(MatrixBlock) vecVal.getValue());
					}
					
					// Stream through matrix blocks and match them with vector blocks
					IndexedMatrixValue tmp = null;
					while((tmp = qIn1.dequeue()) != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixBlock matrixBlock = (MatrixBlock) tmp.getValue();
						long rowIndex = tmp.getIndexes().getRowIndex();
						long colIndex = tmp.getIndexes().getColumnIndex();
						MatrixBlock vectorSlice = vectorCache.get(colIndex);

						// Fail fast if the corresponding vector block is missing
						if (vectorSlice == null)
							throw new DMLRuntimeException("Missing vector block for column block " + colIndex);

						// Now, call the operation with the correct, specific operator.
						MatrixBlock partialResult = matrixBlock.aggregateBinaryOperations(
							matrixBlock, vectorSlice, new MatrixBlock(), (AggregateBinaryOperator) _optr);

						// for single column block, no aggregation neeeded
						if(emitThreshold == 1) {
							qOut.enqueue(new IndexedMatrixValue(new MatrixIndexes(rowIndex, 1), partialResult));
						}
						else {
							// aggregation
							MatrixBlock currAgg = aggTracker.get(rowIndex);
							if (currAgg == null) {
								aggTracker.putAndIncrementCount(rowIndex, partialResult);
							}
							else {
								currAgg = currAgg.binaryOperations(plus, partialResult);
								if (aggTracker.putAndIncrementCount(rowIndex, currAgg)){
									// early block output: emit aggregated block
									MatrixIndexes idx = new MatrixIndexes(rowIndex, 1L);
									qOut.enqueue(new IndexedMatrixValue(idx, currAgg));
									aggTracker.remove(rowIndex);
								}
							}
						}
					}
				}
				catch(Exception ex) {
					throw new DMLRuntimeException(ex);
				}
				finally {
					qOut.closeInput();
				}
		}, qIn1, qIn2, qOut);
	}
}

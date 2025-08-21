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
import java.util.concurrent.ExecutorService;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
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
import org.apache.sysds.runtime.util.CommonThreadPool;

public class MatrixVectorBinaryOOCInstruction extends ComputationOOCInstruction {


	protected MatrixVectorBinaryOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}

	public static MatrixVectorBinaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 4);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]); // the larget matrix (streamed)
		CPOperand in2 = new CPOperand(parts[2]); // the small vector (in-memory)
		CPOperand out = new CPOperand(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new MatrixVectorBinaryOOCInstruction(OOCType.MAPMM, ba, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {
		// 1. Identify the inputs
		MatrixObject min = ec.getMatrixObject(input1); // big matrix
		MatrixBlock vin = ec.getMatrixObject(input2)
			.acquireReadAndRelease(); // in-memory vector

		// 2. Pre-partition the in-memory vector into a hashmap
		HashMap<Long, MatrixBlock> partitionedVector = new HashMap<>();
		int blksize = vin.getDataCharacteristics().getBlocksize();
		if (blksize < 0)
			blksize = ConfigurationManager.getBlocksize();
		for (int i=0; i<vin.getNumRows(); i+=blksize) {
			long key = (long) (i/blksize) + 1; // the key starts at 1
			int end_row = Math.min(i + blksize, vin.getNumRows());
			MatrixBlock vectorSlice = vin.slice(i, end_row - 1);
			partitionedVector.put(key, vectorSlice);
		}

		// number of colBlocks for early block output
		long nBlocks = min.getDataCharacteristics().getNumColBlocks();

		LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());
		ec.getMatrixObject(output).setStreamHandle(qOut);

		ExecutorService pool = CommonThreadPool.get();
		try {
			// Core logic: background thread
			pool.submit(() -> {
				IndexedMatrixValue tmp = null;
				try {
					HashMap<Long, MatrixBlock> partialResults = new  HashMap<>();
					HashMap<Long, Integer> cnt = new HashMap<>();
					while((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixBlock matrixBlock = (MatrixBlock) tmp.getValue();
						long rowIndex = tmp.getIndexes().getRowIndex();
						long colIndex = tmp.getIndexes().getColumnIndex();
						MatrixBlock vectorSlice = partitionedVector.get(colIndex);

						// Now, call the operation with the correct, specific operator.
						MatrixBlock partialResult = matrixBlock.aggregateBinaryOperations(
							matrixBlock, vectorSlice, new MatrixBlock(), (AggregateBinaryOperator) _optr);

						// for single column block, no aggregation neeeded
						if( min.getNumColumns() <= min.getBlocksize() ) {
							qOut.enqueueTask(new IndexedMatrixValue(tmp.getIndexes(), partialResult));
						}
						else {
							// aggregation
							MatrixBlock currAgg = partialResults.get(rowIndex);
							if (currAgg == null) {
								partialResults.put(rowIndex, partialResult);
								cnt.put(rowIndex, 1);
							}
							else {
								currAgg.binaryOperationsInPlace(plus, partialResult);
								int newCnt = cnt.get(rowIndex) + 1;
								
								if(newCnt == nBlocks){
									// early block output: emit aggregated block
									MatrixIndexes idx = new MatrixIndexes(rowIndex, 1L);
									MatrixBlock result = partialResults.get(rowIndex);
									qOut.enqueueTask(new IndexedMatrixValue(idx, result));
									partialResults.remove(rowIndex);
									cnt.remove(rowIndex);
								}
								else {
									// maintain aggregation counts if not output-ready yet
									cnt.replace(rowIndex, newCnt);
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
			});
		} catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			pool.shutdown();
		}
	}
}

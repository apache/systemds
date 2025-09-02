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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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

public class MatrixMultiplyOOCInstruction extends ComputationOOCInstruction {


	protected MatrixMultiplyOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, out, opcode, istr);
	}

	public static MatrixMultiplyOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 4);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]); // the larget matrix (streamed)
		CPOperand in2 = new CPOperand(parts[2]); // the small vector (in-memory)
		CPOperand out = new CPOperand(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new MatrixMultiplyOOCInstruction(OOCType.MAPMM, ba, in1, in2, out, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {

		if (ec.getMatrixObject(input2).getDataCharacteristics().getCols() == 1) {
			_processMatrixVector(ec);
		} else {
			_processMatrixMatrix(ec);
		}
	}

	private void _processMatrixVector( ExecutionContext ec ) {
		// 1. Identify the inputs
		MatrixObject min = ec.getMatrixObject(input1); // big matrix
		MatrixBlock vin = ec.getMatrixObject(input2)
						.acquireReadAndRelease(); // in-memory vector

		// 2. Pre-partition the in-memory vector into a hashmap
		HashMap<Long, MatrixBlock> partitionedVector = new HashMap<>();
		int blksize = vin.getDataCharacteristics().getBlocksize();
		if (blksize < 0)
			blksize = ConfigurationManager.getBlocksize();
		for (int i = 0; i < vin.getNumRows(); i += blksize) {
			long key = (long) (i / blksize) + 1; // the key starts at 1
			int end_row = Math.min(i + blksize, vin.getNumRows());
			MatrixBlock vectorSlice = vin.slice(i, end_row - 1);
			partitionedVector.put(key, vectorSlice);
		}

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
					HashMap<Long, MatrixBlock> partialResults = new HashMap<>();
					while ((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixBlock matrixBlock = (MatrixBlock) tmp.getValue();
						long rowIndex = tmp.getIndexes().getRowIndex();
						long colIndex = tmp.getIndexes().getColumnIndex();
						MatrixBlock vectorSlice = partitionedVector.get(colIndex);

						// Now, call the operation with the correct, specific operator.
						MatrixBlock partialResult = matrixBlock.aggregateBinaryOperations(
										matrixBlock, vectorSlice, new MatrixBlock(), (AggregateBinaryOperator) _optr);

						// for single column block, no aggregation neeeded
						if (min.getNumColumns() <= min.getBlocksize()) {
							qOut.enqueueTask(new IndexedMatrixValue(tmp.getIndexes(), partialResult));
						} else {
							MatrixBlock currAgg = partialResults.get(rowIndex);
							if (currAgg == null)
								partialResults.put(rowIndex, partialResult);
							else
								currAgg.binaryOperationsInPlace(plus, partialResult);
						}
					}

					// emit aggregated blocks
					if (min.getNumColumns() > min.getBlocksize()) {
						for (Map.Entry<Long, MatrixBlock> entry : partialResults.entrySet()) {
							MatrixIndexes outIndexes = new MatrixIndexes(entry.getKey(), 1L);
							qOut.enqueueTask(new IndexedMatrixValue(outIndexes, entry.getValue()));
						}
					}
				} catch (Exception ex) {
					throw new DMLRuntimeException(ex);
				} finally {
					qOut.closeInput();
				}
			});
		} catch (Exception e) {
			throw new DMLRuntimeException(e);
		} finally {
			pool.shutdown();
		}
	}

	private void _processMatrixMatrix( ExecutionContext ec ) {
		// 1. Identify the inputs
		MatrixObject min = ec.getMatrixObject(input1); // big matrix
		MatrixObject min2 = ec.getMatrixObject(input2);

		LocalTaskQueue<IndexedMatrixValue> qIn1 = min.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qIn2 = min2.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
		BinaryOperator plus = InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString());
		ec.getMatrixObject(output).setStreamHandle(qOut);

		// Result matrix rows, cols = rows of A, cols of B
		long resultRowBlocks = min.getDataCharacteristics().getNumRowBlocks();
		long resultColBlocks = min2.getDataCharacteristics().getNumColBlocks();

		ExecutorService pool = CommonThreadPool.get();
		try {
			// Core logic: background thread
			pool.submit(() -> {
				IndexedMatrixValue tmpA = null;
				IndexedMatrixValue tmpB = null;
				try {
					// Phase 1: grouping the output blocks by block Index (The Shuffle)
					Map<MatrixIndexes, List<TaggedMatrixValue>> groupedBlocks = new HashMap<>();
					HashMap<Long, MatrixBlock> partialResults = new  HashMap<>();

					// Process matrix A: each block A(i,k) contributes to C(i,j) for all j
					while((tmpA = qIn1.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						long i = tmpA.getIndexes().getRowIndex() - 1;
						long k = tmpA.getIndexes().getColumnIndex() - 1;

						for (int j=0; j<resultColBlocks; j++) {
							MatrixIndexes index = new MatrixIndexes(i, j); // 1,1= A11,A12,A13,B11,B21,B31

							// Create a copy
							MatrixBlock sourceBlock = (MatrixBlock) tmpA.getValue();
							IndexedMatrixValue valueCopy = new  IndexedMatrixValue(new MatrixIndexes(tmpA.getIndexes()),  sourceBlock);

							TaggedMatrixValue taggedValue = new TaggedMatrixValue(valueCopy, true, k);
							groupedBlocks.computeIfAbsent(index, idx -> new ArrayList<>()).add(taggedValue);
						}
					}

					// Process matrix B: each block B(k,j) contributes to C(i,j) for all i
					while((tmpB = qIn2.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						long k = tmpB.getIndexes().getRowIndex() - 1;
						long j = tmpB.getIndexes().getColumnIndex() - 1;

						for (int i=0; i<resultRowBlocks; i++) {
							MatrixIndexes index = new MatrixIndexes(i, j);

							MatrixBlock sourceBlock = (MatrixBlock) tmpB.getValue();
							IndexedMatrixValue valueCopy = new IndexedMatrixValue(new MatrixIndexes(tmpB.getIndexes()),  sourceBlock);

							TaggedMatrixValue taggedValue = new TaggedMatrixValue(valueCopy, false, k);
							groupedBlocks.computeIfAbsent(index,idx -> new ArrayList<>()).add(taggedValue);
						}
					}


					// Phase 2: Multiplication and Aggregation
					Map<MatrixIndexes, MatrixBlock> resultBlocks = new HashMap<>();

					// Process each output block separately
					for (Map.Entry<MatrixIndexes, List<TaggedMatrixValue>> entry : groupedBlocks.entrySet()) {
						MatrixIndexes outIndex = entry.getKey();
						List<TaggedMatrixValue> outValues = entry.getValue();

						// For this output block, collect left and right input blocks
						Map<Long, MatrixBlock> leftBlocks = new HashMap<>();
						Map<Long, MatrixBlock> rightBlocks = new HashMap<>();

						// Organize blocks by k-index
						for (TaggedMatrixValue taggedValue : outValues) {
							IndexedMatrixValue value = taggedValue.getValue();
							long kIndex = taggedValue.getkIndex();

							if (taggedValue.isFirstInput()) {
								leftBlocks.put(kIndex, (MatrixBlock)value.getValue());
							} else {
								rightBlocks.put(kIndex, (MatrixBlock)value.getValue());
							}
						}

						// Create result block for this (i,j) position
						MatrixBlock resultBlock = null;

						// Find k-indices that exist in both left and right
						Set<Long> commonKIndices = new HashSet<>(leftBlocks.keySet());
						commonKIndices.retainAll(rightBlocks.keySet());

						// Multiply and aggregate matching blocks
						for (Long k : commonKIndices) {
							MatrixBlock leftBlock = leftBlocks.get(k);
							MatrixBlock rightBlock = rightBlocks.get(k);

							// Multiply matching blocks
							MatrixBlock partialResult = leftBlock.aggregateBinaryOperations(leftBlock,
											rightBlock,
											new MatrixBlock(),
											InstructionUtils.getMatMultOperator(1));

							if (resultBlock == null) {
								resultBlock = partialResult;
							} else {
								resultBlock = resultBlock.binaryOperationsInPlace(plus, partialResult);
							}
						}

						// Store the final result for this output block
						if (resultBlock != null) {
							resultBlocks.put(outIndex, resultBlock);
						}
					}

					// Enqueue all results after all multiplications are complete
					for (Map.Entry<MatrixIndexes, MatrixBlock> entry : resultBlocks.entrySet()) {
						MatrixIndexes outIdx0 = entry.getKey();
						MatrixBlock outBlock = entry.getValue();
						MatrixIndexes outIdx = new MatrixIndexes(outIdx0.getRowIndex() + 1,
										outIdx0.getColumnIndex() + 1);
						outBlock.checkSparseRows();
						qOut.enqueueTask(new IndexedMatrixValue(outIdx, outBlock));
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

	/**
	 * Helper class to tag matrix block with their source and k-index
	 */
	private static class TaggedMatrixValue {
		IndexedMatrixValue _value;
		private long _kIndex;
		private boolean _isFirstInput;

		public TaggedMatrixValue(IndexedMatrixValue value, boolean isFirstInput, long kIndex) {
			this._value = value;
			this._isFirstInput = isFirstInput;
			this._kIndex = kIndex;
		}

		public IndexedMatrixValue getValue() {
			return _value;
		}

		public boolean isFirstInput() {
			return _isFirstInput;
		}

		public long getkIndex() {
			return _kIndex;
		}
	}
}

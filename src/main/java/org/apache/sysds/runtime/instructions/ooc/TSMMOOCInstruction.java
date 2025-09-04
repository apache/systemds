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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.lops.MMTSJ;
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

import java.util.HashMap;
import java.util.concurrent.ExecutorService;

public class TSMMOOCInstruction extends ComputationOOCInstruction {
	private final MMTSJ.MMTSJType _type;

	protected TSMMOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand out, MMTSJ.MMTSJType mmtsjType, String opcode, String istr) {
		super(type, op, in1, out, opcode, istr);
		_type = mmtsjType;
	}

	public static TSMMOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]); // the large matrix (streamed), columns <= blocksize
		CPOperand out = new CPOperand(parts[2]);
		MMTSJ.MMTSJType mmtsjType = MMTSJ.MMTSJType.valueOf(parts[3]);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator ba = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg);

		return new TSMMOOCInstruction(OOCType.MMTSJ, ba, in1, out, mmtsjType, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {
		// 1. Identify the inputs
		MatrixObject min = ec.getMatrixObject(input1); // big matrix

		// number of colBlocks for early block output
		int nCols = (int) min.getDataCharacteristics().getCols();

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
					MatrixBlock resultBlock = new MatrixBlock(nCols, nCols, false);
					while((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixBlock matrixBlock = (MatrixBlock) tmp.getValue();

						// Now, call the operation with the correct, specific operator.
						MatrixBlock partialResult = matrixBlock.transposeSelfMatrixMultOperations(new MatrixBlock(), _type);
						resultBlock.binaryOperationsInPlace(plus, partialResult);

					}
					qOut.enqueueTask(new IndexedMatrixValue(new MatrixIndexes(1, 1), resultBlock));
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

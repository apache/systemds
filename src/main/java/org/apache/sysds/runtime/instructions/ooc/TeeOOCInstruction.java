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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.concurrent.ExecutorService;

public class TeeOOCInstruction extends ComputationOOCInstruction {
	private CPOperand output2 = null;

	protected TeeOOCInstruction(OOCType type, CPOperand in1, CPOperand out, CPOperand out2, String opcode, String istr) {
		super(type, null, in1, out, opcode, istr);
		this.output2 = out2;
	}

	public static TeeOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 3);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		CPOperand out2 = new CPOperand(parts[2]);

		return new TeeOOCInstruction(OOCType.Tee, in1, out, out2, opcode, str);
	}

	public void processInstruction( ExecutionContext ec ) {

		// Create thread and process the tee operation
		System.out.println("DEBUG: TeeOOCInstruction.processInstruction()");

		MatrixObject min = ec.getMatrixObject(input1);
		LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();

		if (qIn == null) {
			throw new DMLRuntimeException("Stream handle is null");
		}

		// CHECK STREAM STATE
		System.out.println("=== STREAM DEBUGGING ===");
		System.out.println("  Stream object: " + qIn);

		// Try to peek at stream size/content
		try {
			System.out.println("  Stream size (approx): " + qIn.toString());
		} catch (Exception e) {
			System.out.println("  Cannot get stream size: " + e.getMessage());
		}
		System.out.println("DEBUG: TeeOOCInstruction.processInstruction()");
		System.out.println("  Input1: " + input1.getName());
		System.out.println("  Output1: " + output.getName());
		System.out.println("  Output2: " + output2.getName());

//		MatrixObject min = ec.getMatrixObject(input1);
		System.out.println("  Input matrix object: " + min);
		System.out.println("  Matrix has stream handle: " + (min.getStreamHandle() != null));

		if (min.getStreamHandle() == null) {
			System.out.println("  Matrix is materialized: " + min.isCached(false));
			System.out.println("  Matrix metadata: " + min.getMetaData());
		}

//		LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();

		if (qIn == null) {
			throw new DMLRuntimeException("Stream handle is null for input: " + input1.getName() +
					". This suggests the input stream was not properly created or was already consumed.");
		}
//		MatrixObject min = ec.getMatrixObject(input1);
//		LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();
		LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		System.out.println("We are reaching here");


		ExecutorService pool = CommonThreadPool.get();
//		Thread.dumpStack();
		try {
			pool.submit(() -> {
				IndexedMatrixValue tmp = null;
				try {
					while ((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
						System.out.println("print tmp:");
						System.out.println(tmp);
						IndexedMatrixValue tmpOut = new IndexedMatrixValue();
//						tmpOut.set(tmp.getIndexes(),
//								tmp.getValue().unaryOperations(uop, new MatrixBlock()));
						qOut.enqueueTask(tmpOut);
					}
					qOut.closeInput();
				}
				catch(Exception ex) {
					throw new DMLRuntimeException(ex);
				}
			});
		} catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		} finally {
			pool.shutdown();
		}
	}
}

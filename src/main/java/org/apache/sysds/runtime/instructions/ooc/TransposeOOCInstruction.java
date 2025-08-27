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
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.concurrent.ExecutorService;

public class TransposeOOCInstruction extends ComputationOOCInstruction {

  protected TransposeOOCInstruction(OOCType type, ReorgOperator op, CPOperand in1, CPOperand out, String opcode, String istr) {
    super(type, op, in1, out, opcode, istr);

  }

  public static TransposeOOCInstruction parseInstruction(String str) {
    String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
    InstructionUtils.checkNumFields(parts, 2);
    String opcode = parts[0];
    CPOperand in1 = new CPOperand(parts[1]);
    CPOperand out = new CPOperand(parts[2]);

    ReorgOperator reorg = new ReorgOperator(SwapIndex.getSwapIndexFnObject());
    return new TransposeOOCInstruction(OOCType.Reorg, reorg, in1, out, opcode, str);
  }

  public void processInstruction( ExecutionContext ec ) {

    // Create thread and process the transpose operation
    MatrixObject min = ec.getMatrixObject(input1);
    LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();
    LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
    ec.getMatrixObject(output).setStreamHandle(qOut);


    ExecutorService pool = CommonThreadPool.get();
    try {
      pool.submit(() -> {
        IndexedMatrixValue tmp = null;
        try {
          while ((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
            MatrixBlock inBlock = (MatrixBlock)tmp.getValue();
            long oldRowIdx = tmp.getIndexes().getRowIndex();
            long oldColIdx = tmp.getIndexes().getColumnIndex();

            MatrixBlock outBlock = inBlock.reorgOperations((ReorgOperator) _optr, new MatrixBlock(), -1, -1, -1);
            qOut.enqueueTask(new IndexedMatrixValue(new MatrixIndexes(oldColIdx, oldRowIdx), outBlock));
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

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

import java.util.concurrent.ExecutorService;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.*;
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

        return new MatrixVectorBinaryOOCInstruction(OOCType.AggregateBinary, ba, in1, in2, out, opcode, str);
    }

    @Override
    public void processInstruction( ExecutionContext ec ) {

        //get operator and scalar
        CPOperand scalar = ( input1.getDataType() == DataType.MATRIX ) ? input2 : input1;
        ScalarObject constant = ec.getScalarInput(scalar);
        ScalarOperator sc_op = ((ScalarOperator)_optr).setConstant(constant.getDoubleValue());

        //create thread and process binary operation
        MatrixObject min = ec.getMatrixObject(input1);
        MatrixObject vin = ec.getMatrixObject(input2);
        LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();
        LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
        ec.getMatrixObject(output).setStreamHandle(qOut);

        ExecutorService pool = CommonThreadPool.get();
        try {
            pool.submit(() -> {
                IndexedMatrixValue tmp = null;
                try {
                    while((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
                        IndexedMatrixValue tmpOut = new IndexedMatrixValue();
                        tmpOut.set(tmp.getIndexes(),
                                tmp.getValue().scalarOperations(sc_op, new MatrixBlock()));
                        qOut.enqueueTask(tmpOut);
                    }
                    qOut.closeInput();
                }
                catch(Exception ex) {
                    throw new DMLRuntimeException(ex);
                }
            });
        }
        finally {
            pool.shutdown();
        }
    }
}

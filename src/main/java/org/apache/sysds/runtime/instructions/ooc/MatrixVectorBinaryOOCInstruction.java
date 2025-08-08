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
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.conf.ConfigurationManager;
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
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
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

        // TODO: update the OOCType to specialized operation matrix multiplication
        return new MatrixVectorBinaryOOCInstruction(OOCType.MAPMM, ba, in1, in2, out, opcode, str);
    }

    @Override
    public void processInstruction( ExecutionContext ec ) {

        // Setup: main thread
        // 1. Identify the inputs
        MatrixObject min = ec.getMatrixObject(input1); // big matrix
        MatrixObject vin = ec.getMatrixObject(input2); // in-memory vector

        // 2. Load the vector in-memory
        MatrixBlock vector = ec.getMatrixInput(input2.getName());
        System.out.println("vector: ");
        System.out.println(vector);

        // 3. Pre-partition the in-memory vector into a hashmap
        HashMap<Long, MatrixBlock> partitionedVector = new HashMap<>();
        int blksize = vector.getDataCharacteristics().getBlocksize();
        if (blksize < 0) {
            blksize = ConfigurationManager.getBlocksize();
            System.out.println("ConfigurationManager.getBlocksize(): "+blksize);
        }
        for (int i=0; i<vector.getNumRows(); i+=blksize) {
            long key = (long) (i/blksize) + 1; // the key starts at 1
//            System.out.println("i + blksize + 1: " + i + blksize + 1 );
//            System.out.println("vector.getNumRows() - 1: " + vector.getNumRows() + 1);
//            MatrixBlock vectorSlice = vector.slice(i, Math.min(i + blksize - 1, vector.getNumRows() - 1));
//            System.out.println("vectorSlices: ");
//            System.out.println(vectorSlice);
//            partitionedVector.put(key, vectorSlice);

            int end_row = Math.min(i + blksize, vector.getNumRows());
            int chunk_len = end_row - i;
            MatrixBlock vectorChunk = new MatrixBlock(chunk_len, 1, false);
            vector.copy(i, end_row - 1, 0, 0, vectorChunk, true);
            vectorChunk.recomputeNonZeros();

            partitionedVector.put(key, vectorChunk);
        }
        System.out.println("partitionedVector: \n" + partitionedVector);

        LocalTaskQueue<IndexedMatrixValue> qIn = min.getStreamHandle();

        LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
        ec.getMatrixObject(output).setStreamHandle(qOut);

        ExecutorService pool = CommonThreadPool.get();
        try {
            // Core logic: background thread
            Future<?> task = pool.submit(() -> {
                IndexedMatrixValue tmp = null;
                try {

                    HashMap<Long, MatrixBlock> partialResults = new  HashMap<>();

                    while((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
//                        IndexedMatrixValue tmpOut = new IndexedMatrixValue();
                        MatrixBlock matrixBlock = (MatrixBlock) tmp.getValue();
                        long rowIndex = tmp.getIndexes().getRowIndex();
                        long colIndex = tmp.getIndexes().getColumnIndex();

                        MatrixBlock vectorSlice = partitionedVector.get(colIndex);
                        System.out.println("vectorSlice: " + vectorSlice);
//                        System.out.println("tmp: \n" + tmp);
                        MatrixBlock partialResult = matrixBlock.aggregateBinaryOperations(matrixBlock, vectorSlice,
                                new MatrixBlock(), (AggregateBinaryOperator) _optr);
                        System.out.println("partialResult: " + partialResult);

                        MatrixBlock currAgg = partialResults.get(rowIndex);
                        if (currAgg == null) {
                            partialResults.put(rowIndex, partialResult);
                        } else {
                            currAgg.binaryOperationsInPlace(InstructionUtils.parseBinaryOperator(Opcodes.PLUS.toString()), partialResult);
                        }

                    }
                    for (Map.Entry<Long, MatrixBlock> entry : partialResults.entrySet()) {
                        MatrixIndexes outIndexes = new MatrixIndexes(entry.getKey(), 1L);
                        qOut.enqueueTask(new IndexedMatrixValue(outIndexes, entry.getValue()));
                    }
                }
                catch(Exception ex) {
                    throw new DMLRuntimeException(ex);
                } finally {
                    qOut.closeInput();
                }
            });

            task.get();
        } catch (ExecutionException | InterruptedException e) {
            throw new DMLRuntimeException(e);
        }
        finally {
//            pool.shutdown();
            ec.releaseMatrixInput(input1.getName());
        }
    }
}

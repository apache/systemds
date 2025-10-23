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

import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.HashMap;
import java.util.concurrent.ExecutorService;

public class AggregateUnaryOOCInstruction extends ComputationOOCInstruction {
	private AggregateOperator _aop = null;

	protected AggregateUnaryOOCInstruction(OOCType type, AggregateUnaryOperator auop, AggregateOperator aop, 
			CPOperand in, CPOperand out, String opcode, String istr) {
		super(type, auop, in, out, opcode, istr);
		_aop = aop;
	}

	protected AggregateUnaryOOCInstruction(OOCType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3,
		CPOperand out, String opcode, String istr) {
		super(type, op, in1, in2, in3, out, opcode, istr);
		_aop = null;
	}

	public static AggregateUnaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 2);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		
		String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(opcode);
		CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(opcode);
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrLoc.toString());
		return new AggregateUnaryOOCInstruction(
			OOCType.AggregateUnary, aggun, aop, in1, out, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		//TODO support all types of aggregations, currently only full aggregation, row aggregation and column aggregation
		
		//setup operators and input queue
		AggregateUnaryOperator aggun = (AggregateUnaryOperator) getOperator(); 
		MatrixObject min = ec.getMatrixObject(input1);
		LocalTaskQueue<IndexedMatrixValue> q = min.getStreamHandle();
		int blen = ConfigurationManager.getBlocksize();

		if (aggun.isRowAggregate() || aggun.isColAggregate()) {
			DataCharacteristics chars = ec.getDataCharacteristics(input1.getName());
			// number of blocks to process per aggregation idx (row or column dim)
			long emitThreshold = aggun.isRowAggregate()? chars.getNumColBlocks() : chars.getNumRowBlocks();
			OOCMatrixBlockTracker aggTracker = new OOCMatrixBlockTracker(emitThreshold);
			HashMap<Long, MatrixBlock> corrs = new HashMap<>(); // correction blocks

			LocalTaskQueue<IndexedMatrixValue> qOut = new LocalTaskQueue<>();
			ec.getMatrixObject(output).setStreamHandle(qOut);
			ExecutorService pool = CommonThreadPool.get();
			try {
				pool.submit(() -> {
					IndexedMatrixValue tmp = null;
					try {
						while((tmp = q.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
							long idx  = aggun.isRowAggregate() ?
								tmp.getIndexes().getRowIndex() : tmp.getIndexes().getColumnIndex();
							MatrixBlock ret = aggTracker.get(idx);
							if(ret != null) {
								MatrixBlock corr = corrs.get(idx);

								// aggregation
								MatrixBlock ltmp = (MatrixBlock) ((MatrixBlock) tmp.getValue())
									.aggregateUnaryOperations(aggun, new MatrixBlock(), blen, tmp.getIndexes());
								OperationsOnMatrixValues.incrementalAggregation(ret,
									_aop.existsCorrection() ? corr : null, ltmp, _aop, true);

								if (!aggTracker.putAndIncrementCount(idx, ret)){
									corrs.replace(idx, corr);
									continue;
								}
							}
							else {
								// first block for this idx - init aggregate and correction
								// TODO avoid corr block for inplace incremental aggregation
								int rows = tmp.getValue().getNumRows();
								int cols = tmp.getValue().getNumColumns();
								int extra = _aop.correction.getNumRemovedRowsColumns();
								ret = aggun.isRowAggregate()? new MatrixBlock(rows, 1 + extra, false) : new MatrixBlock(1 + extra, cols, false);
								MatrixBlock corr = aggun.isRowAggregate()? new MatrixBlock(rows, 1 + extra, false) : new MatrixBlock(1 + extra, cols, false);

								// aggregation
								MatrixBlock ltmp = (MatrixBlock) ((MatrixBlock) tmp.getValue()).aggregateUnaryOperations(
									aggun, new MatrixBlock(), blen, tmp.getIndexes());
								OperationsOnMatrixValues.incrementalAggregation(ret,
									_aop.existsCorrection() ? corr : null, ltmp, _aop, true);

								if(emitThreshold > 1){
									aggTracker.putAndIncrementCount(idx, ret);
									corrs.put(idx, corr);
									continue;
								}
							}

							// all input blocks for this idx processed - emit aggregated block
							ret.dropLastRowsOrColumns(_aop.correction);
							MatrixIndexes midx = aggun.isRowAggregate() ?
								new MatrixIndexes(tmp.getIndexes().getRowIndex(), 1) :
								new MatrixIndexes(1, tmp.getIndexes().getColumnIndex());
							IndexedMatrixValue tmpOut = new IndexedMatrixValue(midx, ret);

							qOut.enqueueTask(tmpOut);
							// drop intermediate states
							aggTracker.remove(idx);
							corrs.remove(idx);
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
		// full aggregation
		else {
			IndexedMatrixValue tmp = null;
			//read blocks and aggregate immediately into result
			int extra = _aop.correction.getNumRemovedRowsColumns();
			MatrixBlock ret = new MatrixBlock(1,1+extra,false);
			MatrixBlock corr = new MatrixBlock(1,1+extra,false);
			try {
				while((tmp = q.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
					//block aggregation
					MatrixBlock ltmp = (MatrixBlock) ((MatrixBlock) tmp.getValue())
						.aggregateUnaryOperations(aggun, new MatrixBlock(), blen, tmp.getIndexes());
					//accumulation into final result
					OperationsOnMatrixValues.incrementalAggregation(
						ret, _aop.existsCorrection() ? corr : null, ltmp, _aop, true);
				}
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}

			//create scalar output
			ec.setScalarOutput(output.getName(), new DoubleObject(ret.get(0, 0)));
		}
	}
}

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
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.matrix.operators.CMOperator;

import java.util.*;

public class CentralMomentOOCInstruction extends AggregateUnaryOOCInstruction {

	private CentralMomentOOCInstruction(CMOperator cm, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String str) {
		super(OOCType.CM, cm, in1, in2, in3, out, opcode, str);
	}

	public static CentralMomentOOCInstruction parseInstruction(String str) {
		CentralMomentCPInstruction cpInst = CentralMomentCPInstruction.parseInstruction(str);
		return parseInstruction(cpInst);
	}

	public static CentralMomentOOCInstruction parseInstruction(CentralMomentCPInstruction inst) {
		return new CentralMomentOOCInstruction((CMOperator) inst.getOperator(), inst.input1, inst.input2, inst.input3,
			inst.output, inst.getOpcode(), inst.getInstructionString());
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String output_name = output.getName();

		/*
		 * The "order" of the central moment in the instruction can
		 * be set to INVALID when the exact value is unknown at
		 * compilation time. We first need to determine the exact
		 * order and update the CMOperator, if needed.
		 */

		MatrixObject matObj = ec.getMatrixObject(input1.getName());
		LocalTaskQueue<IndexedMatrixValue> qIn = matObj.getStreamHandle();

		CPOperand scalarInput = (input3 == null ? input2 : input3);
		ScalarObject order = ec.getScalarInput(scalarInput);

		CMOperator cm_op = ((CMOperator) _optr);
		if(cm_op.getAggOpType() == CMOperator.AggregateOperationTypes.INVALID)
			cm_op = cm_op.setCMAggOp((int) order.getLongValue());

		CMOperator finalCm_op = cm_op;

		List<CM_COV_Object> cmObjs = new ArrayList<>();

		if(input3 == null) {
			try {
				IndexedMatrixValue tmp = null;

				while((tmp = qIn.dequeueTask()) != LocalTaskQueue.NO_MORE_TASKS) {
					// We only handle MatrixBlock, other types of MatrixValue will fail here
					cmObjs.add(((MatrixBlock) tmp.getValue()).cmOperations(cm_op));
				}
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		}
		else {
			// Here we use a hash join approach
			// Note that this may keep blocks in the cache for a while, depending on when a matching block arrives in the stream
			MatrixObject wtObj = ec.getMatrixObject(input2.getName());
			LocalTaskQueue<IndexedMatrixValue> wIn = wtObj.getStreamHandle();

			try {
				IndexedMatrixValue tmp = qIn.dequeueTask();
				IndexedMatrixValue tmpW = wIn.dequeueTask();
				Map<MatrixIndexes, MatrixValue> left = new HashMap<>();
				Map<MatrixIndexes, MatrixValue> right = new HashMap<>();

				boolean cont = tmp != LocalTaskQueue.NO_MORE_TASKS || tmpW != LocalTaskQueue.NO_MORE_TASKS;

				while(cont) {
					cont = false;

					if(tmp != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixValue weights = right.remove(tmp.getIndexes());

						if(weights != null)
							cmObjs.add(((MatrixBlock) tmp.getValue()).cmOperations(cm_op, (MatrixBlock) weights));
						else
							left.put(tmp.getIndexes(), tmp.getValue());

						tmp = qIn.dequeueTask();
						cont = tmp != LocalTaskQueue.NO_MORE_TASKS;
					}

					if(tmpW != LocalTaskQueue.NO_MORE_TASKS) {
						MatrixValue q = left.remove(tmpW.getIndexes());

						if(q != null)
							cmObjs.add(((MatrixBlock) q).cmOperations(cm_op, (MatrixBlock) tmpW.getValue()));
						else
							right.put(tmpW.getIndexes(), tmpW.getValue());

						tmpW = wIn.dequeueTask();
						cont |= tmpW != LocalTaskQueue.NO_MORE_TASKS;
					}
				}

				if (!left.isEmpty() || !right.isEmpty())
					throw new DMLRuntimeException("Unmatched blocks: values=" + left.size() + ", weights=" + right.size());
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		}

		Optional<CM_COV_Object> res = cmObjs.stream()
			.reduce((arg0, arg1) -> (CM_COV_Object) finalCm_op.fn.execute(arg0, arg1));

		try {
			ec.setScalarOutput(output_name, new DoubleObject(res.get().getRequiredResult(finalCm_op)));
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}

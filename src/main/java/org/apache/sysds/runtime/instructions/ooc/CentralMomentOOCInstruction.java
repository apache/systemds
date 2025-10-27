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
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CentralMomentCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;

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
		OOCStream<IndexedMatrixValue> qIn = matObj.getStreamHandle();

		CPOperand scalarInput = (input3 == null ? input2 : input3);
		ScalarObject order = ec.getScalarInput(scalarInput);

		CMOperator cm_op = ((CMOperator) _optr);
		if(cm_op.getAggOpType() == CMOperator.AggregateOperationTypes.INVALID)
			cm_op = cm_op.setCMAggOp((int) order.getLongValue());

		CMOperator finalCm_op = cm_op;

		OOCStream<CM_COV_Object> cmObjs = createWritableStream();

		if(input3 == null) {
			mapOOC(qIn, cmObjs, tmp -> ((MatrixBlock) tmp.getValue()).cmOperations(new CMOperator(finalCm_op))); // Need to copy CMOperator as its ValueFunction is stateful
		}
		else {
			// Here we use a hash join approach
			// Note that this may keep blocks in the cache for a while, depending on when a matching block arrives in the stream
			MatrixObject wtObj = ec.getMatrixObject(input2.getName());

			DataCharacteristics dc = ec.getDataCharacteristics(input1.getName());
			DataCharacteristics dcW = ec.getDataCharacteristics(input2.getName());

			if (dc.getBlocksize() != dcW.getBlocksize())
				throw new DMLRuntimeException("Different block sizes are not yet supported");

			OOCStream<IndexedMatrixValue> wIn = wtObj.getStreamHandle();

			joinOOC(qIn, wIn, cmObjs,
				(tmp, weights) ->
					((MatrixBlock) tmp.getValue()).cmOperations(new CMOperator(finalCm_op), (MatrixBlock) weights.getValue()),
				IndexedMatrixValue::getIndexes);
		}

		try {
			CM_COV_Object agg = cmObjs.dequeue();
			CM_COV_Object next;

			while ((next = cmObjs.dequeue()) != LocalTaskQueue.NO_MORE_TASKS)
				agg = (CM_COV_Object) finalCm_op.fn.execute(agg, next);

			ec.setScalarOutput(output_name, new DoubleObject(agg.getRequiredResult(finalCm_op)));
		} catch (Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}

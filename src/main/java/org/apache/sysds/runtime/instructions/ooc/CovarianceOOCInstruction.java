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

import java.util.List;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.LocalTaskQueue;
import org.apache.sysds.runtime.functionobjects.COV;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.COVOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class CovarianceOOCInstruction extends ComputationOOCInstruction {

	private CovarianceOOCInstruction(COVOperator cov, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
										String opcode, String str) {
		super(OOCType.COV, cov, in1, in2, in3, out, opcode, str);
	}

	public static CovarianceOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(!opcode.equalsIgnoreCase(Opcodes.COV.toString()))
			throw new DMLRuntimeException("CovarianceOOCInstruction.parseInstruction():: Unknown opcode " + opcode);

		// the OOC instruction string matches the Spark format,

		COVOperator cov = new COVOperator(COV.getCOMFnObject());
		if(parts.length == 4) { // this is the case for unweighted cov.A.B.out
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			return new CovarianceOOCInstruction(cov, in1, in2, null, out, opcode, str);
		}
		else if(parts.length == 5) {// this is the case for weighted cov.A.B.W.out
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			return new CovarianceOOCInstruction(cov, in1, in2, in3, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid number of arguments in Instruction: " + str);
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		COVOperator cov_op = (COVOperator) _optr;

		MatrixObject mo1 = ec.getMatrixObject(input1.getName());
		MatrixObject mo2 = ec.getMatrixObject(input2.getName());

		OOCStream<IndexedMatrixValue> q1 = mo1.getStreamHandle();
		OOCStream<IndexedMatrixValue> q2 = mo2.getStreamHandle();

		OOCStream<CmCovObject> covObjs = createWritableStream();

		if(input3 == null) {
			// unweighted covariance join the two tile streams by block index
			joinOOC(q1, q2, covObjs,
					(a, b) -> ((MatrixBlock) a.getValue()).covOperations(cov_op, (MatrixBlock) b.getValue()),
					IndexedMatrixValue::getIndexes);
		}
		else {
			// weighted covariance additionally join the weights tile stream
			MatrixObject mo3 = ec.getMatrixObject(input3.getName());

			DataCharacteristics dc1 = ec.getDataCharacteristics(input1.getName());
			DataCharacteristics dc2 = ec.getDataCharacteristics(input2.getName());
			DataCharacteristics dc3 = ec.getDataCharacteristics(input3.getName());
			if(dc1.getBlocksize() != dc2.getBlocksize() || dc1.getBlocksize() != dc3.getBlocksize())
				throw new DMLRuntimeException("Different block sizes are not yet supported");

			OOCStream<IndexedMatrixValue> q3 = mo3.getStreamHandle();

			joinOOC(List.of(q1, q2, q3), covObjs,
					tiles -> ((MatrixBlock) tiles.get(0).getValue()).covOperations(cov_op,
							(MatrixBlock) tiles.get(1).getValue(), (MatrixBlock) tiles.get(2).getValue()),
					IndexedMatrixValue::getIndexes);
		}

		try {
			CmCovObject agg = covObjs.dequeue();
			CmCovObject next;

			while((next = covObjs.dequeue()) != LocalTaskQueue.NO_MORE_TASKS)
				agg = (CmCovObject) cov_op.fn.execute(agg, next);

			ec.setScalarOutput(output.getName(), new DoubleObject(agg.getRequiredResult(cov_op)));
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}

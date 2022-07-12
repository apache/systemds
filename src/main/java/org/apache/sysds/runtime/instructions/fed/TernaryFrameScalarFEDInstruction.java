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

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.TernaryFrameScalarCPInstruction;
import org.apache.sysds.runtime.instructions.spark.TernaryFrameScalarSPInstruction;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;

public class TernaryFrameScalarFEDInstruction extends TernaryFEDInstruction
{
	protected TernaryFrameScalarFEDInstruction(TernaryOperator op, CPOperand in1,
			CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr, FederatedOutput fedOut) {
		super(op, in1, in2, in3, out, opcode, istr, fedOut);
	}

	public static TernaryFrameScalarFEDInstruction parseInstruction(TernaryFrameScalarCPInstruction instr) {
		return new TernaryFrameScalarFEDInstruction((TernaryOperator) instr.getOperator(), instr.input1, instr.input2,
			instr.input3, instr.output, instr.getOpcode(), instr.getInstructionString(),
			FEDInstruction.FederatedOutput.NONE);
	}

	public static TernaryFrameScalarFEDInstruction parseInstruction(TernaryFrameScalarSPInstruction instr) {
		return new TernaryFrameScalarFEDInstruction((TernaryOperator) instr.getOperator(), instr.input1, instr.input2,
			instr.input3, instr.output, instr.getOpcode(), instr.getInstructionString(),
			FEDInstruction.FederatedOutput.NONE);
	}

	@Override
	public void processInstruction(ExecutionContext ec)  {
		// get input frames
		FrameObject fo = ec.getFrameObject(input1);
		FederationMap fedMap = fo.getFedMapping();

		//compute results
		FederatedRequest fr1 = FederationUtils.callInstruction(instString,
			output, new CPOperand[] {input1}, new long[] {fedMap.getID()});
		fedMap.execute(getTID(), true, fr1);

		FrameObject out = ec.getFrameObject(output);
		out.setSchema(fo.getSchema());
		out.getDataCharacteristics().set(fo.getDataCharacteristics());
		out.setFedMapping(fedMap.copyWithNewID(fr1.getID()));
	}
}

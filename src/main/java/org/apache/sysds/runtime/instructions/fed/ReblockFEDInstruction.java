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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.ReblockSPInstruction;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

public class ReblockFEDInstruction extends UnaryFEDInstruction {
	private int blen;

	private ReblockFEDInstruction(Operator op, CPOperand in, CPOperand out, int blen, boolean emptyBlocks,
		String opcode, String instr) {
		super(FEDInstruction.FEDType.Reblock, op, in, out, opcode, instr);
		this.blen = blen;
	}

	public static ReblockFEDInstruction parseInstruction(ReblockSPInstruction instr) {
		return new ReblockFEDInstruction(instr.getOperator(), instr.input1, instr.output, instr.getBlockLength(),
			instr.getOutputEmptyBlocks(), instr.getOpcode(), instr.getInstructionString());
	}

	public static ReblockFEDInstruction parseInstruction(String str) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if(!opcode.equals("rblk")) {
			throw new DMLRuntimeException("Incorrect opcode for ReblockFEDInstruction:" + opcode);
		}

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen=Integer.parseInt(parts[3]);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[4]);

		Operator op = null; // no operator for ReblockFEDInstruction
		return new ReblockFEDInstruction(op, in, out, blen, outputEmptyBlocks, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//set the output characteristics
		CacheableData<?> obj = ec.getCacheableData(input1.getName());
		DataCharacteristics mc = ec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = ec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), blen, mc.getNonZeros());

		//get the source format from the meta data
		MetaDataFormat iimd = (MetaDataFormat) obj.getMetaData();
		if(iimd == null)
			throw new DMLRuntimeException("Error ReblockFEDInstruction: Metadata not found");

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest[] fr1 = new FederatedRequest[obj.getFedMapping().getSize()];
		int i = 0;
		for(FederatedRange range : obj.getFedMapping().getFederatedRanges()) {
			fr1[i] = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id,
				new MatrixCharacteristics(range.getSize(0), range.getSize(1)), obj.getDataType());
			i++;
		}
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, id,
			new CPOperand[]{input1}, new long[]{ obj.getFedMapping().getID()}, Types.ExecType.SPARK, false);

		//execute federated operations and set output
		obj.getFedMapping().execute(getTID(), true, fr1, fr2);
		CacheableData<?> out = ec.getCacheableData(output);
		out.setFedMapping(obj.getFedMapping().copyWithNewID(fr2.getID()));
		out.getDataCharacteristics().set(mcOut);
	}
}

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

import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

public class ReblockFEDInstruction extends UnaryFEDInstruction {
	private int blen;
	private boolean outputEmptyBlocks;

	private ReblockFEDInstruction(Operator op, CPOperand in, CPOperand out, int br, int bc, boolean emptyBlocks,
		String opcode, String instr) {
		super(FEDInstruction.FEDType.Reblock, op, in, out, opcode, instr);
		blen = br;
		blen = bc;
		outputEmptyBlocks = emptyBlocks;
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
		return new ReblockFEDInstruction(op, in, out, blen, blen, outputEmptyBlocks, opcode, str);
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

		String modifiedInstString = InstructionUtils.concatOperands(instString, Boolean.toString(true));
		FederatedRequest fr = FederationUtils.callInstruction(modifiedInstString, output,
			new CPOperand[]{input1}, new long[]{ obj.getFedMapping().getID()});

		//execute federated operations and set output
		Future<FederatedResponse>[] tmp = obj.getFedMapping().execute(getTID(), true, fr);
		CacheableData<?> out = ec.getMatrixObject(output);
		out.setFedMapping(obj.getFedMapping().copyWithNewID(fr.getID()));
		out.getDataCharacteristics().set(mcOut);
	}
}

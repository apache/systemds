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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;

public class ReorgFEDInstruction extends UnaryFEDInstruction {
	
	public ReorgFEDInstruction(CPOperand in1, CPOperand out, String opcode, String istr) {
		super(FEDType.Reorg, null, in1, out, opcode, istr);
	}

	public static ReorgFEDInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if ( opcode.equalsIgnoreCase("r'") ) {
			InstructionUtils.checkNumFields(str, 2, 3);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[2]);
			return new ReorgFEDInstruction(in, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("ReorgFEDInstruction: unsupported opcode: "+opcode);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		
		if( !mo1.isFederated() )
			throw new DMLRuntimeException("Federated Reorg: "
				+ "Federated input expected, but invoked w/ "+mo1.isFederated());
	
		//execute transpose at federated site
		FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
			new CPOperand[]{input1}, new long[]{mo1.getFedMapping().getID()});
		mo1.getFedMapping().execute(getTID(), true, fr1);
		
		//drive output federated mapping
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getNumColumns(),
			mo1.getNumRows(), (int)mo1.getBlocksize(), mo1.getNnz());
		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr1.getID()).transpose());
	}
}

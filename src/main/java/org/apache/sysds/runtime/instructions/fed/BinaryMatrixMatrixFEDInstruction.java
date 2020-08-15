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
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class BinaryMatrixMatrixFEDInstruction extends BinaryFEDInstruction
{
	protected BinaryMatrixMatrixFEDInstruction(Operator op,
		CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(FEDType.Binary, op, in1, in2, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo1 = ec.getMatrixObject(input1);
		MatrixObject mo2 = ec.getMatrixObject(input2);
		
		if( mo2.isFederated() ) {
			throw new DMLRuntimeException("Matrix-matrix binary operations "
				+ " with a federated right input are not supported yet.");
		}
		
		//matrix-matrix binary operations -> lhs fed input -> fed output
		FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
			new CPOperand[]{input1, input2}, new long[]{mo1.getFedMapping().getID(), fr1.getID()});
		
		//execute federated instruction and cleanup intermediates
		mo1.getFedMapping().execute(fr1, fr2);
		mo1.getFedMapping().cleanup(fr1.getID());
		
		//derive new fed mapping for output
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getDataCharacteristics());
		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
	}
}

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

import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

public class AggregateUnaryFEDInstruction extends UnaryFEDInstruction {
	
	private AggregateUnaryFEDInstruction(AggregateUnaryOperator auop, CPOperand in,
			CPOperand out, String opcode, String istr) {
		super(FEDType.AggregateUnary, auop, in, out, opcode, istr);
	}
	
	public static AggregateUnaryFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		AggregateUnaryOperator aggun = InstructionUtils.parseBasicAggregateUnaryOperator(opcode);
		if(InstructionUtils.getExecType(str) == ExecType.SPARK)
			str = InstructionUtils.replaceOperand(str, 4, "-1");
		return new AggregateUnaryFEDInstruction(aggun, in1, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		AggregateUnaryOperator aop = (AggregateUnaryOperator) _optr;
		MatrixObject in = ec.getMatrixObject(input1);
		FederationMap map = in.getFedMapping();
		
		//create federated commands for aggregation
		FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
			new CPOperand[]{input1}, new long[]{in.getFedMapping().getID()});
		FederatedRequest fr2 = new FederatedRequest(RequestType.GET_VAR, fr1.getID());
		FederatedRequest fr3 = map.cleanup(getTID(), fr1.getID());
		
		//execute federated commands and cleanups
		Future<FederatedResponse>[] tmp = map.execute(getTID(), fr1, fr2, fr3);
		if( output.isScalar() )
			ec.setVariable(output.getName(), FederationUtils.aggScalar(aop, tmp));
		else
			ec.setMatrixOutput(output.getName(), FederationUtils.aggMatrix(aop, tmp, map));
	}
}

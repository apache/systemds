/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.	See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.fed;

import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;

import java.util.concurrent.Future;

public class QuaternaryWCeMMFEDInstruction extends QuaternaryFEDInstruction
{
	// input1 ... federated X
	// input2 ... U
	// input3 ... V
	// _input4 ... W (=epsilon)
	protected QuaternaryWCeMMFEDInstruction(Operator operator,
		CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4,
		CPOperand out, String opcode, String instruction_str)
	{
		super(FEDType.Quaternary, operator, in1, in2, in3, in4, out, opcode, instruction_str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
	{
		QuaternaryOperator qop = (QuaternaryOperator) _optr;
		MatrixObject X = ec.getMatrixObject(input1);
		MatrixObject U = ec.getMatrixObject(input2);
		MatrixObject V = ec.getMatrixObject(input3);
		ScalarObject eps = null;
		
		if(qop.hasFourInputs()) {
			eps = (_input4.getDataType() == DataType.SCALAR) ?
				ec.getScalarInput(_input4) :
				new DoubleObject(ec.getMatrixInput(_input4.getName()).quickGetValue(0, 0));
		}

		if(!(X.isFederated() && !U.isFederated() && !V.isFederated()))
			throw new DMLRuntimeException("Unsupported federated inputs (X, U, V) = ("
				+X.isFederated()+", "+U.isFederated()+", "+V.isFederated()+")");
		
		FederationMap fedMap = X.getFedMapping();
		FederatedRequest[] fr1 = fedMap.broadcastSliced(U, false);
		FederatedRequest fr2 = fedMap.broadcast(V);
		FederatedRequest fr3 = null;
		FederatedRequest frComp = null;

		// broadcast the scalar epsilon if there are four inputs
		if(eps != null) {
			fr3 = fedMap.broadcast(eps);
			// change the is_literal flag from true to false because when broadcasted it is no literal anymore
			instString = instString.replace("true", "false");
			frComp = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2, input3, _input4},
				new long[]{fedMap.getID(), fr1[0].getID(), fr2.getID(), fr3.getID()});
		}
		else {
			frComp = FederationUtils.callInstruction(instString, output,
			new CPOperand[]{input1, input2, input3},
			new long[]{fedMap.getID(), fr1[0].getID(), fr2.getID()});
		}
		
		FederatedRequest frGet = new FederatedRequest(RequestType.GET_VAR, frComp.getID());
		FederatedRequest frClean1 = fedMap.cleanup(getTID(), frComp.getID());
		FederatedRequest frClean2 = fedMap.cleanup(getTID(), fr1[0].getID());
		FederatedRequest frClean3 = fedMap.cleanup(getTID(), fr2.getID());

		Future<FederatedResponse>[] response;
		if(fr3 != null) {
			FederatedRequest frClean4 = fedMap.cleanup(getTID(), fr3.getID());
			// execute federated instructions
			response = fedMap.execute(getTID(), true, fr1, fr2, fr3,
				frComp, frGet, frClean1, frClean2, frClean3, frClean4);
		}
		else {
			// execute federated instructions
			response = fedMap.execute(getTID(), true, fr1, fr2,
				frComp, frGet, frClean1, frClean2, frClean3);
		}
		
		//aggregate partial results from federated responses
		AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
		ec.setVariable(output.getName(), FederationUtils.aggScalar(aop, response));
	}
}

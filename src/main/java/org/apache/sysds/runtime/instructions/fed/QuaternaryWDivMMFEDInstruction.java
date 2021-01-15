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

import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.lops.WeightedDivMM.WDivMMType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.QuaternaryOperator;

import java.util.concurrent.Future;

public class QuaternaryWDivMMFEDInstruction extends QuaternaryFEDInstruction
{
	// input1 ... federated X
	// input2 ... U
	// input3 ... V
	// _input4 ... W (=epsilon)
	protected QuaternaryWDivMMFEDInstruction(Operator operator,
		CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4, CPOperand out, String opcode, String instruction_str)
	{
		super(FEDType.Quaternary, operator, in1, in2, in3, in4, out, opcode, instruction_str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
	{
		QuaternaryOperator qop = (QuaternaryOperator) _optr;
		final WDivMMType wdivmm_type = qop.wtype3;
		MatrixObject X = ec.getMatrixObject(input1);
		MatrixObject U = ec.getMatrixObject(input2);
		MatrixObject V = ec.getMatrixObject(input3);
		ScalarObject eps = null;
		MatrixObject MX = null;

		if(qop.hasFourInputs()) {
			if(wdivmm_type == WDivMMType.MULT_MINUS_4_LEFT || wdivmm_type == WDivMMType.MULT_MINUS_4_RIGHT)
			{
				MX = ec.getMatrixObject(_input4);
			}
			else
			{
				eps = (_input4.getDataType() == DataType.SCALAR) ?
					ec.getScalarInput(_input4) :
					new DoubleObject(ec.getMatrixInput(_input4.getName()).quickGetValue(0, 0));
			}
		}

		if(!(X.isFederated() && !U.isFederated() && !V.isFederated()))
			throw new DMLRuntimeException("Unsupported federated inputs (X, U, V) = ("
				+X.isFederated()+", "+U.isFederated()+", "+V.isFederated() + ")");

		FederationMap fedMap = X.getFedMapping();
		FederatedRequest[] frInit1 = fedMap.broadcastSliced(U, false);
		FederatedRequest frInit2 = fedMap.broadcast(V);

		FederatedRequest frInit3 = null;
		FederatedRequest frInit3Arr[] = null;
		FederatedRequest frCompute1 = null;
		// broadcast scalar epsilon if there are four inputs
		if(eps != null)
		{
			frInit3 = fedMap.broadcast(eps);
			// change the is_literal flag from true to false because when broadcasted it is no literal anymore
			instString = instString.replace("true", "false");
			frCompute1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2, input3, _input4},
				new long[]{fedMap.getID(), frInit1[0].getID(), frInit2.getID(), frInit3.getID()});
		}
		else if(MX != null)
		{
			frInit3Arr = fedMap.broadcastSliced(MX, false);
			frCompute1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2, input3, _input4},
				new long[]{fedMap.getID(), frInit1[0].getID(), frInit2.getID(), frInit3Arr[0].getID()});
		}
		else
		{
			frCompute1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{input1, input2, input3},
				new long[]{fedMap.getID(), frInit1[0].getID(), frInit2.getID()});
		}

		// get partial results from federated workers
		FederatedRequest frGet1 = new FederatedRequest(RequestType.GET_VAR, frCompute1.getID());

		FederatedRequest frCleanup1 = fedMap.cleanup(getTID(), frCompute1.getID());
		FederatedRequest frCleanup2 = fedMap.cleanup(getTID(), frInit1[0].getID());
		FederatedRequest frCleanup3 = fedMap.cleanup(getTID(), frInit2.getID());

		Future<FederatedResponse>[] response;
		if(frInit3 != null)
		{
			FederatedRequest frCleanup4 = fedMap.cleanup(getTID(), frInit3.getID());
			// execute federated instructions
			response = fedMap.execute(getTID(), true,
				frInit1, frInit2, frInit3,
				frCompute1, frGet1,
				frCleanup1, frCleanup2, frCleanup3, frCleanup4);
		}
		else if(frInit3Arr != null)
		{
			FederatedRequest frCleanup4 = fedMap.cleanup(getTID(), frInit3Arr[0].getID());
			// execute federated instructions
			fedMap.execute(getTID(), true,
				frInit1, frInit2);
			response = fedMap.execute(getTID(), true,
				frInit3Arr,
				frCompute1, frGet1,
				frCleanup1, frCleanup2, frCleanup3, frCleanup4);
		}
		else
		{
			// execute federated instructions
			response = fedMap.execute(getTID(), true,
				frInit1, frInit2,
				frCompute1, frGet1,
				frCleanup1, frCleanup2, frCleanup3);
		}

		if(wdivmm_type.isLeft())
		{
			// aggregate partial results from federated responses
			AggregateUnaryOperator aop = InstructionUtils.parseBasicAggregateUnaryOperator("uak+");
			ec.setMatrixOutput(output.getName(), FederationUtils.aggMatrix(aop, response, fedMap));
		}
		else if(wdivmm_type.isRight() || wdivmm_type.isBasic())
		{
			// bind partial results from federated responses
			ec.setMatrixOutput(output.getName(), FederationUtils.bind(response, false));
		}
		else
		{
			throw new DMLRuntimeException("Federated WDivMM only supported for BASIC, LEFT or RIGHT variants.");
		}

	}
}

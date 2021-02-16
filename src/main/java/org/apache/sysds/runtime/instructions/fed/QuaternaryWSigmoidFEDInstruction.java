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
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class QuaternaryWSigmoidFEDInstruction extends QuaternaryFEDInstruction {

	/**
	 * This instruction performs:
	 *
	 * UV = U %*% t(V); Z = X * log(1 / (1 + exp(-UV)));
	 *
	 * @param operator        Weighted Sigmoid Federated Instruction.
	 * @param in1             X
	 * @param in2             U
	 * @param in3             V
	 * @param out             The Federated Result Z
	 * @param opcode          ...
	 * @param instruction_str ...
	 */
	protected QuaternaryWSigmoidFEDInstruction(Operator operator, CPOperand in1, CPOperand in2, CPOperand in3,
		CPOperand out, String opcode, String instruction_str) {
		super(FEDType.Quaternary, operator, in1, in2, in3, out, opcode, instruction_str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject X = ec.getMatrixObject(input1);
		MatrixObject U = ec.getMatrixObject(input2);
		MatrixObject V = ec.getMatrixObject(input3);

		if(!(X.isFederated() && !U.isFederated() && !V.isFederated()))
			throw new DMLRuntimeException("Unsupported federated inputs (X, U, V) = (" + X.isFederated() + ", "
				+ U.isFederated() + ", " + V.isFederated() + ")");

		FederationMap fedMap = X.getFedMapping();
		FederatedRequest[] frInit1 = fedMap.broadcastSliced(U, false);
		FederatedRequest frInit2 = fedMap.broadcast(V);

		FederatedRequest frCompute1 = FederationUtils.callInstruction(instString,
			output,
			new CPOperand[] {input1, input2, input3},
			new long[] {fedMap.getID(), frInit1[0].getID(), frInit2.getID()});

		// get partial results from federated workers
		FederatedRequest frGet1 = new FederatedRequest(RequestType.GET_VAR, frCompute1.getID());

		FederatedRequest frCleanup1 = fedMap.cleanup(getTID(), frCompute1.getID());
		FederatedRequest frCleanup2 = fedMap.cleanup(getTID(), frInit1[0].getID());
		FederatedRequest frCleanup3 = fedMap.cleanup(getTID(), frInit2.getID());

		// execute federated instructions
		Future<FederatedResponse>[] response = fedMap
			.execute(getTID(), true, frInit1, frInit2, frCompute1, frGet1, frCleanup1, frCleanup2, frCleanup3);

		// bind partial results from federated responses
		ec.setMatrixOutput(output.getName(), FederationUtils.bind(response, false));

	}
}

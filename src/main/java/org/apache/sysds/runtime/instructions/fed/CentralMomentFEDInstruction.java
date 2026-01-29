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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CentralMomentCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.spark.CentralMomentSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class CentralMomentFEDInstruction extends AggregateUnaryFEDInstruction {

	private CentralMomentFEDInstruction(Operator cm, CPOperand in1,
		CPOperand in2, CPOperand in3, CPOperand out, String opcode, String str)
	{
		super(cm, in1, in2, in3, out, opcode, str);
	}

	public static CentralMomentFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		FederatedOutput fedOut = FederatedOutput.valueOf(parts[parts.length-1]);
		String cleanInstStr = InstructionUtils.removeFEDOutputFlag(str);
		CentralMomentCPInstruction cpInst = CentralMomentCPInstruction.parseInstruction(cleanInstStr);
		CentralMomentFEDInstruction fedInst = parseInstruction(cpInst);
		fedInst._fedOut = fedOut;
		return fedInst;
	}

	public static CentralMomentFEDInstruction parseInstruction(CentralMomentCPInstruction inst) {
		return new CentralMomentFEDInstruction(inst.getOperator(), inst.input1, inst.input2, inst.input3, inst.output,
			inst.getOpcode(), inst.getInstructionString());
	}

	public static CentralMomentFEDInstruction parseInstruction(CentralMomentSPInstruction inst) {
		return new CentralMomentFEDInstruction(inst.getOperator(), inst.input1, inst.input2, inst.input3, inst.output,
			inst.getOpcode(), inst.getInstructionString());
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject mo = ec.getMatrixObject(input1.getName());
		ScalarObject order = ec.getScalarInput(input3 == null ? input2 : input3);

		CMOperator cm_op = ((CMOperator) _optr);
		if (cm_op.getAggOpType() == CMOperator.AggregateOperationTypes.INVALID)
			cm_op = cm_op.setCMAggOp((int) order.getLongValue());

		FederationMap fedMapping = mo.getFedMapping();
		List<CmCovObject> globalCmobj = new ArrayList<>();

		long varID = FederationUtils.getNextFedDataID();
		CMOperator finalCm_op = cm_op;
		fedMapping.mapParallel(varID, (range, data) -> {

			FederatedResponse response;
			try {
				if (input3 == null) {
					response = data
							.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
									new CentralMomentFEDInstruction.CMFunction(data.getVarID(), finalCm_op)))
							.get();
				}
				else {
					MatrixBlock wtBlock = ec.getMatrixInput(input2.getName());

					response = data.executeFederatedOperation(new FederatedRequest(
							FederatedRequest.RequestType.EXEC_UDF, -1,
							new CentralMomentFEDInstruction.CMWeightsFunction(data.getVarID(), finalCm_op, wtBlock)))
							.get();
				}

				if (!response.isSuccessful())
					response.throwExceptionFromResponse();
				synchronized (globalCmobj) {
					globalCmobj.add((CmCovObject) response.getData()[0]);
				}
			}
			catch (Exception e) {
				throw new DMLRuntimeException(e);
			}
			return null;
		});

		Optional<CmCovObject> res = globalCmobj.stream()
				.reduce((arg0, arg1) -> (CmCovObject) finalCm_op.fn.execute(arg0, arg1));
		try {
			ec.setScalarOutput(output.getName(), new DoubleObject(res.get().getRequiredResult(finalCm_op)));
		}
		catch (Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static class CMFunction extends FederatedUDF {
		private static final long serialVersionUID = 7460149207607220994L;
		private final CMOperator _op;

		public CMFunction(long input, CMOperator op) {
			super(new long[] {input});
			_op = op;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, mb.cmOperations(_op));
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	private static class CMWeightsFunction extends FederatedUDF {
		private static final long serialVersionUID = -3685746246551622021L;
		private final CMOperator _op;
		private final MatrixBlock _weights;

		protected CMWeightsFunction(long input, CMOperator op, MatrixBlock weights) {
			super(new long[] {input});
			_op = op;
			_weights = weights;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, mb.cmOperations(_op, _weights));
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}

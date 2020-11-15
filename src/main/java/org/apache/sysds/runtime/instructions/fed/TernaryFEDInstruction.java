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

import io.netty.util.internal.MathUtil;
import org.apache.commons.math3.util.MathUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.IfElse;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.util.IndexRange;
import scala.math.Ordering;

public class TernaryFEDInstruction extends ComputationFEDInstruction {

	private TernaryFEDInstruction(TernaryOperator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String str) {
		super(FEDInstruction.FEDType.Ternary, op, in1, in2, in3, out, opcode, str);
	}

	public static TernaryFEDInstruction parseInstruction(String str)
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode=parts[0];
		CPOperand operand1 = new CPOperand(parts[1]);
		CPOperand operand2 = new CPOperand(parts[2]);
		CPOperand operand3 = new CPOperand(parts[3]);
		CPOperand outOperand = new CPOperand(parts[4]);
		TernaryOperator op = InstructionUtils.parseTernaryOperator(opcode);
		return new TernaryFEDInstruction(op, operand1, operand2, operand3, outOperand, opcode, str);
	}

	@Override
	public void processInstruction( ExecutionContext ec ) {

		MatrixObject mo1 = input1.isMatrix() ? ec.getMatrixObject(input1.getName()) : null;
		MatrixBlock so1 = input1.isScalar() || (mo1 != null && !mo1.isFederated()) ?
			new MatrixBlock(ec.getScalarInput(input1).getDoubleValue()) : null;

		MatrixObject mo2 = input2.isMatrix() ? ec.getMatrixObject(input2.getName()) : null;
		MatrixBlock so2 = input2.isScalar()  || (mo2 != null && !mo2.isFederated())
			? new MatrixBlock(ec.getScalarInput(input2).getDoubleValue()) : null;

		assert input3 != null;
		MatrixObject mo3 = input3.isMatrix() ? ec.getMatrixObject(input3.getName()) : null;
		MatrixBlock so3 = input3.isScalar() || (mo3 != null && !mo3.isFederated())
			? new MatrixBlock(ec.getScalarInput(input3).getDoubleValue()) : null;


//		//prepare inputs
//		final boolean s1 = (rlen==1 && clen==1);
//		final boolean s2 = (m2.rlen==1 && m2.clen==1);
//		final boolean s3 = (m3.rlen==1 && m3.clen==1);
//		final double d1 = s1 ? quickGetValue(0, 0) : Double.NaN;
//		final double d2 = s2 ? m2.quickGetValue(0, 0) : Double.NaN;
//		final double d3 = s3 ? m3.quickGetValue(0, 0) : Double.NaN;
//		final int m = Math.max(Math.max(rlen, m2.rlen), m3.rlen);
//		final int n = Math.max(Math.max(clen, m2.clen), m3.clen);
//		final long nnz = nonZeros;
//
//		//error handling
//		if( (!s1 && (rlen != m || clen != n))
//			|| (!s2 && (m2.rlen != m || m2.clen != n))
//			|| (!s3 && (m3.rlen != m || m3.clen != n)) ) {
//			throw new DMLRuntimeException("Block sizes are not matched for ternary cell operations: "
//				+ rlen + "x" + clen + " vs " + m2.rlen + "x" + m2.clen + " vs " + m3.rlen + "x" + m3.clen);
//		}

		List<Integer> exprs = new ArrayList<>();
		boolean f = false, c = false;
		if(mo1 != null && mo1.isFederated()) {
			mo1.getFedMapping().forEachParallel((range, data) ->  {
				try {
					FederatedResponse response = data
						.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
							new TernaryFEDInstruction.GetExpr(data.getVarID()))).get();

					if(!response.isSuccessful())
						response.throwExceptionFromResponse();
					synchronized(exprs) {
						int expr = (int) response.getData()[0];
						exprs.add(expr);
					}
				} catch(Exception e) {
					throw new DMLRuntimeException(e);
				}
				return null;
			});
			f = (exprs.stream().mapToInt(Integer::intValue).sum() == exprs.size());
		} else if(input1.isMatrix())
			c = so1.sum() == (so1.getNumRows() * so1.getNumColumns());


		if(mo1 == null && so1 != null) {
			if ((so1 != null && so1.quickGetValue(0, 0) == 1.0) || f || c) {
				if (f) {
					MatrixObject out = ec.getMatrixObject(output);
					out.getDataCharacteristics().set(mo2.getNumColumns(),
						mo2.getNumRows(), (int)mo2.getBlocksize(), mo2.getNnz());
					out.setFedMapping(mo2.getFedMapping().copyWithNewID());
				}
				else if(c) ec.setMatrixOutput(output.getName(), so2);
				else {
					ec.setScalarOutput(output.getName(), ScalarObjectFactory
						.createScalarObject(output.getValueType(), so2.getValue(0, 0)));
				}
			}
			else {
				if(mo3 == null && input3.isScalar()) {
					ec.setScalarOutput(output.getName(), ScalarObjectFactory
						.createScalarObject(output.getValueType(), so3.getValue(0, 0)));
				}
				else if (mo3 != null && mo3.isFederated()) {
					MatrixObject out = ec.getMatrixObject(output);
					out.getDataCharacteristics().set(mo3.getNumColumns(),
						mo3.getNumRows(), (int)mo3.getBlocksize(), mo3.getNnz());
					out.setFedMapping(mo3.getFedMapping().copyWithNewID());
				}
				else ec.setMatrixOutput(output.getName(), so3);
			}
		}
	}

	private static class GetExpr extends FederatedUDF {

		private static final long serialVersionUID = 5956832933333848772L;

		private GetExpr(long input) {
			super(new long[] {input});
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();

			int res = (mb.getNonZeros() == (mb.getNumColumns() * mb.getNumRows())) ? 1 : 0;
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, res);
		}
	}
}

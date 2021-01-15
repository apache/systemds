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
import java.util.concurrent.Future;

import io.netty.util.internal.MathUtil;
import org.apache.commons.math3.util.MathUtils;
import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
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
		ScalarObject so1 = mo1 == null ? ec.getScalarInput(input1) : null;
//		mo1 = mo1 == null ? new MatrixBlock(ec.getScalarInput(input1).getDoubleValue()) : null;

		MatrixObject mo2 = input2.isMatrix() ? ec.getMatrixObject(input2.getName()) : null;
		ScalarObject so2 = mo2 == null ? ec.getScalarInput(input2) : null;
//		mo2 = mo2 == null ? new MatrixBlock(ec.getScalarInput(input2).getDoubleValue()) : null;

		MatrixObject mo3 = input3.isMatrix() ? ec.getMatrixObject(input3.getName()) : null;
		ScalarObject so3 = mo3 == null ? ec.getScalarInput(input3) : null;
//		mo3 = mo3 == null ? new MatrixBlock(ec.getScalarInput(input3).getDoubleValue()) : null;

		if(output.isMatrix() && mo1 != null && mo2 != null && mo3 != null)
			processMatrixOutput(ec, mo1, mo2, mo3);
		else {
			// 3 fed
			if((mo1 != null && mo1.isFederated()) && (mo2 != null && mo2.isFederated())
				&& mo3 == null && so3!= null) {
				FederatedRequest fr1 = mo1.getFedMapping().broadcast(so3);
				FederatedRequest fr2 = FederationUtils.callInstruction(instString,
					output, new CPOperand[] {input1, input2, input3},
					new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), fr1.getID()});
				FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
				Future<FederatedResponse>[] ffr = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

				if(output.isMatrix()) {
					//derive new fed mapping for output
					MatrixObject out = ec.getMatrixObject(output);
					out.getDataCharacteristics().set(mo1.getDataCharacteristics());
					out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
				} else {
					try {
						ScalarObject out = (ScalarObject)ffr[0].get().getData()[0];;
						ec.setScalarOutput(output.getName(), out);
					}
					catch(Exception e) {
						e.printStackTrace();
					}
				}
			}
			else if((mo1 != null && mo1.isFederated()) && (mo3 != null && mo3.isFederated())
				&& mo2 == null && so2 != null) {
				FederatedRequest fr1 = mo1.getFedMapping().broadcast(so2);
				FederatedRequest fr2 = FederationUtils.callInstruction(instString,
					output,
					new CPOperand[] {input1, input2, input3},
					new long[] {mo1.getFedMapping().getID(), fr1.getID(), mo3.getFedMapping().getID()});
				FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
				Future<FederatedResponse>[] ffr = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

				if(output.isMatrix()) {
					//derive new fed mapping for output
					MatrixObject out = ec.getMatrixObject(output);
					out.getDataCharacteristics().set(mo1.getDataCharacteristics());
					out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
				} else {
					try {
						ScalarObject out = (ScalarObject)ffr[0].get().getData()[0];;
						ec.setScalarOutput(output.getName(), out);
					}
					catch(Exception e) {
						e.printStackTrace();
					}
				}
			}
			else if((mo2 != null && mo2.isFederated()) && (mo3 != null && mo3.isFederated())
				&& mo1 == null && so1 != null) {
				if(so1.getDoubleValue() == 0) {
					mo2 = mo3;
					so2 = so3;
				}

				if(output.isMatrix()) {
					MatrixObject out = ec.getMatrixObject(output);
					out.getDataCharacteristics().set(mo2.getDataCharacteristics());
					out.setFedMapping(mo2.getFedMapping().copyWithNewID());
				} else {
					try {
						ec.setScalarOutput(output.getName(), so2);
					}
					catch(Exception e) {
						e.printStackTrace();
					}
				}
			}

			// 3 fed
			else if(mo1.isFederated() && !mo2.isFederated() && !mo3.isFederated()) {
				FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
				FederatedRequest[] fr2 = mo1.getFedMapping().broadcastSliced(mo3, false);
				FederatedRequest fr3 = FederationUtils.callInstruction(instString,
					output,
					new CPOperand[] {input1, input2, input3},
					new long[] {mo1.getFedMapping().getID(), fr1[0].getID(), fr2[0].getID()});
				FederatedRequest fr4 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
				FederatedRequest fr5 = mo1.getFedMapping().cleanup(getTID(), fr2[0].getID());
				mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4, fr5);
				//derive new fed mapping for output
				MatrixObject out = ec.getMatrixObject(output);
				out.getDataCharacteristics().set(mo1.getDataCharacteristics());
				out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr3.getID()));
			}
			else if(!mo1.isFederated() && !mo3.isFederated() && mo2.isFederated()) {
				FederatedRequest[] fr1 = mo2.getFedMapping().broadcastSliced(mo1, false);
				FederatedRequest[] fr2 = mo2.getFedMapping().broadcastSliced(mo3, false);
				FederatedRequest fr3 = FederationUtils.callInstruction(instString,
					output,
					new CPOperand[] {input1, input2, input3},
					new long[] {fr1[0].getID(), mo2.getFedMapping().getID(), fr2[0].getID()});
				FederatedRequest fr4 = mo2.getFedMapping().cleanup(getTID(), fr1[0].getID());
				FederatedRequest fr5 = mo2.getFedMapping().cleanup(getTID(), fr2[0].getID());
				mo2.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4, fr5);
				//derive new fed mapping for output
				MatrixObject out = ec.getMatrixObject(output);
				out.getDataCharacteristics().set(mo2.getDataCharacteristics());
				out.setFedMapping(mo2.getFedMapping().copyWithNewID(fr3.getID()));
			}
			else if(!mo2.isFederated() && mo3.isFederated() && !mo1.isFederated()) {
				FederatedRequest[] fr1 = mo3.getFedMapping().broadcastSliced(mo1, false);
				FederatedRequest[] fr2 = mo3.getFedMapping().broadcastSliced(mo3, false);
				FederatedRequest fr3 = FederationUtils.callInstruction(instString,
					output,
					new CPOperand[] {input1, input2, input3},
					new long[] {fr1[0].getID(), fr2[0].getID(), mo3.getFedMapping().getID()});
				FederatedRequest fr4 = mo3.getFedMapping().cleanup(getTID(), fr1[0].getID());
				FederatedRequest fr5 = mo3.getFedMapping().cleanup(getTID(), fr2[0].getID());
				mo3.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4, fr5);
				//derive new fed mapping for output
				MatrixObject out = ec.getMatrixObject(output);
				out.getDataCharacteristics().set(mo3.getDataCharacteristics());
				out.setFedMapping(mo3.getFedMapping().copyWithNewID(fr3.getID()));
			} else {
				throw new DMLRuntimeException("TernaryFEDInstruction: unsupported federated input.");
			}
		}

		// TODO check input dimensions

//		MatrixBlock so1 = input1.isScalar() || (mo1 != null && !mo1.isFederated()) ?
//			new MatrixBlock(ec.getScalarInput(input1).getDoubleValue()) : null;
//
//		MatrixObject mo2 = input2.isMatrix() ? ec.getMatrixObject(input2.getName()) : null;
//		MatrixBlock so2 = input2.isScalar()  || (mo2 != null && !mo2.isFederated())
//			? new MatrixBlock(ec.getScalarInput(input2).getDoubleValue()) : null;
//
//		assert input3 != null;
//		MatrixObject mo3 = input3.isMatrix() ? ec.getMatrixObject(input3.getName()) : null;
//		MatrixBlock so3 = input3.isScalar() || (mo3 != null && !mo3.isFederated())
//			? new MatrixBlock(ec.getScalarInput(input3).getDoubleValue()) : null;

//		MatrixObject mo1 = ec.getMatrixObject(_ins.input1);
//		MatrixObject mo2 = ec.getMatrixObject(_ins.input2);
//		MatrixObject mo3 = _ins.input3.isLiteral() ? null : ec.getMatrixObject(_ins.input3);
//
//		if(mo1.isFederated() && mo2.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false) &&
//			mo3 == null) {
//			FederatedRequest fr1 = mo1.getFedMapping().broadcast(ec.getScalarInput(_ins.input3));
//			FederatedRequest fr2 = FederationUtils.callInstruction(_ins.getInstructionString(),
//				_ins.getOutput(),
//				new CPOperand[] {_ins.input1, _ins.input2, _ins.input3},
//				new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), fr1.getID()});
//			FederatedRequest fr3 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());
//			FederatedRequest fr4 = mo2.getFedMapping().cleanup(getTID(), fr1.getID(), fr2.getID());
//			Future<FederatedResponse>[] tmp = mo1.getFedMapping().execute(getTID(), fr1, fr2, fr3, fr4);


//
//		List<Integer> exprs = new ArrayList<>();
//		boolean f = false, c = false;
//		if(mo1 != null && mo1.isFederated()) {
//			mo1.getFedMapping().forEachParallel((range, data) ->  {
//				try {
//					FederatedResponse response = data
//						.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
//							new TernaryFEDInstruction.GetExpr(data.getVarID()))).get();
//
//					if(!response.isSuccessful())
//						response.throwExceptionFromResponse();
//					synchronized(exprs) {
//						int expr = (int) response.getData()[0];
//						exprs.add(expr);
//					}
//				} catch(Exception e) {
//					throw new DMLRuntimeException(e);
//				}
//				return null;
//			});
//			f = (exprs.stream().mapToInt(Integer::intValue).sum() == exprs.size());
//		} else if(input1.isMatrix())
//			c = so1.sum() == (so1.getNumRows() * so1.getNumColumns());
//
//
//		if(mo1 == null && so1 != null) {
//			if ((so1 != null && so1.quickGetValue(0, 0) == 1.0) || f || c) {
//				if (f) {
//					MatrixObject out = ec.getMatrixObject(output);
//					out.getDataCharacteristics().set(mo2.getNumColumns(),
//						mo2.getNumRows(), (int)mo2.getBlocksize(), mo2.getNnz());
//					out.setFedMapping(mo2.getFedMapping().copyWithNewID());
//				}
//				else if(c) ec.setMatrixOutput(output.getName(), so2);
//				else {
//					ec.setScalarOutput(output.getName(), ScalarObjectFactory
//						.createScalarObject(output.getValueType(), so2.getValue(0, 0)));
//				}
//			}
//			else {
//				if(mo3 == null && input3.isScalar()) {
//					ec.setScalarOutput(output.getName(), ScalarObjectFactory
//						.createScalarObject(output.getValueType(), so3.getValue(0, 0)));
//				}
//				else if (mo3 != null && mo3.isFederated()) {
//					MatrixObject out = ec.getMatrixObject(output);
//					out.getDataCharacteristics().set(mo3.getNumColumns(),
//						mo3.getNumRows(), (int)mo3.getBlocksize(), mo3.getNnz());
//					out.setFedMapping(mo3.getFedMapping().copyWithNewID());
//				}
//				else ec.setMatrixOutput(output.getName(), so3);
//			}
//		}
	}

	private void processMatrixOutput(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2, MatrixObject mo3) {
		// 3 fed
		if(mo1.isFederated() && mo2.isFederated() && mo3.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false) && mo1.getFedMapping().isAligned(mo3.getFedMapping(), false)) {
			FederatedRequest fr2 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), mo3.getFedMapping().getID()});
			mo1.getFedMapping().execute(getTID(), fr2);

			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo1.getDataCharacteristics());
			out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
		}

		// 2 fed
		else if(mo1.isFederated() && mo2.isFederated() && !mo3.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false)) {
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo3, false);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), fr1[0].getID()});
			FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
			mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo1.getDataCharacteristics());
			out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
		}
		else if(mo1.isFederated() && mo3.isFederated() && !mo2.isFederated() && mo1.getFedMapping().isAligned(mo3.getFedMapping(), false)) {
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), fr1[0].getID(), mo3.getFedMapping().getID()});
			FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
			mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo1.getDataCharacteristics());
			out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
		}
		else if(mo2.isFederated() && mo3.isFederated() && !mo1.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false)) {
			FederatedRequest[] fr1 = mo2.getFedMapping().broadcastSliced(mo1, false);
			FederatedRequest fr2 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {fr1[0].getID(), mo2.getFedMapping().getID(), mo3.getFedMapping().getID()});
			FederatedRequest fr3 = mo2.getFedMapping().cleanup(getTID(), fr1[0].getID());
			mo2.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo2.getDataCharacteristics());
			out.setFedMapping(mo2.getFedMapping().copyWithNewID(fr2.getID()));
		}

		// 3 fed
		else if(mo1.isFederated() && !mo2.isFederated() && !mo3.isFederated()) {
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
			FederatedRequest[] fr2 = mo1.getFedMapping().broadcastSliced(mo3, false);
			FederatedRequest fr3 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), fr1[0].getID(), fr2[0].getID()});
			FederatedRequest fr4 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
			FederatedRequest fr5 = mo1.getFedMapping().cleanup(getTID(), fr2[0].getID());
			mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4, fr5);
			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo1.getDataCharacteristics());
			out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr3.getID()));
		}
		else if(!mo1.isFederated() && !mo3.isFederated() && mo2.isFederated()) {
			FederatedRequest[] fr1 = mo2.getFedMapping().broadcastSliced(mo1, false);
			FederatedRequest[] fr2 = mo2.getFedMapping().broadcastSliced(mo3, false);
			FederatedRequest fr3 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {fr1[0].getID(), mo2.getFedMapping().getID(), fr2[0].getID()});
			FederatedRequest fr4 = mo2.getFedMapping().cleanup(getTID(), fr1[0].getID());
			FederatedRequest fr5 = mo2.getFedMapping().cleanup(getTID(), fr2[0].getID());
			mo2.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4, fr5);
			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo2.getDataCharacteristics());
			out.setFedMapping(mo2.getFedMapping().copyWithNewID(fr3.getID()));
		}
		else if(!mo2.isFederated() && mo3.isFederated() && !mo1.isFederated()) {
			FederatedRequest[] fr1 = mo3.getFedMapping().broadcastSliced(mo1, false);
			FederatedRequest[] fr2 = mo3.getFedMapping().broadcastSliced(mo3, false);
			FederatedRequest fr3 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {fr1[0].getID(), fr2[0].getID(), mo3.getFedMapping().getID()});
			FederatedRequest fr4 = mo3.getFedMapping().cleanup(getTID(), fr1[0].getID());
			FederatedRequest fr5 = mo3.getFedMapping().cleanup(getTID(), fr2[0].getID());
			mo3.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4, fr5);
			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo3.getDataCharacteristics());
			out.setFedMapping(mo3.getFedMapping().copyWithNewID(fr3.getID()));
		} else {
			throw new DMLRuntimeException("TernaryFEDInstruction: unsupported federated input.");
		}
	}

//	private static class GetExpr extends FederatedUDF {
//
//		private static final long serialVersionUID = 5956832933333848772L;
//
//		private GetExpr(long input) {
//			super(new long[] {input});
//		}
//
//		@Override
//		public FederatedResponse execute(ExecutionContext ec, Data... data) {
//			MatrixBlock mb = ((MatrixObject) data[0]).acquireReadAndRelease();
//
//			int res = (mb.getNonZeros() == (mb.getNumColumns() * mb.getNumRows())) ? 1 : 0;
//			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, res);
//		}
//	}
}

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
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap.FType;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;

import java.util.concurrent.Future;

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

		//canonicalization for federated lhs
		if( !mo1.isFederated() && mo2.isFederated()
			&& mo1.getDataCharacteristics().equalDims(mo2.getDataCharacteristics())
			&& ((BinaryOperator)_optr).isCommutative() ) {
			mo1 = ec.getMatrixObject(input2);
			mo2 = ec.getMatrixObject(input1);
		}

		//execute federated operation on mo1 or mo2
		FederatedRequest fr2 = null;
		// Future<FederatedResponse>[] response = null;
		if( mo2.isFederated() ) {
			if(mo1.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false)) {
				fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
					new long[]{mo1.getFedMapping().getID(), mo2.getFedMapping().getID()});
				mo1.getFedMapping().execute(getTID(), true, fr2);
			}
			else {
				throw new DMLRuntimeException("Matrix-matrix binary operations with a "
					+ "federated right input are only supported for special cases yet.");
			}
		}
		else {
			boolean isRowPartitioned = mo1.isFederated(FType.ROW);

			if(!isRowPartitioned && !mo1.isFederated(FType.COL)) {
				throw new DMLRuntimeException("Matrix-matrix binary operations only "
					+ "supported with a row partitioned or column partitioned federated "
					+ " input.");
			}

			if(mo1.isFederated(FType.ROW))
			{
				if(mo2.getNumRows() == 1 && mo2.getNumColumns() > 1) { //MV row vector
					FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
					fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
					new long[]{mo1.getFedMapping().getID(), fr1.getID()});
					FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
					//execute federated instruction and cleanup intermediates
					mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
				}
				else { // MM and MV col vector
					FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
					fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
						new long[]{mo1.getFedMapping().getID(), fr1[0].getID()});
					FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
					//execute federated instruction and cleanup intermediates
					mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
				}
			}
			else if(mo1.isFederated(FType.COL))
			{
				// FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
				// fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
				// 	new long[]{mo1.getFedMapping().getID(), fr1.getID()});
				// // get partial results from federated workers
				// FederatedRequest frGet1 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());
				// FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
				// //execute federated instruction and cleanup intermediates
				// response = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, frGet1, fr3);

				// FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
				// fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
				// 	new long[]{mo1.getFedMapping().getID(), fr1[0].getID()});
				// // get partial results from federated workers
				// FederatedRequest frGet1 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());
				// FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
				// //execute federated instruction and cleanup intermediates
				// response = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, frGet1, fr3);

				// FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, true);
				// fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
				// 	new long[]{mo1.getFedMapping().getID(), fr1[0].getID()});
				// // get partial results from federated workers
				// FederatedRequest frGet1 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());
				// FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
				// //execute federated instruction and cleanup intermediates
				// response = mo1.getFedMapping().execute(getTID(), true, fr1, fr2, frGet1, fr3);


				// // get partial results from federated workers
				// FederatedRequest frGet1 = new FederatedRequest(RequestType.GET_VAR, fr2.getID());

				// response = mo1.getFedMapping().execute(getTID(), true, fr1, fr2);
			}

			// //matrix-matrix binary operations -> lhs fed input -> fed output
			// if(mo2.getNumRows() > 1 && mo2.getNumColumns() == 1 && mo1.isFederated(FType.ROW) ) { //MV col vector
			// 	FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
			// 	fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
			// 		new long[]{mo1.getFedMapping().getID(), fr1[0].getID()});
			// 	FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
			// 	//execute federated instruction and cleanup intermediates
			// 	mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			// }
			// else if(mo2.getNumRows() == 1 && mo2.getNumColumns() > 1) { //MV row vector
			// 	assert false: "BinaryMatrixMatrixFEDInstruction.java";
			// 	FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
			// 	fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
			// 		new long[]{mo1.getFedMapping().getID(), fr1.getID()});
			// 	FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
			// 	//execute federated instruction and cleanup intermediates
			// 	mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			// }
			// else { //MM
			// 	if(mo1.isFederated(FType.ROW)) {
			// 		assert false: "BinaryMatrixMatrixFEDInstruction.java:101 - size: " + mo1.getFedMapping().getSize();
			// 		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
			// 		fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
			// 			new long[]{mo1.getFedMapping().getID(), fr1[0].getID()});
			// 		FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
			// 		//execute federated instruction and cleanup intermediates
			// 		mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			// 	}
			// 	else {
			// 		FederatedRequest fr1 = mo1.getFedMapping().broadcast(mo2);
			// 		fr2 = FederationUtils.callInstruction(instString, output, new CPOperand[]{input1, input2},
			// 			new long[]{mo1.getFedMapping().getID(), fr1.getID()});
			// 		FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1.getID());
			// 		//execute federated instruction and cleanup intermediates
			// 		mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);
			// 	}
			// }
		}

		if(mo1.isFederated(FType.ROW))
		{
			// derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);

			out.getDataCharacteristics().set(mo1.getDataCharacteristics());
			out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));
		}
		else if(mo1.isFederated(FType.COL))
		{
			// derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);

			out.getDataCharacteristics().set(mo1.getDataCharacteristics());
			out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr2.getID()));


			// MatrixBlock[] res_mb = null;
			// try {
			// 	 res_mb = FederationUtils.getResults(response);
			// } catch(Exception e) {
			// 	assert false: "BinaryMatrixMatrixFEDInstruction.java:242 - failure in getResults\n" + e.toString();
			// }
			// assert false: "BinaryMatrixMatrixFEDInstruction.java:240 - res_mb size: " + res_mb.length;
		}
	}
}

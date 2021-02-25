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

import java.util.Objects;
import java.util.concurrent.Future;

import com.sun.tools.javac.util.List;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;

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
		MatrixObject mo2 = input2.isMatrix() ? ec.getMatrixObject(input2.getName()) : null;
		MatrixObject mo3 = input3.isMatrix() ? ec.getMatrixObject(input3.getName()) : null;

		long matrixInputsCount = List.of(mo1, mo2, mo3).stream().filter(Objects::nonNull).count();

		if(matrixInputsCount == 3)
			processMatrixInput(ec, mo1, mo2, mo3);
		else if (matrixInputsCount == 1) {
			CPOperand in = mo1 == null ? mo2 == null ? input3 : input2 : input1;
			mo1 = mo1 == null ? mo2 == null ? mo3 : mo2 : mo1;
			processMatrixScalarInput(ec, mo1, in);
		} else
			process2MatrixScalarInput(ec, mo1, mo2, mo3);
	}

	private void processMatrixScalarInput(ExecutionContext ec, MatrixObject mo1, CPOperand in) {
		FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
			new CPOperand[] {in}, new long[] {mo1.getFedMapping().getID()});
		mo1.getFedMapping().execute(getTID(), true, fr1);

		//derive new fed mapping for output
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getDataCharacteristics());
		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr1.getID()));
	}


	private void process2MatrixScalarInput(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2, MatrixObject mo3) {
		CPOperand[] inputArgs = new CPOperand[] {input1, input2};
		if(mo1 != null && mo1.isFederated() && mo2 == null) {
			mo2 = mo3;
			inputArgs = new CPOperand[] {input1, input3};
		} else if(mo2 != null && mo2.isFederated() && mo1 == null) {
			mo1 = mo2;
			mo2 = mo3;
			inputArgs = new CPOperand[] {input2, input3};
		} else if(mo2 != null && mo2.isFederated() && mo1 != null) {
			mo1 = mo2;
			mo2 = ec.getMatrixObject(input1);
			inputArgs = new CPOperand[] {input2, input1};
		} else if(mo3 != null && mo3.isFederated() && mo1 == null) {
			mo1 = mo3;
			inputArgs = new CPOperand[] {input3, input2};
		} else if(mo3 != null && mo3.isFederated() && mo1 != null) {
			mo1 = mo3;
			mo2 = ec.getMatrixObject(input1);

			inputArgs = new CPOperand[] {input3, input1};
		}

		FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);

		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output,
			inputArgs, new long[] {mo1.getFedMapping().getID(), fr1[0].getID()});

		FederatedRequest fr3 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID());
		mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3);

		//derive new fed mapping for output
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getDataCharacteristics());
		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr3.getID()));
	}


	private void processMatrixInput(ExecutionContext ec, MatrixObject mo1, MatrixObject mo2, MatrixObject mo3) {
		if(!mo1.isFederated())
			if(mo2.isFederated()) {
				mo1 = mo2;
				mo2 = ec.getMatrixObject(input1);
			} else {
				mo1 = mo3;
				mo3 = ec.getMatrixObject(input1);
			}

		FederatedRequest fr3;
		// all 3 inputs aligned on the one worker
		if(mo1.isFederated() && mo2.isFederated() && mo3.isFederated() && mo1.getFedMapping().isAligned(mo2.getFedMapping(), false) && mo1.getFedMapping().isAligned(mo3.getFedMapping(), false)) {
			fr3 = FederationUtils.callInstruction(instString,
				output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), mo2.getFedMapping().getID(), mo3.getFedMapping().getID()});
			mo1.getFedMapping().execute(getTID(), fr3);
		} else {
			FederatedRequest[] fr1 = mo1.getFedMapping().broadcastSliced(mo2, false);
			FederatedRequest[] fr2 = mo1.getFedMapping().broadcastSliced(mo3, false);

			if(!mo1.isFederated())
				if(mo2.isFederated())
					fr3 = FederationUtils.callInstruction(instString,
						output,
						new CPOperand[] {input1, input2, input3},
						new long[] {fr1[0].getID(), mo1.getFedMapping().getID(), fr2[0].getID()});
				else
					fr3 = FederationUtils.callInstruction(instString,
						output,
						new CPOperand[] {input1, input2, input3},
						new long[] {fr1[0].getID(), fr2[0].getID(), mo1.getFedMapping().getID()});
			else fr3 = FederationUtils.callInstruction(instString, output,
				new CPOperand[] {input1, input2, input3},
				new long[] {mo1.getFedMapping().getID(), fr1[0].getID(), fr2[0].getID()});

			FederatedRequest fr4 = mo1.getFedMapping().cleanup(getTID(), fr1[0].getID(), fr2[0].getID());
			mo1.getFedMapping().execute(getTID(), true, fr1, fr2, fr3, fr4);
		}
		//derive new fed mapping for output
		MatrixObject out = ec.getMatrixObject(output);
		out.getDataCharacteristics().set(mo1.getDataCharacteristics());
		out.setFedMapping(mo1.getFedMapping().copyWithNewID(fr3.getID()));
	}
}

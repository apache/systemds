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
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.functionobjects.OffsetColumnIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class AppendFEDInstruction extends BinaryFEDInstruction {
	protected boolean _cbind; // otherwise rbind

	protected AppendFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, boolean cbind,
		String opcode, String istr) {
		super(FEDType.Append, op, in1, in2, out, opcode, istr);
		_cbind = cbind;
	}

	public static AppendFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 6, 5, 4);

		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand out = new CPOperand(parts[parts.length - 2]);
		boolean cbind = Boolean.parseBoolean(parts[parts.length - 1]);

		Operator op = new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1));
		return new AppendFEDInstruction(op, in1, in2, out, cbind, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		// get inputs
		MatrixObject mo1 = ec.getMatrixObject(input1.getName());
		MatrixObject mo2 = ec.getMatrixObject(input2.getName());
		DataCharacteristics dc1 = mo1.getDataCharacteristics();
		DataCharacteristics dc2 = mo1.getDataCharacteristics();

		// check input dimensions
		if(_cbind && mo1.getNumRows() != mo2.getNumRows()) {
			StringBuilder sb = new StringBuilder();
			sb.append("Append-cbind is not possible for federated input matrices ");
			sb.append(input1.getName()).append(" and ").append(input2.getName());
			sb.append(" with different number of rows: ");
			sb.append(mo1.getNumRows()).append(" vs ").append(mo2.getNumRows());
			throw new DMLRuntimeException(sb.toString());
		}
		else if(!_cbind && mo1.getNumColumns() != mo2.getNumColumns()) {
			StringBuilder sb = new StringBuilder();
			sb.append("Append-rbind is not possible for federated input matrices ");
			sb.append(input1.getName()).append(" and ").append(input2.getName());
			sb.append(" with different number of columns: ");
			sb.append(mo1.getNumColumns()).append(" vs ").append(mo2.getNumColumns());
			throw new DMLRuntimeException(sb.toString());
		}

		FederationMap fm1;
		if(mo1.isFederated())
			fm1 = mo1.getFedMapping();
		else
			fm1 = FederationUtils.federateLocalData(mo1);
		FederationMap fm2;
		if(mo2.isFederated())
			fm2 = mo2.getFedMapping();
		else
			fm2 = FederationUtils.federateLocalData(mo2);

		MatrixObject out = ec.getMatrixObject(output);
		long id = FederationUtils.getNextFedDataID();
		if(_cbind) {
			out.getDataCharacteristics().set(dc1.getRows(),
				dc1.getCols() + dc2.getCols(),
				dc1.getBlocksize(),
				dc1.getNonZeros() + dc2.getNonZeros());
			out.setFedMapping(fm1.identCopy(getTID(), id).bind(0, dc1.getCols(), fm2.identCopy(getTID(), id)));
		}
		else {
			out.getDataCharacteristics().set(dc1.getRows() + dc2.getRows(),
				dc1.getCols(),
				dc1.getBlocksize(),
				dc1.getNonZeros() + dc2.getNonZeros());
			out.setFedMapping(fm1.identCopy(getTID(), id).bind(dc1.getRows(), 0, fm2.identCopy(getTID(), id)));
		}
	}
}

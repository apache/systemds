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
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.CastSPInstruction;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class CastFEDInstruction extends UnaryFEDInstruction {

	private CastFEDInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr) {
		super(FEDInstruction.FEDType.Cast, op, in, out, opcode, istr);
	}

	public static CastFEDInstruction parseInstruction(CastSPInstruction spInstruction) {
		return new CastFEDInstruction(spInstruction.getOperator(), spInstruction.input1, spInstruction.output,
			spInstruction.getOpcode(), spInstruction.getInstructionString());
	}

	public static CastFEDInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 2);
		String opcode = parts[0];
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		return new CastFEDInstruction(null, in, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(getOpcode().equals(OpOp1.CAST_AS_MATRIX.toString()))
			processCastAsMatrixVariableInstruction(ec);
		else if(getOpcode().equals(OpOp1.CAST_AS_FRAME.toString()))
			processCastAsFrameVariableInstruction(ec);
		else
			throw new DMLRuntimeException("Unsupported Opcode for federated Variable Instruction : " + getOpcode());
	}

	private void processCastAsMatrixVariableInstruction(ExecutionContext ec) {

		FrameObject mo1 = ec.getFrameObject(input1);

		if(!mo1.isFederated())
			throw new DMLRuntimeException(
				"Federated Cast: " + "Federated input expected, but invoked w/ " + mo1.isFederated());

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest fr1 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), Types.DataType.MATRIX);

		// execute function at federated site.
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, id,
			new CPOperand[] {input1}, new long[] {mo1.getFedMapping().getID()}, Types.ExecType.SPARK, false);
		mo1.getFedMapping().execute(getTID(), true, fr1, fr2);

		// Construct output local.

		MatrixObject out = ec.getMatrixObject(output);
		FederationMap outMap = mo1.getFedMapping().copyWithNewID(fr1.getID());
		List<Pair<FederatedRange, FederatedData>> newMap = new ArrayList<>();
		for(Pair<FederatedRange, FederatedData> pair : outMap.getMap()) {
			FederatedData om = pair.getValue();
			FederatedData nf = new FederatedData(Types.DataType.MATRIX,
				om.getAddress(), om.getFilepath(), om.getVarID());
			newMap.add(Pair.of(pair.getKey(), nf));
		}
		out.setFedMapping(outMap);
	}

	private void processCastAsFrameVariableInstruction(ExecutionContext ec) {

		MatrixObject mo1 = ec.getMatrixObject(input1);

		if(!mo1.isFederated())
			throw new DMLRuntimeException(
				"Federated Reorg: " + "Federated input expected, but invoked w/ " + mo1.isFederated());

		long id = FederationUtils.getNextFedDataID();
		FederatedRequest fr1 = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, new MatrixCharacteristics(-1, -1), Types.DataType.FRAME);

		// execute function at federated site.
		FederatedRequest fr2 = FederationUtils.callInstruction(instString, output, id,
			new CPOperand[] {input1}, new long[] {mo1.getFedMapping().getID()}, Types.ExecType.SPARK, false);
		mo1.getFedMapping().execute(getTID(), true, fr1, fr2);

		// Construct output local.
		FrameObject out = ec.getFrameObject(output);
		out.getDataCharacteristics().set(mo1.getNumRows(), mo1.getNumColumns(), (int) mo1.getBlocksize(), mo1.getNnz());
		FederationMap outMap = mo1.getFedMapping().copyWithNewID(fr2.getID());
		List<Pair<FederatedRange, FederatedData>> newMap = new ArrayList<>();
		for(Map.Entry<FederatedRange, FederatedData> pair : outMap.getMap()) {
			FederatedData om = pair.getValue();
			FederatedData nf = new FederatedData(Types.DataType.FRAME,
				om.getAddress(), om.getFilepath(), om.getVarID());
			newMap.add(Pair.of(pair.getKey(), nf));
		}
		ValueType[] schema = new ValueType[(int) mo1.getDataCharacteristics().getCols()];
		Arrays.fill(schema, ValueType.FP64);
		out.setSchema(schema);
		out.setFedMapping(outMap);
	}

}

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
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction.VariableOperationCode;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageTraceable;

// TODO: merge with `CastFEDInstruction`
public class VariableFEDInstruction extends FEDInstruction implements LineageTraceable {
	private static final Log LOG = LogFactory.getLog(VariableFEDInstruction.class.getName());

	private final VariableCPInstruction _in;

	protected VariableFEDInstruction(VariableCPInstruction in) {
		super(null, in.getOperator(), in.getOpcode(), in.getInstructionString());
		_in = in;
	}

	public static VariableFEDInstruction parseInstruction(VariableCPInstruction inst, ExecutionContext ec) {
		if(inst.getVariableOpcode() == VariableOperationCode.Write && inst.getInput1().isMatrix() &&
			inst.getInput3().getName().contains("federated")) {
			return VariableFEDInstruction.parseInstruction(inst);
		}
		else if(inst.getVariableOpcode() == VariableOperationCode.CastAsFrameVariable && inst.getInput1().isMatrix() &&
			ec.getCacheableData(inst.getInput1()).isFederatedExcept(FType.BROADCAST)) {
			return VariableFEDInstruction.parseInstruction(inst);
		}
		else if(inst.getVariableOpcode() == VariableOperationCode.CastAsMatrixVariable && inst.getInput1().isFrame() &&
			ec.getCacheableData(inst.getInput1()).isFederatedExcept(FType.BROADCAST)) {
			return VariableFEDInstruction.parseInstruction(inst);
		}
		return null;
	}

	private static VariableFEDInstruction parseInstruction(VariableCPInstruction cpInstruction) {
		return new VariableFEDInstruction(cpInstruction);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		VariableOperationCode opcode = _in.getVariableOpcode();
		switch(opcode) {
			case Write:
				processWriteInstruction(ec);
				break;
			case CastAsMatrixVariable:
				processCastAsMatrixVariableInstruction(ec);
				break;
			case CastAsFrameVariable:
				processCastAsFrameVariableInstruction(ec);
				break;
			default:
				throw new DMLRuntimeException("Unsupported Opcode for federated Variable Instruction : " + opcode);
		}
	}

	private void processWriteInstruction(ExecutionContext ec) {
		LOG.warn("Processing write command federated");
		// TODO Add write command to the federated site if the matrix has been modified
		// this has to be done while appending some string to the federated output file.
		// furthermore the outputted file on the federated sites path should be returned
		// the controller.
		_in.processInstruction(ec);
	}

	private void processCastAsMatrixVariableInstruction(ExecutionContext ec) {

		FrameObject mo1 = ec.getFrameObject(_in.getInput1());

		if(!mo1.isFederated())
			throw new DMLRuntimeException(
				"Federated Reorg: " + "Federated input expected, but invoked w/ " + mo1.isFederated());

		// execute function at federated site.
		FederatedRequest fr1 = FederationUtils.callInstruction(_in.getInstructionString(),
			_in.getOutput(),
			new CPOperand[] {_in.getInput1()},
			new long[] {mo1.getFedMapping().getID()});
		mo1.getFedMapping().execute(getTID(), true, fr1);

		// Construct output local.

		MatrixObject out = ec.getMatrixObject(_in.getOutput());
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

		MatrixObject mo1 = ec.getMatrixObject(_in.getInput1());

		if(!mo1.isFederated())
			throw new DMLRuntimeException(
				"Federated Reorg: " + "Federated input expected, but invoked w/ " + mo1.isFederated());

		// execute function at federated site.
		FederatedRequest fr1 = FederationUtils.callInstruction(_in.getInstructionString(),
			_in.getOutput(), new CPOperand[] {_in.getInput1()}, new long[] {mo1.getFedMapping().getID()});
		mo1.getFedMapping().execute(getTID(), true, fr1);

		// Construct output local.
		FrameObject out = ec.getFrameObject(_in.getOutput());
		out.getDataCharacteristics().set(mo1.getNumRows(), mo1.getNumColumns(), (int) mo1.getBlocksize(), mo1.getNnz());
		FederationMap outMap = mo1.getFedMapping().copyWithNewID(fr1.getID());
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

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return _in.getLineageItem(ec);
	}
}

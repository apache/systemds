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

import java.util.Arrays;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ReshapeCPInstruction;
import org.apache.sysds.runtime.instructions.spark.MatrixReshapeSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class ReshapeFEDInstruction extends UnaryFEDInstruction {
	private final CPOperand _opRows;
	private final CPOperand _opCols;
	private final CPOperand _opDims;
	private final CPOperand _opByRow;

	private ReshapeFEDInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand in4,
		CPOperand in5, CPOperand out, String opcode, String istr) {
		super(FEDInstruction.FEDType.Reshape, op, in1, out, opcode, istr);
		_opRows = in2;
		_opCols = in3;
		_opDims = in4;
		_opByRow = in5;
	}

	public static ReshapeFEDInstruction parseInstruction(ReshapeCPInstruction instr) {
		return new ReshapeFEDInstruction(instr.getOperator(), instr.input1, instr.getOpRows(), instr.getOpCols(),
			instr.getOpDims(), instr.getOpByRow(), instr.output, instr.getOpcode(), instr.getInstructionString());
	}

	public static ReshapeFEDInstruction parseInstruction(MatrixReshapeSPInstruction instr) {
		// TODO: add dims argument (for tensors) to MatrixReshapeSPInstruction
		return new ReshapeFEDInstruction(instr.getOperator(), instr.input1, instr.getOpRows(), instr.getOpCols(), null,
			instr.getOpByRow(), instr.output, instr.getOpcode(), instr.getInstructionString());
	}

	public static ReshapeFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 6, 7);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand in4 = new CPOperand(parts[4]);
		CPOperand in5 = new CPOperand(parts[5]);
		CPOperand out = new CPOperand(parts[6]);
		if(!opcode.equalsIgnoreCase("rshape"))
			throw new DMLRuntimeException("Unknown opcode while parsing an ReshapeInstruction: " + str);
		else
			return new ReshapeFEDInstruction(new Operator(true), in1, in2, in3, in4, in5, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(output.getDataType() == Types.DataType.MATRIX) {
			MatrixObject mo1 = ec.getMatrixObject(input1);
			BooleanObject byRow = (BooleanObject) ec
				.getScalarInput(_opByRow.getName(), Types.ValueType.BOOLEAN, _opByRow.isLiteral());
			int rows = (int) ec.getScalarInput(_opRows).getLongValue();
			int cols = (int) ec.getScalarInput(_opCols).getLongValue();

			if(!mo1.isFederated())
				throw new DMLRuntimeException("Federated Rshape: " 
					+ "Federated input expected, but invoked w/ " + mo1.isFederated());
			if(mo1.getNumColumns() * mo1.getNumRows() != rows * cols)
				throw new DMLRuntimeException("Reshape matrix requires consistent numbers of input/output cells (" 
					+ mo1.getNumRows() + ":" + mo1.getNumColumns() + ", " + rows + ":" + cols + ").");

			boolean isNotAligned = Arrays.stream(mo1.getFedMapping().getFederatedRanges())
				.map(e -> e.getSize() % (byRow.getBooleanValue() ? cols : rows) == 0).collect(Collectors.toList())
				.contains(false);

			if(isNotAligned)
				throw new DMLRuntimeException(
					"Reshape matrix requires consistent numbers of input/output cells for each worker.");

			String[] newInstString = getNewInstString(mo1, instString, rows, cols, byRow.getBooleanValue());

			long id = FederationUtils.getNextFedDataID();
			FederatedRequest tmp = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, id, mo1.getMetaData().getDataCharacteristics(), mo1.getDataType());

			//execute at federated site
			FederatedRequest[] fr1 = FederationUtils.callInstruction(newInstString, output, id,
				new CPOperand[] {input1}, new long[] {mo1.getFedMapping().getID()}, InstructionUtils.getExecType(instString));
			mo1.getFedMapping().execute(getTID(), true, tmp);
			mo1.getFedMapping().execute(getTID(), true, fr1, new FederatedRequest[0]);

			// set new fed map
			FederationMap reshapedFedMap = mo1.getFedMapping();
			for(int i = 0; i < reshapedFedMap.getFederatedRanges().length; i++) {
				long cells = reshapedFedMap.getFederatedRanges()[i].getSize();
				long row = byRow.getBooleanValue() ? cells / cols : rows;
				long col = byRow.getBooleanValue() ? cols : cells / rows;

				reshapedFedMap.getFederatedRanges()[i].setBeginDim(0,
					(reshapedFedMap.getFederatedRanges()[i].getBeginDims()[0] == 0 || i == 0) ? 0 : 
					reshapedFedMap.getFederatedRanges()[i - 1].getEndDims()[0]);
				reshapedFedMap.getFederatedRanges()[i]
					.setEndDim(0, reshapedFedMap.getFederatedRanges()[i].getBeginDims()[0] + row);
				reshapedFedMap.getFederatedRanges()[i].setBeginDim(1,
					(reshapedFedMap.getFederatedRanges()[i].getBeginDims()[1] == 0 || i == 0) ? 0 :
					reshapedFedMap.getFederatedRanges()[i - 1].getEndDims()[1]);
				reshapedFedMap.getFederatedRanges()[i]
					.setEndDim(1, reshapedFedMap.getFederatedRanges()[i].getBeginDims()[1] + col);
			}

			//derive output federated mapping
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(rows, cols, (int) mo1.getBlocksize(), mo1.getNnz());
			out.setFedMapping(reshapedFedMap.copyWithNewID(fr1[0].getID()));
		}
		else {
			// TODO support tensor out, frame and list
			throw new DMLRuntimeException("Federated Reshape Instruction only supports matrix as output.");
		}
	}

	// replace old reshape values for each worker
	private static String[] getNewInstString(MatrixObject mo1, String instString, int rows, int cols, boolean byRow) {
		String[] instStrings = new String[mo1.getFedMapping().getSize()];

		int sameFedSize = Arrays.stream(mo1.getFedMapping().getFederatedRanges()).map(FederatedRange::getSize)
			.collect(Collectors.toSet()).size();
		sameFedSize = sameFedSize == 1 ? 1 : mo1.getFedMapping().getSize();

		for(int i = 0; i < sameFedSize; i++) {
			String[] instParts = instString.split(Lop.OPERAND_DELIMITOR);
			long size = mo1.getFedMapping().getFederatedRanges()[i].getSize();
			String oldInstStringPart = byRow ? instParts[3] : instParts[4];
			String newInstStringPart = byRow ? 
				oldInstStringPart.replace(String.valueOf(rows), String.valueOf(size/cols)) :
				oldInstStringPart.replace(String.valueOf(cols), String.valueOf(size/rows));
			instStrings[i] = instString.replace(oldInstStringPart, newInstStringPart);
		}

		if(sameFedSize == 1)
			Arrays.fill(instStrings, instStrings[0]);

		return instStrings;
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(output.getName(),
			new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, input1, _opRows, _opCols, _opDims, _opByRow)));
	}
}

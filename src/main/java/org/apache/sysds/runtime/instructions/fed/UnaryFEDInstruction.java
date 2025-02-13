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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CentralMomentCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuantileSortCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReshapeCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryMatrixCPInstruction;
import org.apache.sysds.runtime.instructions.spark.AggregateUnarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.CentralMomentSPInstruction;
import org.apache.sysds.runtime.instructions.spark.IndexingSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuantileSortSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReblockSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReorgSPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnaryMatrixSPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnarySPInstruction;
import org.apache.sysds.runtime.matrix.operators.Operator;

public abstract class UnaryFEDInstruction extends ComputationFEDInstruction {
	
	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in, CPOperand out, String opcode, String instr) {
		this(type, op, in, null, null, out, opcode, instr);
	}

	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in, CPOperand out, String opcode, String instr,
		FederatedOutput fedOut) {
		this(type, op, in, null, null, out, opcode, instr, fedOut);
	}
	
	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String instr) {
		this(type, op, in1, in2, null, out, opcode, instr);
	}

	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
		String instr, FederatedOutput fedOut) {
		this(type, op, in1, in2, null, out, opcode, instr, fedOut);
	}
	
	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
			String opcode, String instr) {
		this(type, op, in1, in2, in3, out, opcode, instr, FederatedOutput.NONE);
	}

	protected UnaryFEDInstruction(FEDType type, Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String instr, FederatedOutput fedOut) {
		super(type, op, in1, in2, in3, out, opcode, instr, fedOut);
	}
	
	public static UnaryFEDInstruction parseInstruction(UnaryCPInstruction inst, ExecutionContext ec) {
		if(inst instanceof IndexingCPInstruction) {
			// matrix and frame indexing
			IndexingCPInstruction minst = (IndexingCPInstruction) inst;
			if((minst.input1.isMatrix() || minst.input1.isFrame()) &&
				ec.getCacheableData(minst.input1).isFederatedExcept(FType.BROADCAST)) {
				return IndexingFEDInstruction.parseInstruction(minst);
			}
		}
		else if(inst instanceof ReorgCPInstruction &&
				(inst.getOpcode().equals(Opcodes.TRANSPOSE.toString()) || inst.getOpcode().equals(Opcodes.DIAG.toString())
						|| inst.getOpcode().equals(Opcodes.REV.toString()) || inst.getOpcode().equals(Opcodes.ROLL.toString()))) {
			ReorgCPInstruction rinst = (ReorgCPInstruction) inst;
			CacheableData<?> mo = ec.getCacheableData(rinst.input1);

			if((mo instanceof MatrixObject || mo instanceof FrameObject) && mo.isFederatedExcept(FType.BROADCAST))
				return ReorgFEDInstruction.parseInstruction(rinst);
		}
		else if(inst.input1 != null && inst.input1.isMatrix() && ec.containsVariable(inst.input1)) {

			MatrixObject mo1 = ec.getMatrixObject(inst.input1);
			if(mo1.isFederatedExcept(FType.BROADCAST)) {
				if(inst instanceof CentralMomentCPInstruction)
					return CentralMomentFEDInstruction.parseInstruction((CentralMomentCPInstruction) inst);
				else if(inst instanceof QuantileSortCPInstruction) {
					if(mo1.isFederated(FType.ROW) ||
						mo1.getFedMapping().getFederatedRanges().length == 1 && mo1.isFederated(FType.COL))
						return QuantileSortFEDInstruction.parseInstruction((QuantileSortCPInstruction) inst);
				}
				else if(inst instanceof ReshapeCPInstruction)
					return ReshapeFEDInstruction.parseInstruction((ReshapeCPInstruction) inst);
				else if(inst instanceof AggregateUnaryCPInstruction &&
					((AggregateUnaryCPInstruction) inst).getAUType() == AggregateUnaryCPInstruction.AUType.DEFAULT)
					return AggregateUnaryFEDInstruction.parseInstruction((AggregateUnaryCPInstruction) inst);
				else if(inst instanceof UnaryMatrixCPInstruction) {
					if(UnaryMatrixFEDInstruction.isValidOpcode(inst.getOpcode()) &&
						!(inst.getOpcode().equalsIgnoreCase("ucumk+*") && mo1.isFederated(FType.COL)))
						return UnaryMatrixFEDInstruction.parseInstruction((UnaryMatrixCPInstruction) inst);
				}
			}
		}
		return null;
	}

	public static UnaryFEDInstruction parseInstruction(UnarySPInstruction inst, ExecutionContext ec) {
		if(inst instanceof IndexingSPInstruction) {
			// matrix and frame indexing
			IndexingSPInstruction minst = (IndexingSPInstruction) inst;
			if((minst.input1.isMatrix() || minst.input1.isFrame()) &&
				ec.getCacheableData(minst.input1).isFederatedExcept(FType.BROADCAST)) {
				return IndexingFEDInstruction.parseInstruction(minst);
			}
		}
		else if(inst instanceof CentralMomentSPInstruction) {
			CentralMomentSPInstruction cinstruction = (CentralMomentSPInstruction) inst;
			Data data = ec.getVariable(cinstruction.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederated() &&
				((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
				return CentralMomentFEDInstruction.parseInstruction(cinstruction);
		}
		else if(inst instanceof QuantileSortSPInstruction) {
			QuantileSortSPInstruction qinstruction = (QuantileSortSPInstruction) inst;
			Data data = ec.getVariable(qinstruction.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederated() &&
				((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
				return QuantileSortFEDInstruction.parseInstruction(qinstruction);
		}
		else if(inst instanceof AggregateUnarySPInstruction) {
			AggregateUnarySPInstruction auinstruction = (AggregateUnarySPInstruction) inst;
			Data data = ec.getVariable(auinstruction.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederated() &&
				((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
				if(ArrayUtils.contains(new String[] {"uarimin", "uarimax"}, auinstruction.getOpcode())) {
					if(((MatrixObject) data).getFedMapping().getType() == FType.ROW)
						return AggregateUnaryFEDInstruction.parseInstruction(auinstruction);
				}
				else
					return AggregateUnaryFEDInstruction.parseInstruction(auinstruction);
		}
		else if(inst instanceof ReorgSPInstruction &&
				(inst.getOpcode().equals(Opcodes.TRANSPOSE.toString()) || inst.getOpcode().equals(Opcodes.DIAG.toString())
						|| inst.getOpcode().equals(Opcodes.REV.toString()) || inst.getOpcode().equals(Opcodes.ROLL.toString()))) {
			ReorgSPInstruction rinst = (ReorgSPInstruction) inst;
			CacheableData<?> mo = ec.getCacheableData(rinst.input1);
			if((mo instanceof MatrixObject || mo instanceof FrameObject) && mo.isFederated() &&
				mo.isFederatedExcept(FType.BROADCAST))
				return ReorgFEDInstruction.parseInstruction(rinst);
		}
		else if(inst instanceof ReblockSPInstruction && inst.input1 != null &&
			(inst.input1.isFrame() || inst.input1.isMatrix())) {
			ReblockSPInstruction rinst = (ReblockSPInstruction) inst;
			CacheableData<?> data = ec.getCacheableData(rinst.input1);
			if(data.isFederatedExcept(FType.BROADCAST))
				return ReblockFEDInstruction.parseInstruction((ReblockSPInstruction) inst);
		}
		else if(inst.input1 != null && inst.input1.isMatrix() && ec.containsVariable(inst.input1)) {
			MatrixObject mo1 = ec.getMatrixObject(inst.input1);
			if(mo1.isFederatedExcept(FType.BROADCAST)) {
				if(inst.getOpcode().equalsIgnoreCase(Opcodes.CM.toString()))
					return CentralMomentFEDInstruction.parseInstruction((CentralMomentSPInstruction) inst);
				else if(inst.getOpcode().equalsIgnoreCase(Opcodes.QSORT.toString())) {
					if(mo1.getFedMapping().getFederatedRanges().length == 1)
						return QuantileSortFEDInstruction.parseInstruction(inst.getInstructionString(), false);
				}
				else if(inst.getOpcode().equalsIgnoreCase("rshape")) {
					return ReshapeFEDInstruction.parseInstruction(inst.getInstructionString());
				}
				else if(inst instanceof UnaryMatrixSPInstruction) {
					if(UnaryMatrixFEDInstruction.isValidOpcode(inst.getOpcode()))
						return UnaryMatrixFEDInstruction.parseInstruction((UnaryMatrixSPInstruction) inst);
				}
			}
		}
		return null;
	}

	protected static String parseUnaryInstruction(String instr, CPOperand in, CPOperand out) {
		//TODO: simplify once all fed instructions have consistent flags
		int num = InstructionUtils.checkNumFields(instr, 2, 3, 4);
		if(num == 2)
			return parse(instr, in, null, null, out); 
		else {
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
			String opcode = parts[0];
			in.split(parts[1]);
			out.split(parts[2]);
			return opcode;
		}
	}
	
	protected static String parseUnaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand out) {
		InstructionUtils.checkNumFields(instr, 3);
		return parse(instr, in1, in2, null, out);
	}
	
	protected static String parseUnaryInstruction(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out) {
		InstructionUtils.checkNumFields(instr, 4);
		return parse(instr, in1, in2, in3, out);
	}
	
	private static String parse(String instr, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		
		// first part is the opcode, last part is the output, middle parts are input operands
		String opcode = parts[0];
		out.split(parts[parts.length - 1]);
		
		switch (parts.length) {
			case 3:
				in1.split(parts[1]);
				in2 = null;
				in3 = null;
				break;
			case 4:
				in1.split(parts[1]);
				in2.split(parts[2]);
				in3 = null;
				break;
			case 5:
				in1.split(parts[1]);
				in2.split(parts[2]);
				in3.split(parts[3]);
				break;
			default:
				throw new DMLRuntimeException("Unexpected number of operands in the instruction: " + instr);
		}
		return opcode;
	}
	
	/**
	 * Parse and return federated output flag from given instr string at given position.
	 * If the position given is greater than the length of the instruction, FederatedOutput.NONE is returned.
	 * @param instr instruction string to be parsed
	 * @param position of federated output flag
	 * @return parsed federated output flag or FederatedOutput.NONE
	 */
	static FederatedOutput parseFedOutFlag(String instr, int position){
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(instr);
		if ( parts.length > position )
			return FederatedOutput.valueOf(parts[position]);
		else return FederatedOutput.NONE;
	}
}

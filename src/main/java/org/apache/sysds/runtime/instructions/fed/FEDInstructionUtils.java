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

import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap.FType;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.instructions.spark.AggregateUnarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendGAlignedSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MapmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.WriteSPInstruction;

public class FEDInstructionUtils {
	// This is currently a rather simplistic to our solution of replacing instructions with their correct federated
	// counterpart, since we do not propagate the information that a matrix is federated, therefore we can not decide
	// to choose a federated instruction earlier.
	public static Instruction checkAndReplaceCP(Instruction inst, ExecutionContext ec) {
		FEDInstruction fedinst = null;
		if (inst instanceof AggregateBinaryCPInstruction) {
			AggregateBinaryCPInstruction instruction = (AggregateBinaryCPInstruction) inst;
			if( instruction.input1.isMatrix() && instruction.input2.isMatrix() ) {
				MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
				MatrixObject mo2 = ec.getMatrixObject(instruction.input2);
				if (mo1.isFederated(FType.ROW) || mo2.isFederated(FType.ROW)) {
					fedinst = AggregateBinaryFEDInstruction.parseInstruction(inst.getInstructionString());
				}
			}
		}
		else if( inst instanceof MMChainCPInstruction) {
			MMChainCPInstruction linst = (MMChainCPInstruction) inst;
			MatrixObject mo = ec.getMatrixObject(linst.input1);
			if( mo.isFederated() )
				fedinst = MMChainFEDInstruction.parseInstruction(linst.getInstructionString());
		}
		else if( inst instanceof MMTSJCPInstruction ) {
			MMTSJCPInstruction linst = (MMTSJCPInstruction) inst;
			MatrixObject mo = ec.getMatrixObject(linst.input1);
			if( mo.isFederated() )
				fedinst = TsmmFEDInstruction.parseInstruction(linst.getInstructionString());
		}
		else if (inst instanceof AggregateUnaryCPInstruction) {
			AggregateUnaryCPInstruction instruction = (AggregateUnaryCPInstruction) inst;
			if( instruction.input1.isMatrix() && ec.containsVariable(instruction.input1) ) {
				MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
				if (mo1.isFederated() && instruction.getAUType() == AggregateUnaryCPInstruction.AUType.DEFAULT)
					fedinst = AggregateUnaryFEDInstruction.parseInstruction(inst.getInstructionString());
			}
		}
		else if (inst instanceof BinaryCPInstruction) {
			BinaryCPInstruction instruction = (BinaryCPInstruction) inst;
			if( (instruction.input1.isMatrix() && ec.getMatrixObject(instruction.input1).isFederated())
				|| (instruction.input2.isMatrix() && ec.getMatrixObject(instruction.input2).isFederated()) ) {
				if(!instruction.getOpcode().equals("append")) //TODO support rbind/cbind
					fedinst = BinaryFEDInstruction.parseInstruction(inst.getInstructionString());
			}
		}
		else if( inst instanceof ParameterizedBuiltinCPInstruction ) {
			ParameterizedBuiltinCPInstruction pinst = (ParameterizedBuiltinCPInstruction) inst;
			if(pinst.getOpcode().equals("replace") && pinst.getTarget(ec).isFederated()) {
				fedinst = ParameterizedBuiltinFEDInstruction.parseInstruction(pinst.getInstructionString());
			}
			else if((pinst.getOpcode().equals("transformdecode") || pinst.getOpcode().equals("transformapply")) &&
				pinst.getTarget(ec).isFederated()) {
				return ParameterizedBuiltinFEDInstruction.parseInstruction(pinst.getInstructionString());
			}
		}
		else if (inst instanceof MultiReturnParameterizedBuiltinCPInstruction) {
			MultiReturnParameterizedBuiltinCPInstruction minst = (MultiReturnParameterizedBuiltinCPInstruction) inst;
			if(minst.getOpcode().equals("transformencode") && minst.input1.isFrame()) {
				CacheableData<?> fo = ec.getCacheableData(minst.input1);
				if(fo.isFederated()) {
					fedinst = MultiReturnParameterizedBuiltinFEDInstruction
						.parseInstruction(minst.getInstructionString());
				}
			}
		}
		else if(inst instanceof ReorgCPInstruction && inst.getOpcode().equals("r'")) {
			ReorgCPInstruction rinst = (ReorgCPInstruction) inst;
			CacheableData<?> mo = ec.getCacheableData(rinst.input1);
			if( mo.isFederated() )
				fedinst = ReorgFEDInstruction.parseInstruction(rinst.getInstructionString());
		}
		
		//set thread id for federated context management
		if( fedinst != null ) {
			fedinst.setTID(ec.getTID());
			return fedinst;
		}
		
		return inst;
	}
	
	public static Instruction checkAndReplaceSP(Instruction inst, ExecutionContext ec) {
		FEDInstruction fedinst = null;
		if (inst instanceof MapmmSPInstruction) {
			// FIXME does not yet work for MV multiplication. SPARK execution mode not supported for federated l2svm
			MapmmSPInstruction instruction = (MapmmSPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if (data instanceof MatrixObject && ((MatrixObject) data).isFederated()) {
				// TODO correct FED instruction string
				fedinst = new AggregateBinaryFEDInstruction(instruction.getOperator(),
					instruction.input1, instruction.input2, instruction.output, "ba+*", "FED...");
			}
		}
		else if (inst instanceof AggregateUnarySPInstruction) {
			AggregateUnarySPInstruction instruction = (AggregateUnarySPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if (data instanceof MatrixObject && ((MatrixObject) data).isFederated())
				fedinst = AggregateUnaryFEDInstruction.parseInstruction(inst.getInstructionString());
		}
		else if (inst instanceof WriteSPInstruction) {
			WriteSPInstruction instruction = (WriteSPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if (data instanceof MatrixObject && ((MatrixObject) data).isFederated()) {
				// Write spark instruction can not be executed for federated matrix objects (tries to get rdds which do
				// not exist), therefore we replace the instruction with the VariableCPInstruction.
				return VariableCPInstruction.parseInstruction(instruction.getInstructionString());
			}
		}
		else if (inst instanceof AppendGAlignedSPInstruction) {
			// TODO other Append Spark instructions
			AppendGAlignedSPInstruction instruction = (AppendGAlignedSPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if (data instanceof MatrixObject && ((MatrixObject) data).isFederated()) {
				fedinst = AppendFEDInstruction.parseInstruction(instruction.getInstructionString());
			}
		}
		//set thread id for federated context management
		if( fedinst != null ) {
			fedinst.setTID(ec.getTID());
			return fedinst;
		}
		
		return inst;
	}
}

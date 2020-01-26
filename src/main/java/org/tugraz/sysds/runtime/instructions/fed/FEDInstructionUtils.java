/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.instructions.fed;

import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.tugraz.sysds.runtime.instructions.spark.AggregateUnarySPInstruction;
import org.tugraz.sysds.runtime.instructions.spark.MapmmSPInstruction;
import org.tugraz.sysds.runtime.instructions.spark.WriteSPInstruction;

public class FEDInstructionUtils {
	public static Instruction checkAndReplaceCP(Instruction inst, ExecutionContext ec) {
		if (inst instanceof AggregateBinaryCPInstruction) {
			AggregateBinaryCPInstruction instruction = (AggregateBinaryCPInstruction) inst;
			if( instruction.input1.isMatrix() && instruction.input2.isMatrix() ) {
				MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
				MatrixObject mo2 = ec.getMatrixObject(instruction.input2);
				if (mo1.isFederated() && mo2.getNumColumns() == 1 || mo1.getNumRows() == 1 && mo2.isFederated()) {
					// currently only vm/mv is supported
					return AggregateBinaryFEDInstruction.parseInstruction(inst.getInstructionString());
				}
			}
		}
		else if (inst instanceof AggregateUnaryCPInstruction) {
			AggregateUnaryCPInstruction instruction = (AggregateUnaryCPInstruction) inst;
			if( instruction.input1.isMatrix() ) {
				MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
				if (mo1.isFederated() && instruction.getAUType() == AggregateUnaryCPInstruction.AUType.DEFAULT)
					return AggregateUnaryFEDInstruction.parseInstruction(inst.getInstructionString());
			}
		}
		return inst;
	}
	
	public static Instruction checkAndReplaceSP(Instruction inst, ExecutionContext ec) {
		if (inst instanceof MapmmSPInstruction) {
			// FIXME does not yet work for MV multiplication. SPARK execution mode not supported for federated l2svm
			MapmmSPInstruction instruction = (MapmmSPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if (data instanceof MatrixObject && ((MatrixObject) data).isFederated()) {
				return new AggregateBinaryFEDInstruction(instruction.getOperator(),
					instruction.input1, instruction.input2, instruction.output, "ba+*", "FED..."); 
				// TODO correct FED instruction string
			}
		}
		else if (inst instanceof AggregateUnarySPInstruction) {
			AggregateUnarySPInstruction instruction = (AggregateUnarySPInstruction) inst;
			MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
			if (mo1.isFederated())
				return AggregateUnaryFEDInstruction.parseInstruction(inst.getInstructionString());
		}
		else if (inst instanceof WriteSPInstruction) {
			WriteSPInstruction instruction = (WriteSPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if (data instanceof MatrixObject && ((MatrixObject) data).isFederated()) {
				// Write spark instruction can not be executed for federeted matrix objects (tries to get rdds which do not exist)
				return VariableCPInstruction.parseInstruction(instruction.getInstructionString());
			}
		}
		return inst;
	}
}

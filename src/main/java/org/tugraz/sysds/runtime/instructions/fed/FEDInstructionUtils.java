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

public class FEDInstructionUtils {
	public static Instruction checkAndReplace(Instruction inst, ExecutionContext ec) {
		if( inst instanceof AggregateBinaryCPInstruction ) {
			AggregateBinaryCPInstruction instruction = (AggregateBinaryCPInstruction) inst;
			MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
			MatrixObject mo2 = ec.getMatrixObject(instruction.input2);
			if (mo1.isFederated() && mo2.getNumColumns() == 1 
				|| mo1.getNumRows() == 1 && mo2.isFederated()) {
				return AggregateBinaryFEDInstruction.parseInstruction(inst.getInstructionString());
			}
		}
		return inst;
	}
}

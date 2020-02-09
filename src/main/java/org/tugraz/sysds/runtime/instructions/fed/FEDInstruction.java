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

import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.matrix.operators.Operator;

public abstract class FEDInstruction extends Instruction {
	
	public enum FEDType {
		Init, AggregateBinary, AggregateUnary, Append
	}
	
	protected final FEDType _fedType;
	protected final Operator _optr;
	
	protected FEDInstruction(FEDType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}
	
	protected FEDInstruction(FEDType type, Operator op, String opcode, String istr) {
		_fedType = type;
		_optr = op;
		instString = istr;
		instOpcode = opcode;
	}
	
	@Override
	public IType getType() {
		return IType.FEDERATED;
	}
	
	public FEDType getFEDInstructionType() {
		return _fedType;
	}
	
	@Override
	public Instruction preprocessInstruction(ExecutionContext ec) {
		return super.preprocessInstruction(ec);
	}
}

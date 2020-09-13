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

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.privacy.PrivacyPropagator;

public abstract class FEDInstruction extends Instruction {
	
	public enum FEDType {
		AggregateBinary,
		AggregateUnary,
		Append,
		Binary,
		Init,
		MultiReturnParameterizedBuiltin,
		ParameterizedBuiltin,
		Tsmm,
		MMChain,
		Reorg,
	}
	
	protected final FEDType _fedType;
	protected long _tid = -1; //main
	
	protected FEDInstruction(FEDType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}
	
	protected FEDInstruction(FEDType type, Operator op, String opcode, String istr) {
		super(op);
		_fedType = type;
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
	
	public long getTID() {
		return _tid;
	}
	
	public void setTID(long tid) {
		_tid = tid;
	}
	
	@Override
	public Instruction preprocessInstruction(ExecutionContext ec) {
		Instruction tmp = super.preprocessInstruction(ec);
		tmp = PrivacyPropagator.preprocessInstruction(tmp, ec);
		return tmp;
	}
}

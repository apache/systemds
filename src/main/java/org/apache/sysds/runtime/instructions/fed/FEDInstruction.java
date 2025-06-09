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

import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.operators.Operator;

public abstract class FEDInstruction extends Instruction {

	public enum FEDType {
		AggregateBinary,
		AggregateUnary,
		AggregateTernary,
		Append,
		Binary,
		Cast,
		CentralMoment,
		Checkpoint,
		Covariance,
		CSVReblock,
		Ctable,
		CumulativeAggregate,
		CumsumOffset,
		Init,
		MultiReturnParameterizedBuiltin,
		MMChain,
		MAPMM,
		MatrixIndexing,
		Ternary,
		Tsmm,
		ParameterizedBuiltin,
		Quaternary,
		QSort,
		QPick,
		Reblock,
		Reorg,
		Reshape,
		SpoofFused,
		Unary
	}
	
	public enum FederatedOutput {
		FOUT, // forced federated output 
		LOUT, // forced local output (consolidated in CP)
		NONE; // runtime heuristics
		public boolean isForcedFederated() {
			return this == FOUT;
		}
		public boolean isForcedLocal() {
			return this == LOUT;
		}
		public boolean isForced(){
			return this == FOUT || this == LOUT;
		}
	}

	protected final FEDType _fedType;
	protected long _tid = -1; //main
	protected FederatedOutput _fedOut = FederatedOutput.NONE;

	protected FEDInstruction(FEDType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}

	protected FEDInstruction(FEDType type, Operator op, String opcode, String istr) {
		this(type, op, opcode, istr, FederatedOutput.NONE);
	}

	protected FEDInstruction(FEDType type, Operator op, String opcode, String istr, FederatedOutput fedOut) {
		super(op);
		_fedType = type;
		instString = istr;
		instOpcode = opcode;
		_fedOut = fedOut;
		
		// Debug output to terminal
		System.out.println("[FED-CREATE] " + this.getClass().getSimpleName() + 
			" | Type: " + _fedType + 
			" | Opcode: " + instOpcode + 
			" | Output: " + _fedOut + 
			" | TID: " + _tid + 
			" | Thread: " + Thread.currentThread().getName());
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
}

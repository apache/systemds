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

package org.apache.sysml.runtime.instructions.cp;

import org.apache.sysml.lops.runtime.RunMRJobs;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.CPInstructionParser;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.operators.Operator;


public abstract class CPInstruction extends Instruction 
{
	public enum CPINSTRUCTION_TYPE { INVALID, 
		AggregateUnary, AggregateBinary, AggregateTernary, ArithmeticBinary, 
		Ternary, Quaternary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, 
		BuiltinMultiple, MultiReturnParameterizedBuiltin, ParameterizedBuiltin, MultiReturnBuiltin, 
		Builtin, Reorg, RelationalBinary, Variable, External, Append, Rand, QSort, QPick, 
		MatrixIndexing, MMTSJ, PMMJ, MMChain, MatrixReshape, Partition, Compression, SpoofFused,
		StringInit, CentralMoment, Covariance, UaggOuterChain, Convolution };
	
	protected CPINSTRUCTION_TYPE _cptype;
	protected Operator _optr;
	
	protected boolean _requiresLabelUpdate = false;
	
	public CPInstruction(String opcode, String istr) {
		type = INSTRUCTION_TYPE.CONTROL_PROGRAM;
		instString = istr;
		
		//prepare opcode and update requirement for repeated usage
		instOpcode = opcode;
		_requiresLabelUpdate = super.requiresLabelUpdate();
	}
	
	public CPInstruction(Operator op, String opcode, String istr) {
		this(opcode, istr);
		_optr = op;
	}
	
	public CPINSTRUCTION_TYPE getCPInstructionType() {
		return _cptype;
	}
	
	@Override
	public boolean requiresLabelUpdate()
	{
		return _requiresLabelUpdate;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	@Override
	public Instruction preprocessInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		//default preprocess behavior (e.g., debug state)
		Instruction tmp = super.preprocessInstruction(ec);
		
		//instruction patching
		if( tmp.requiresLabelUpdate() ) //update labels only if required
		{
			//note: no exchange of updated instruction as labels might change in the general case
			String updInst = RunMRJobs.updateLabels(tmp.toString(), ec.getVariables());
			tmp = CPInstructionParser.parseSingleInstruction(updInst);
		}

		return tmp;
	}

	@Override 
	public abstract void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException;
}

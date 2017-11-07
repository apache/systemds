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
	public enum CPType { INVALID, 
		AggregateUnary, AggregateBinary, AggregateTernary, ArithmeticBinary, 
		Ternary, Quaternary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, BuiltinNary, 
		MultiReturnParameterizedBuiltin, ParameterizedBuiltin, MultiReturnBuiltin, 
		Builtin, Reorg, RelationalBinary, Variable, External, Append, Rand, QSort, QPick, 
		MatrixIndexing, MMTSJ, PMMJ, MMChain, MatrixReshape, Partition, Compression, SpoofFused,
		StringInit, CentralMoment, Covariance, UaggOuterChain, Convolution }
	
	protected final CPType _cptype;
	protected final Operator _optr;
	protected final boolean _requiresLabelUpdate;
	
	// Generic miscellaneous timers that are applicable to all CP (and few SP) instructions 
	public final static String MISC_TIMER_GET_SPARSE_MB =          		"aqrs";	// time spent in bringing input sparse matrix block
	public final static String MISC_TIMER_GET_DENSE_MB =          		"aqrd";	// time spent in bringing input dense matrix block
	public final static String MISC_TIMER_ACQ_MODIFY_SPARSE_MB =        "aqms";	// time spent in bringing output sparse matrix block
	public final static String MISC_TIMER_ACQ_MODIFY_DENSE_MB =         "aqmd";	// time spent in bringing output dense matrix block
	public final static String MISC_TIMER_RELEASE_INPUT_MB =      		"rlsi";	// time spent in release input matrix block
	public final static String MISC_TIMER_RELEASE_EVICTION =			"rlsev";// time spent in buffer eviction of release operation
	public final static String MISC_TIMER_RELEASE_BUFF_WRITE =			"rlswr";// time spent in buffer write in release operation
	public final static String MISC_TIMER_SPARSE_TO_DENSE =				"s2d";  // time spent in sparse to dense conversion
	public final static String MISC_TIMER_DENSE_TO_SPARSE =				"d2s";  // time spent in sparse to dense conversion
	public final static String MISC_TIMER_RECOMPUTE_NNZ =				"rnnz"; // time spent in recompute non-zeroes
	
	// Instruction specific miscellaneous timers that were found as potential bottlenecks in one of performance analysis.
	// SystemML committers have to be judicious about adding them by weighing the tradeoffs between reuse in future analysis and unnecessary overheads.
	public final static String MISC_TIMER_CSR_LIX_COPY =				"csrlix";// time spent in CSR-specific method to address performance issues due to repeated re-shifting on update-in-place.
	public final static String MISC_TIMER_LIX_COPY =					"lixcp";// time spent in range copy

	protected CPInstruction(CPType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}

	protected CPInstruction(CPType type, Operator op, String opcode, String istr) {
		_cptype = type;
		_optr = op;
		super.type = IType.CONTROL_PROGRAM;
		instString = istr;

		// prepare opcode and update requirement for repeated usage
		instOpcode = opcode;
		_requiresLabelUpdate = super.requiresLabelUpdate();
	}

	public CPType getCPInstructionType() {
		return _cptype;
	}
	
	@Override
	public boolean requiresLabelUpdate() {
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

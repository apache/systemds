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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.SPInstructionParser;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.fed.FEDInstructionUtils;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.utils.Statistics;

public abstract class SPInstruction extends Instruction {

	public enum SPType { 
		MAPMM, MAPMMCHAIN, CPMM, RMM, TSMM, TSMM2, PMM, ZIPMM, PMAPMM, //matrix multiplication instructions  
		MatrixIndexing, Reorg, Binary, Ternary,
		AggregateUnary, AggregateTernary, Reblock, CSVReblock, 
		Builtin, Unary, BuiltinNary, MultiReturnBuiltin, Checkpoint, Compression, Cast,
		CentralMoment, Covariance, QSort, QPick, 
		ParameterizedBuiltin, MAppend, RAppend, GAppend, GAlignedAppend, Rand, 
		MatrixReshape, Ctable, Quaternary, CumsumAggregate, CumsumOffset, BinUaggChain, UaggOuterChain, 
		Write, SpoofFused, Dnn
	}

	protected final SPType _sptype;
	protected final boolean _requiresLabelUpdate;

	protected SPInstruction(SPType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}

	protected SPInstruction(SPType type, Operator op, String opcode, String istr) {
		super(op);
		_sptype = type;
		instString = istr;

		// prepare opcode and update requirement for repeated usage
		instOpcode = opcode;
		_requiresLabelUpdate = super.requiresLabelUpdate();
	}
	
	@Override
	public IType getType() {
		return IType.SPARK;
	}

	public SPType getSPInstructionType() {
		return _sptype;
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
	public Instruction preprocessInstruction(ExecutionContext ec) {
		//default pre-process behavior (e.g., debug state)
		Instruction tmp = super.preprocessInstruction(ec);
		
		//instruction patching
		if( tmp.requiresLabelUpdate() ) //update labels only if required
		{
			//note: no exchange of updated instruction as labels might change in the general case
			String updInst = CPInstruction.updateLabels(tmp.toString(), ec.getVariables());
			tmp = SPInstructionParser.parseSingleInstruction(updInst);
		}
		
		//robustness federated instructions (runtime assignment)
		tmp = FEDInstructionUtils.checkAndReplaceSP(tmp, ec);
		
		return tmp;
	}

	@Override 
	public abstract void processInstruction(ExecutionContext ec);

	@Override
	public void postprocessInstruction(ExecutionContext ec) {
		//maintain statistics
		Statistics.incrementNoOfExecutedSPInst();
		
		//default post-process behavior
		super.postprocessInstruction(ec);
	}
}

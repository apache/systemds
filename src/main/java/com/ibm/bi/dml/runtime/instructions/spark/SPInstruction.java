/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.api.MLContext;
import com.ibm.bi.dml.api.MLContextProxy;
import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SPInstructionParser;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * 
 */
public abstract class SPInstruction extends Instruction 
{
	
	public enum SPINSTRUCTION_TYPE { 
		MAPMM, MAPMMCHAIN, CPMM, RMM, TSMM, PMM, //matrix multiplication instructions  
		MatrixIndexing, Reorg, ArithmeticBinary, RelationalBinary, AggregateUnary, AggregateTernary, Reblock, CSVReblock, 
		Builtin, BuiltinUnary, BuiltinBinary, Checkpoint, 
		CentralMoment, Covariance, QSort, QPick,
		ParameterizedBuiltin, MAppend, RAppend, GAppend, GAlignedAppend, Rand, 
		MatrixReshape, Ternary, Quaternary, CumsumAggregate, CumsumOffset, BinUaggChain, UaggOuterChain, 
		Write, INVALID, 
	};
	
	protected SPINSTRUCTION_TYPE _sptype;
	protected Operator _optr;
	
	protected boolean _requiresLabelUpdate = false;
	
	public SPInstruction(String opcode, String istr) {
		type = INSTRUCTION_TYPE.SPARK;
		instString = istr;
		instOpcode = opcode;
		
		//update requirement for repeated usage
		_requiresLabelUpdate = super.requiresLabelUpdate();
	}
	
	public SPInstruction(Operator op, String opcode, String istr) {
		this(opcode, istr);
		_optr = op;
	}
	
	public SPINSTRUCTION_TYPE getSPInstructionType() {
		return _sptype;
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
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//default pre-process behavior (e.g., debug state)
		Instruction tmp = super.preprocessInstruction(ec);
		
		//instruction patching
		if( tmp.requiresLabelUpdate() ) //update labels only if required
		{
			//note: no exchange of updated instruction as labels might change in the general case
			String updInst = RunMRJobs.updateLabels(tmp.toString(), ec.getVariables());
			tmp = SPInstructionParser.parseSingleInstruction(updInst);
		}
		

		//spark-explain-specific handling of current instructions 
		//This only relevant for ComputationSPInstruction as in postprocess we call setDebugString which is valid only for ComputationSPInstruction
		MLContext mlCtx = MLContextProxy.getActiveMLContext();
		if(    tmp instanceof ComputationSPInstruction 
			&& mlCtx != null && mlCtx.getMonitoringUtil() != null 
			&& ec instanceof SparkExecutionContext ) 
		{
			mlCtx.getMonitoringUtil().addCurrentInstruction((SPInstruction)tmp);
			MLContextProxy.setInstructionForMonitoring(tmp);
		}
		
		return tmp;
	}

	@Override 
	public abstract void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException, DMLUnsupportedOperationException;

	@Override
	public void postprocessInstruction(ExecutionContext ec)
			throws DMLRuntimeException 
	{
		//spark-explain-specific handling of current instructions
		MLContext mlCtx = MLContextProxy.getActiveMLContext();
		if(    this instanceof ComputationSPInstruction 
			&& mlCtx != null && mlCtx.getMonitoringUtil() != null
			&& ec instanceof SparkExecutionContext ) 
		{
			SparkExecutionContext sec = (SparkExecutionContext) ec;
			sec.setDebugString(this, ((ComputationSPInstruction) this).getOutputVariableName());
			mlCtx.getMonitoringUtil().removeCurrentInstruction(this);
		}
		
		//maintain statistics
		Statistics.incrementNoOfExecutedSPInst();
		
		//default post-process behavior
		super.postprocessInstruction(ec);
	}
	
}

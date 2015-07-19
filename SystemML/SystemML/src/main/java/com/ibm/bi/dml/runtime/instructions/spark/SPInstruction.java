/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.SPInstructionParser;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.utils.Explain;

/**
 * 
 * 
 */
public abstract class SPInstruction extends Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum SPINSTRUCTION_TYPE { 
		MAPMM, MAPMMCHAIN, CPMM, RMM, TSMM, PMM, //matrix multiplication instructions  
		MatrixIndexing, Reorg, ArithmeticBinary, RelationalBinary, AggregateUnary, Reblock, CSVReblock, 
		Builtin, BuiltinUnary, BuiltinBinary, Sort, Variable, Checkpoint, CentralMoment, Covariance,
		ParameterizedBuiltin, MAppend, RAppend, GAppend, GAlignedAppend, Rand, 
		MatrixReshape, Ternary, Quaternary, CumsumAggregate, CumsumOffset,
		INVALID, 
	};
	
	protected SPINSTRUCTION_TYPE _sptype;
	protected Operator _optr;
	
	protected boolean _requiresLabelUpdate = false;
	
	// Fields that help monitor spark execution
	protected String           debugString;
	public ArrayList<Integer> stageSubmittedIds = new ArrayList<Integer>();
	public ArrayList<Integer> stageCompletedIds = new ArrayList<Integer>();
		
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
		//TODO why is this only relevant for ComputationSPInstruction  
		if(    tmp instanceof ComputationSPInstruction 
			&& Explain.PRINT_EXPLAIN_WITH_LINEAGE 
			&& ec instanceof SparkExecutionContext ) 
		{
			SparkExecutionContext sec = (SparkExecutionContext) ec;
			if(sec.getSparkListener() != null)
				sec.getSparkListener().addCurrentInstruction((SPInstruction)tmp);
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
		if(    this instanceof ComputationSPInstruction 
			&& Explain.PRINT_EXPLAIN_WITH_LINEAGE
			&& ec instanceof SparkExecutionContext ) 
		{
			SparkExecutionContext sec = (SparkExecutionContext) ec;
			sec.setDebugString(this, ((ComputationSPInstruction) this).getOutputVariableName());
			if(sec.getSparkListener() != null)
				sec.getSparkListener().removeCurrentInstruction(this);
		}
		
		//default post-process behavior
		super.postprocessInstruction(ec);
	}
	
	///////////////////////////////////////
	// debug functionality for monitoring
	////////
	
	/**
	 * 
	 * @param al
	 * @return
	 */
	private String getStringFromArrayList(ArrayList<Integer> al) 
	{
		StringBuilder sb = new StringBuilder("");
		for(Integer i : al) {
			if( sb.length() > 0 )
				sb.append(", ");
			sb.append(i);
		}
		
		return sb.toString();
	}
	
	/**
	 * 
	 * @return
	 */
	public String getSparkInfo() 
	{
		return "\nStage Submitted IDs:[" + getStringFromArrayList(stageSubmittedIds) + "]" +
				"\nStage Completed IDs:[" + getStringFromArrayList(stageCompletedIds) + "]";
	}
	

	/**
	 * Do not call this method directly, instead go through SparkUtils.setLineageInfoForExplain
	 *
	 * @param debugString
	 */
	public void setDebugString(String debugString) {
		if(this.debugString == null) {
			this.debugString = debugString;
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public String getDebugString() {
		return debugString;
	}
	
}

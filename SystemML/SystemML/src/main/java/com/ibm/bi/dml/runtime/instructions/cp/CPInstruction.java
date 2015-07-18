/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;


public abstract class CPInstruction extends Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum CPINSTRUCTION_TYPE { INVALID, AggregateUnary, AggregateBinary, AggregateTernary, ArithmeticBinary, Ternary, Quaternary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, ParameterizedBuiltin, MultiReturnBuiltin, Builtin, Reorg, RelationalBinary, File, Variable, External, Append, Rand, Sort, MatrixIndexing, MMTSJ, PMMJ, MMChain, MatrixReshape, Partition, StringInit, CentralMoment }; 
	
	protected CPINSTRUCTION_TYPE _cptype;
	protected Operator _optr;
	
	protected String _opcode = null;
	protected boolean _requiresLabelUpdate = false;
	
	public CPInstruction(String opcode, String istr) {
		type = INSTRUCTION_TYPE.CONTROL_PROGRAM;
		instString = istr;
		
		//prepare opcode and update requirement for repeated usage
		_opcode = opcode;
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
	
	public String getOpcode()
	{
		return _opcode;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	@Override
	public Instruction preprocessInstruction(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException 
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
			throws DMLRuntimeException, DMLUnsupportedOperationException;
}

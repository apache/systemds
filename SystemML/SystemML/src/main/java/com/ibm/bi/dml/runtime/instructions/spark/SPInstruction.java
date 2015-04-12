/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * The main integration decisions are as follows:
 * - A new exec type has been added in rtplatform (so as to allow for testing). Also an exec type has been added in lop properties.
 * - The Lops and Hops are reused as much as possible.
 * - The Spark instructions have spark exec type and have controlprogram exec location. This allows us to make minimal changes in Dag.java
 * - The Spark instructions are compiled in similar way as CP, through SPInstructionParser, which is called when exec type of Lop is spark.
 * 
 */
public abstract class SPInstruction extends Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	// public enum SPINSTRUCTION_TYPE { INVALID, AggregateUnary, AggregateBinary, AggregateTertiary, ArithmeticBinary, Tertiary, BooleanBinary, BooleanUnary, BuiltinBinary, BuiltinUnary, ParameterizedBuiltin, MultiReturnBuiltin, Builtin, Reorg, RelationalBinary, File, Variable, External, Append, Rand, Sort, MatrixIndexing, MMTSJ, PMMJ, MatrixReshape, Partition, StringInit }; 
	public enum SPINSTRUCTION_TYPE { 
		INVALID, MMCJ, MapMult, MatrixIndexing, Reorg, ArithmeticBinary, RelationalBinary, AggregateUnary, Reblock, CSVReblock, Builtin, BuiltinUnary, BuiltinBinary, Sort, Variable, Checkpoint };
	
		protected SPINSTRUCTION_TYPE _sptype;
	protected Operator _optr;
	
	protected String _opcode = null;
	protected boolean _requiresLabelUpdate = false;
	
	public SPInstruction(String opcode, String istr) {
		type = INSTRUCTION_TYPE.SPARK;
		instString = istr;
		
		//prepare opcode and update requirement for repeated usage
		_opcode = opcode;
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
	
	public String getOpcode()
	{
		return _opcode;
	}
	
	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		return null;
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		return null;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	
	
	/**
	 * This method should be used to execute the instruction. It's abstract to force 
	 * subclasses to override it. 
	 * 
	 * @param ec
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	public abstract void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;

}

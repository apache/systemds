/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import java.util.HashMap;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


/**
 * Defines a LOP for functions.
 * 
 */
public class ParameterizedBuiltin extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum OperationTypes { INVALID, CDF, RMEMPTY };
	
	OperationTypes operation;
			
	//private Operation _operation;
	private HashMap<String, Lop> _inputParams;

	/**
	 * Creates a new builtin function LOP.
	 * 
	 * @param target
	 *            target identifier
	 * @param params
	 *            parameter list
	 * @param inputParameters
	 *            list of input LOPs
	 * @param function
	 *            builtin function
	 * @param numRows
	 *            number of resulting rows
	 * @param numCols
	 *            number of resulting columns
	 */
	public ParameterizedBuiltin(HashMap<String, Lop> 
				inputParametersLops, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.ParameterizedBuiltin, dt, vt);
		operation = op;
		
		for (Lop lop : inputParametersLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		/*
		 * This lop is executed in control program. 
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties(inputs, ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
	}
	
	public ParameterizedBuiltin(ExecType et, HashMap<String, Lop> 
		       inputParametersLops, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.ParameterizedBuiltin, dt, vt);
		operation = op;
		
		for (Lop lop : inputParametersLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		/*
		 * This lop is executed in control program. 
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
	}

	//@Override
	public String getInstructions(String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		switch(operation) {
		case CDF:
			sb.append( "cdf" );
			sb.append( OPERAND_DELIMITOR );
			
			for ( String s : _inputParams.keySet() ) 
			{	
				sb.append( s );
				sb.append( NAME_VALUE_SEPARATOR );
				
				// get the value/label of the scalar input associated with name "s"
				Lop iLop = _inputParams.get(s);
				sb.append( iLop.prepScalarLabel() );
				sb.append( OPERAND_DELIMITOR );
			}
			break;
			
		case RMEMPTY:
			sb.append("rmempty");
			sb.append(OPERAND_DELIMITOR);
			
			for ( String s : _inputParams.keySet() ) {
				
				sb.append(s);
				sb.append(NAME_VALUE_SEPARATOR);
				
				// instruction patching not required because rmEmpty always executed as CP/CP_FILE
				Lop iLop = _inputParams.get(s);
				sb.append(iLop.getOutputParameters().getLabel());
				sb.append(OPERAND_DELIMITOR);
			}
			break;
		default:
			throw new LopsException(this.printErrorLocation() + "In ParameterizedBuiltin Lop, Unknown operation: " + operation);
		}
		
		sb.append(this.prepOutputOperand(output));
		
		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(operation.toString());

		if (getInputs().size() > 0)
			sb.append("(");
		for (Lop cur : getInputs()) {
			sb.append(cur.toString());
		}
		if (getInputs().size() > 0)
			sb.append(") ");

		sb.append(" ; num_rows=" + this.getOutputParameters().getNum_rows());
		sb.append(" ; num_cols=" + this.getOutputParameters().getNum_cols());
		sb.append(" ; format=" + this.getOutputParameters().getFormat());
		sb.append(" ; blocked=" + this.getOutputParameters().isBlocked_representation());
		return sb.toString();
	}

}

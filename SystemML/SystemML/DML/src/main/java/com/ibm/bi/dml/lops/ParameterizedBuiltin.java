/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import java.util.HashMap;

import com.ibm.bi.dml.hops.HopsException;
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum OperationTypes { INVALID, CDF, RMEMPTY, REPLACE };
	
	private OperationTypes _operation;
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
		_operation = op;
		
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
		       inputParametersLops, OperationTypes op, DataType dt, ValueType vt) throws HopsException 
	{
		super(Lop.Type.ParameterizedBuiltin, dt, vt);
		_operation = op;
		
		for (Lop lop : inputParametersLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = inputParametersLops;
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		ExecLocation eloc = null;
		if( _operation == OperationTypes.REPLACE && et==ExecType.MR )
		{
			eloc = ExecLocation.MapOrReduce;
			//lps.addCompatibility(JobType.CSV_REBLOCK);
			//lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.REBLOCK);
		}
		else if( _operation == OperationTypes.RMEMPTY && et==ExecType.MR )
		{
			eloc = ExecLocation.Reduce;
			//lps.addCompatibility(JobType.CSV_REBLOCK);
			//lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.REBLOCK);
			breaksAlignment=true;
		}
		else if ( et == ExecType.SPARK )  {
			throw new HopsException("Parameterized for ParameterizedBuiltinOp not implemented for Spark");
		}
		else //executed in CP / CP_FILE
		{
			eloc = ExecLocation.ControlProgram;
			lps.addCompatibility(JobType.INVALID);
		}
		lps.setProperties(inputs, et, eloc, breaksAlignment, aligner, definesMRJob);
	}

	@Override
	public String getInstructions(String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		switch(_operation) 
		{
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
					
					// get the value/label of the scalar input associated with name "s"
					Lop iLop = _inputParams.get(s);
					if( s.equals("target") )
						sb.append(iLop.getOutputParameters().getLabel());
					else
						sb.append( iLop.prepScalarLabel() );
					
					sb.append(OPERAND_DELIMITOR);
				}
				break;
			
			case REPLACE:
				sb.append( "replace" );
				sb.append( OPERAND_DELIMITOR );
				
				for ( String s : _inputParams.keySet() ) 
				{	
					sb.append( s );
					sb.append( NAME_VALUE_SEPARATOR );
					
					// get the value/label of the scalar input associated with name "s"
					Lop iLop = _inputParams.get(s);
					if( s.equals("target") )
						sb.append(iLop.getOutputParameters().getLabel());
					else
						sb.append( iLop.prepScalarLabel() );
					sb.append( OPERAND_DELIMITOR );
				}
				break;
			
			default:
				throw new LopsException(this.printErrorLocation() + "In ParameterizedBuiltin Lop, Unknown operation: " + _operation);
		}
		
		sb.append(this.prepOutputOperand(output));
		
		return sb.toString();
	}

	@Override 
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		switch(_operation) 
		{
			case REPLACE:
			{
				sb.append( "replace" );
				sb.append( OPERAND_DELIMITOR );
		
				Lop iLop = _inputParams.get("target");
				int pos = getInputs().indexOf(iLop);
				int index = (pos==0)? input_index1 : (pos==1)? input_index2 : input_index3;
				//input_index
				sb.append(prepInputOperand(index));
				sb.append( OPERAND_DELIMITOR );
				
				Lop iLop2 = _inputParams.get("pattern");
				sb.append(iLop2.prepScalarLabel());
				sb.append( OPERAND_DELIMITOR );
				
				Lop iLop3 = _inputParams.get("replacement");
				sb.append(iLop3.prepScalarLabel());
				sb.append( OPERAND_DELIMITOR );
				
				break;
			}	
				
			default:
				throw new LopsException(this.printErrorLocation() + "In ParameterizedBuiltin Lop, Unknown operation: " + _operation);
		}
		
		sb.append( prepOutputOperand(output_index));
		
		return sb.toString();
	}

	@Override 
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		switch(_operation) 
		{
			case RMEMPTY:
			{
				sb.append("rmempty");
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop1 = _inputParams.get("target");
				int pos1 = getInputs().indexOf(iLop1);
				int index1 = (pos1==0)? input_index1 : (pos1==1)? input_index2 : (pos1==2)? input_index3 : input_index4;
				sb.append(prepInputOperand(index1));
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop2 = _inputParams.get("offset");
				int pos2 = getInputs().indexOf(iLop2);
				int index2 = (pos2==0)? input_index1 : (pos2==1)? input_index2 : (pos1==2)? input_index3 : input_index4;
				sb.append(prepInputOperand(index2));
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop3 = _inputParams.get("maxdim");
				sb.append( iLop3.prepScalarLabel() );
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop4 = _inputParams.get("margin");
				sb.append( iLop4.prepScalarLabel() );
				
				sb.append( OPERAND_DELIMITOR );
				
				break;
			}
				
			default:
				throw new LopsException(this.printErrorLocation() + "In ParameterizedBuiltin Lop, Unknown operation: " + _operation);
		}
		
		sb.append( prepOutputOperand(output_index));
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(_operation.toString());

		if( !getInputs().isEmpty() )
			sb.append("(");
		for (Lop cur : getInputs()) {
			sb.append(cur.toString());
		}
		if( !getInputs().isEmpty() )
			sb.append(") ");

		sb.append(" ; num_rows=" + this.getOutputParameters().getNumRows());
		sb.append(" ; num_cols=" + this.getOutputParameters().getNumCols());
		sb.append(" ; format=" + this.getOutputParameters().getFormat());
		sb.append(" ; blocked=" + this.getOutputParameters().isBlocked());
		return sb.toString();
	}

}

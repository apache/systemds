package com.ibm.bi.dml.lops;

import java.util.HashMap;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Defines a LOP for functions.
 * 
 */
public class ParameterizedBuiltin extends Lops {

	public enum OperationTypes { INVALID, CDF, RMEMPTY };
	
	OperationTypes operation;
			
	//private Operation _operation;
	private HashMap<String, Lops> _inputParams;

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
	public ParameterizedBuiltin(HashMap<String, Lops> 
				inputParametersLops, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.ParameterizedBuiltin, dt, vt);
		operation = op;
		
		for (Lops lop : inputParametersLops.values()) {
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
	
	public ParameterizedBuiltin(ExecType et, HashMap<String, Lops> 
		       inputParametersLops, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lops.Type.ParameterizedBuiltin, dt, vt);
		operation = op;
		
		for (Lops lop : inputParametersLops.values()) {
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
		sb.append( Lops.OPERAND_DELIMITOR );

		switch(operation) {
		case CDF:
			sb.append( "cdf" );
			sb.append( OPERAND_DELIMITOR );
			
			for ( String s : _inputParams.keySet() ) 
			{	
				sb.append( s );
				sb.append( NAME_VALUE_SEPARATOR );
				
				// get the value/label of the scalar input associated with name "s"
				Lops iLop = _inputParams.get(s);
				if ( iLop.getExecLocation() == ExecLocation.Data 
						&& ((Data)iLop).isLiteral() ) {
					sb.append( iLop.getOutputParameters().getLabel() );
				}
				else {
					sb.append( "##" );
					sb.append( iLop.getOutputParameters().getLabel() );
					sb.append( "##" );
				}
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
				Lops iLop = _inputParams.get(s);
				//if ( iLop.getExecLocation() == ExecLocation.Data 
				//		&& ((Data)iLop).isLiteral() ) 
					sb.append(iLop.getOutputParameters().getLabel());
				//else 
				//	inst.append("##").append(iLop.getOutputParameters().getLabel()).append("##");
				sb.append(OPERAND_DELIMITOR);
			}
			break;
		default:
			throw new LopsException(this.printErrorLocation() + "In ParameterizedBuiltin Lop, Unknown operation: " + operation);
		}
		
		sb.append(output);
		sb.append(DATATYPE_PREFIX);
		sb.append(get_dataType());
		sb.append(VALUETYPE_PREFIX);
		sb.append(get_valueType());
		
		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(operation.toString());

		if (getInputs().size() > 0)
			sb.append("(");
		for (Lops cur : getInputs()) {
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

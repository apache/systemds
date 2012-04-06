package dml.lops;

import java.util.ArrayList;
import java.util.HashMap;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.OutputParameters.Format;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;
import dml.parser.GenericFunctionDoubleParam;
import dml.parser.GenericFunctionLongParam;
import dml.parser.GenericFunctionParam;
import dml.parser.GenericFunctionStringParam;
import dml.parser.GenericFunctionVarParam;
import dml.parser.GenericFunction;

/**
 * <p>Defines a new LOP for builtin functions.</p>
 * 
 * @author schnetter
 */
public class BuiltinFunction extends Lops
{
	private HashMap<String, GenericFunctionParam> params;
	private GenericFunction function;
	
	
	/**
	 * <p>Creates a new builtin function LOP.</p>
	 * 
	 * @param target target identifier
	 * @param params parameter list
	 * @param inputs list of input LOPs
	 * @param function builtin function
	 * @param numRows number of resulting rows
	 * @param numCols number of resulting columns
	 */
	public BuiltinFunction(String target, HashMap<String, GenericFunctionParam> params,
			ArrayList<Lops> inputs, GenericFunction genericFunction,
			long numRows, long numCols, boolean blocked, Format format, ExecLocation execLoc, DataType dt, ValueType vt)
	{
		super(Lops.Type.GenericFunctionLop, dt, vt);
		this.params = params;
		this.function = genericFunction;
		this.getOutputParameters().num_rows = new Long(numRows);
		this.getOutputParameters().num_cols = new Long(numCols);
		this.getOutputParameters().setFormat(format);
		this.getOutputParameters().blocked_representation = blocked;

		/*
		 * This lop can be executed in all jobs except PARTITION, SORT_KEYS, STANDALONE
		 */
		lps.addCompatibility(JobType.ANY);
		lps.removeCompatibility(JobType.PARTITION);
		lps.removeCompatibility(JobType.SORT);
		lps.removeCompatibility(JobType.COMBINE);
		
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		// Currently, none of the builtin functions break alignment
		// TODO: set to execute always in MR.. check it!
		this.lps.setProperties(ExecType.MR, execLoc, breaksAlignment, aligner, definesMRJob);

		// TODO: add block size
//		this.getOutputParameters().num_rows_per_block = new Long(GenericFunction.BLOCK_ROW_SIZE);
//		this.getOutputParameters().num_cols_per_block = new Long(GenericFunction.BLOCK_COL_SIZE);
		
		for(Lops lop : inputs)
		{
			this.addInput(lop);
			lop.addOutput(this);
		}
	}
	
	/**
	 * <p>Returns the builtin function.</p>
	 * 
	 * @return builtin function
	 */
	public GenericFunction getFunction()
	{
		return function;
	}
	
	/**
	 * <p>Returns the value of a parameter in a string representation.</p>
	 * 
	 * @param paramName parameter name
	 * @return parameter value
	 */
	public String getParamValue(String paramName)
	{
		if(params.containsKey(paramName))
		{
			GenericFunctionParam param = params.get(paramName);
			switch(param.getParamType())
			{
			case INT:
				return Long.toString(((GenericFunctionLongParam) param).getParamValue());
			case DOUBLE:
				return Double.toString(((GenericFunctionDoubleParam) param).getParamValue());
			case STRING:
				return ((GenericFunctionStringParam) param).getParamValue();
			case VAR:
				return ((GenericFunctionVarParam) param).getParamValue().getName();
			}
		}
		
		return null;
	}
	
	@Override
	public String getInstructions(int input_index, int output_index)
	{
		String opString = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		
		switch(function)
		{
		case RAND:
			opString += "bfrand"; break;
		default:
			throw new UnsupportedOperationException("Instruction is not defined for Builtin function " + function);
		}
		
		String inst = new String("");
		inst += opString + OPERAND_DELIMITOR + 
				input_index + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output_index + VALUETYPE_PREFIX + this.get_valueType() ;
		
		return inst;
	}
	
	/**
	 * <p>Returns all parameters in a string representation.</p>
	 * 
	 * @return string representation of parameters
	 */
	protected String getInstructionsParams()
	{
		StringBuilder sb = new StringBuilder();
		for(GenericFunctionParam param : params.values())
		{
			switch(param.getParamType())
			{
			case INT:
				sb.append(" " + param.getParamName() + "=" + ((GenericFunctionLongParam) param).getParamValue());
				break;
			case DOUBLE:
				sb.append(" " + param.getParamName() + "=" + ((GenericFunctionDoubleParam) param).getParamValue());
				break;
			case STRING:
				sb.append(" " + param.getParamName() + "=" + ((GenericFunctionStringParam) param).getParamValue());
				break;
			case VAR:
				sb.append(" " + param.getParamName() + "=" +
						((GenericFunctionVarParam) param).getParamValue().getName());
				break;
			}
		}
		
		return sb.toString();
	}

	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append(function.getFunctionName());
		for(GenericFunctionParam param : params.values())
		{
			sb.append(" " + param.toString());
		}
		sb.append(" ; num_rows=" + this.getOutputParameters().getNum_rows());
		sb.append(" ; num_cols=" + this.getOutputParameters().getNum_cols());
		sb.append(" ; format=" + this.getOutputParameters().getFormat());
		sb.append(" ; blocked=" + this.getOutputParameters().isBlocked_representation());
		return sb.toString();
	}
}

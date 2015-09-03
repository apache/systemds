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

package com.ibm.bi.dml.lops;

import java.util.HashMap;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.ParameterizedBuiltinFunctionExpression;


/**
 * Defines a LOP for functions.
 * 
 */
public class ParameterizedBuiltin extends Lop 
{
	
	public enum OperationTypes { 
		INVALID, CDF, INVCDF, RMEMPTY, REPLACE, REXPAND, 
		PNORM, QNORM, PT, QT, PF, QF, PCHISQ, QCHISQ, PEXP, QEXP,
		TRANSFORM
	};
	
	private OperationTypes _operation;
	private HashMap<String, Lop> _inputParams;
	private boolean _bRmEmptyBC;

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
	public ParameterizedBuiltin(HashMap<String, Lop> paramLops, OperationTypes op, DataType dt, ValueType vt) 
	{
		super(Lop.Type.ParameterizedBuiltin, dt, vt);
		_operation = op;
		
		for (Lop lop : paramLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = paramLops;
		
		/*
		 * This lop is executed in control program. 
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties(inputs, ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
	}
	
	public ParameterizedBuiltin(HashMap<String, Lop> paramLops, OperationTypes op, DataType dt, ValueType vt, ExecType et) 
		throws HopsException 
	{
		super(Lop.Type.ParameterizedBuiltin, dt, vt);
		_operation = op;
		
		for (Lop lop : paramLops.values()) {
			this.addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = paramLops;
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		ExecLocation eloc = null;
		
		if( _operation == OperationTypes.REPLACE && et==ExecType.MR )
		{
			eloc = ExecLocation.MapOrReduce;
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
		}
		else if( _operation == OperationTypes.RMEMPTY && et==ExecType.MR )
		{
			eloc = ExecLocation.Reduce;
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			breaksAlignment=true;
		}
		else if( _operation == OperationTypes.REXPAND && et==ExecType.MR )
		{
			eloc = ExecLocation.MapOrReduce;
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.REBLOCK);
			breaksAlignment=true;
		}
		else if ( _operation == OperationTypes.TRANSFORM && et == ExecType.MR ) {
			definesMRJob = true;
			eloc = ExecLocation.MapAndReduce;
			lps.addCompatibility(JobType.TRANSFORM);
		}
		else //executed in CP / CP_FILE / SPARK
		{
			eloc = ExecLocation.ControlProgram;
			lps.addCompatibility(JobType.INVALID);
		}
		lps.setProperties(inputs, et, eloc, breaksAlignment, aligner, definesMRJob);
	}

	public ParameterizedBuiltin(HashMap<String, Lop> paramLops, OperationTypes op, DataType dt, ValueType vt, ExecType et, boolean bRmEmptyBC) 
			throws HopsException 
	{
		this(paramLops, op, dt, vt, et);
		_bRmEmptyBC = bRmEmptyBC;
	}
	
	public OperationTypes getOp() { 
		return _operation; 
	}
	
	public int getInputIndex(String name) { 
		Lop n = _inputParams.get(name);
		for(int i=0; i<getInputs().size(); i++) 
			if(getInputs().get(i) == n)
				return i;
		return -1;
	}
	
	public Lop getNamedInput(String name) {
		return _inputParams.get(name);
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
			case INVCDF:
				sb.append( (_operation == OperationTypes.CDF ? "cdf" : "invcdf") );
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
					// (offset and maxdim only apply to exec type spark)
					Lop iLop = _inputParams.get(s);
					if( s.equals( "target") || getExecType()==ExecType.SPARK )
						sb.append( iLop.getOutputParameters().getLabel());
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
			
			case REXPAND:
				sb.append("rexpand");
				sb.append(OPERAND_DELIMITOR);
				
				for ( String s : _inputParams.keySet() ) {
					
					sb.append(s);
					sb.append(NAME_VALUE_SEPARATOR);
					
					// get the value/label of the scalar input associated with name "s"
					// (offset and maxdim only apply to exec type spark)
					Lop iLop = _inputParams.get(s);
					if( s.equals( "target") || getExecType()==ExecType.SPARK )
						sb.append( iLop.getOutputParameters().getLabel());
					else
						sb.append( iLop.prepScalarLabel() );
					
					sb.append(OPERAND_DELIMITOR);
				}
				
				break;
				
			default:
				throw new LopsException(this.printErrorLocation() + "In ParameterizedBuiltin Lop, Unknown operation: " + _operation);
		}
		
		if (_operation == OperationTypes.RMEMPTY) {			
			sb.append("bRmEmptyBC");
			sb.append(NAME_VALUE_SEPARATOR);
			sb.append( _bRmEmptyBC );
			sb.append(OPERAND_DELIMITOR);
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
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int input_index5, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		switch(_operation) 
		{
			case REXPAND:
			{
				sb.append("rexpand");
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop1 = _inputParams.get("target");
				int pos1 = getInputs().indexOf(iLop1);
				int index1 = (pos1==0)? input_index1 : (pos1==1)? input_index2 : (pos1==2)? input_index3 : (pos1==3)? input_index4 : input_index5;
				sb.append(prepInputOperand(index1));
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop2 = _inputParams.get("max");
				sb.append( iLop2.prepScalarLabel() );
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop3 = _inputParams.get("dir");
				sb.append( iLop3.prepScalarLabel() );
				
				sb.append(OPERAND_DELIMITOR);
				
				Lop iLop4 = _inputParams.get("cast");
				sb.append( iLop4.prepScalarLabel() );
				
				sb.append( OPERAND_DELIMITOR );
				
				Lop iLop5 = _inputParams.get("ignore");
				sb.append( iLop5.prepScalarLabel() );
				
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
	public String getInstructions(int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		switch(_operation) 
		{
			case TRANSFORM:
			{
				int inputIndex = getInputIndex("target");
				
				sb.append( "transform" );
				sb.append( OPERAND_DELIMITOR );
		
				Lop iLop = _inputParams.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_DATA);
				sb.append(iLop.prepInputOperand(inputIndex));
				sb.append( OPERAND_DELIMITOR );
				
				Lop iLop2 = _inputParams.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_TXMTD);
				sb.append(iLop2.prepScalarLabel());
				sb.append( OPERAND_DELIMITOR );
				
				// either applyTransformPath or transformSpec should be specified
				boolean isApply = false;
				Lop iLop3 = null;
				if ( _inputParams.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_APPLYMTD) != null ) {
					// apply transform
					isApply = true;
					iLop3 = _inputParams.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_APPLYMTD);
				}
				else {
					iLop3 = _inputParams.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_TXSPEC);
				}
				sb.append(iLop3.prepScalarLabel());
				sb.append( OPERAND_DELIMITOR );
				
				sb.append(isApply);
				sb.append( OPERAND_DELIMITOR );
				
				Lop iLop4 = _inputParams.get(ParameterizedBuiltinFunctionExpression.TF_FN_PARAM_OUTNAMES);
				if( iLop4 != null ) 
				{
					sb.append(iLop4.prepScalarLabel());
					sb.append( OPERAND_DELIMITOR );
				}

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

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

package org.apache.sysds.lops;

import java.util.HashMap;
import java.util.Map.Entry;

 
import org.apache.sysds.lops.LopProperties.ExecType;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ValueType;


/**
 * Defines a LOP for functions.
 * 
 */
public class ParameterizedBuiltin extends Lop 
{
	private ParamBuiltinOp _operation;
	private HashMap<String, Lop> _inputParams;
	private boolean _bRmEmptyBC;

	//cp-specific parameters
	private int _numThreads = 1;
	
	public ParameterizedBuiltin(HashMap<String, Lop> paramLops, ParamBuiltinOp op, DataType dt, ValueType vt, ExecType et) {
		this(paramLops, op, dt, vt, et, 1);
	}
	
	public ParameterizedBuiltin(HashMap<String, Lop> paramLops, ParamBuiltinOp op, DataType dt, ValueType vt, ExecType et, int k) {
		super(Lop.Type.ParameterizedBuiltin, dt, vt);
		_operation = op;
		
		for (Lop lop : paramLops.values()) {
			addInput(lop);
			lop.addOutput(this);
		}
		
		_inputParams = paramLops;
		_numThreads = k;
		
		lps.setProperties(inputs, et);
	}

	public ParameterizedBuiltin(HashMap<String, Lop> paramLops, ParamBuiltinOp op, DataType dt, ValueType vt, ExecType et, boolean bRmEmptyBC) {
		this(paramLops, op, dt, vt, et);
		_bRmEmptyBC = bRmEmptyBC;
	}
	
	public ParamBuiltinOp getOp() { 
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
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );

		switch(_operation) 
		{
			case CDF:
			case INVCDF:
				sb.append( (_operation == ParamBuiltinOp.CDF ? "cdf" : "invcdf") );
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
					if( s.equals( "target") || s.equals( "select") || getExecType()==ExecType.SPARK )
						sb.append( iLop.getOutputParameters().getLabel());
					else
						sb.append( iLop.prepScalarLabel() );
					
					sb.append(OPERAND_DELIMITOR);
				}
				
				break;
			
			case REPLACE: {
				sb.append( "replace" );
				sb.append( OPERAND_DELIMITOR );
				sb.append(compileGenericParamMap(_inputParams));
				break;
			}
			
			case LOWER_TRI: {
				sb.append( "lowertri" );
				sb.append( OPERAND_DELIMITOR );
				sb.append(compileGenericParamMap(_inputParams));
				break;
			}
			
			case UPPER_TRI: {
				sb.append( "uppertri" );
				sb.append( OPERAND_DELIMITOR );
				sb.append(compileGenericParamMap(_inputParams));
				break;
			}
			
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
			
			case TRANSFORMAPPLY:
			case TRANSFORMDECODE:
			case TRANSFORMCOLMAP:
			case TRANSFORMMETA:{ 
				sb.append(_operation.name().toLowerCase()); //opcode
				sb.append(OPERAND_DELIMITOR);
				sb.append(compileGenericParamMap(_inputParams));
				break;
			}
			case LIST: {
				sb.append("nvlist"); //opcode
				sb.append(OPERAND_DELIMITOR);
				sb.append(compileGenericParamMap(_inputParams));
				break;
			}
			case TOSTRING: {
				sb.append("toString"); //opcode
				sb.append(OPERAND_DELIMITOR);
				sb.append(compileGenericParamMap(_inputParams));
				break;
			}

			case PARAMSERV: {
				sb.append("paramserv");
				sb.append(OPERAND_DELIMITOR);
				sb.append(compileGenericParamMap(_inputParams));
				break;
			}
				
			default:
				throw new LopsException(this.printErrorLocation() + "In ParameterizedBuiltin Lop, Unknown operation: " + _operation);
		}
		
		if (_operation == ParamBuiltinOp.RMEMPTY) {
			sb.append("bRmEmptyBC");
			sb.append(NAME_VALUE_SEPARATOR);
			sb.append( _bRmEmptyBC );
			sb.append(OPERAND_DELIMITOR);
		}
		
		if( getExecType()==ExecType.CP && _operation == ParamBuiltinOp.REXPAND ) {
			sb.append( "k" );
			sb.append( Lop.NAME_VALUE_SEPARATOR );
			sb.append( _numThreads );	
			sb.append(OPERAND_DELIMITOR);
		}
		
		sb.append(prepOutputOperand(output));
		
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
	
	private static String compileGenericParamMap(HashMap<String, Lop> params) {
		StringBuilder sb = new StringBuilder();		
		for ( Entry<String, Lop> e : params.entrySet() ) {
			sb.append(e.getKey());
			sb.append(NAME_VALUE_SEPARATOR);
			if( e.getValue().getDataType() != DataType.SCALAR )
				sb.append( e.getValue().getOutputParameters().getLabel());
			else
				sb.append( e.getValue().prepScalarLabel() );
			sb.append(OPERAND_DELIMITOR);
		}
		
		return sb.toString();
	}
}

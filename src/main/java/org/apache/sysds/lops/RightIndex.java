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

import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
 
import org.apache.sysds.common.Types.ExecType;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


public class RightIndex extends Lop 
{
	public static final String OPCODE = "rightIndex";
	
	private boolean forLeftIndexing = false;

	//optional attribute for spark exec type
	private SparkAggType _aggtype = SparkAggType.MULTI_BLOCK;

	public RightIndex(Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU,
		DataType dt, ValueType vt, ExecType et, boolean forleft)
	{
		super(Lop.Type.RightIndex, dt, vt);
		init(input, rowL, rowU, colL, colU, dt, vt, et, forleft);
	}

	public RightIndex(Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU,
		DataType dt, ValueType vt, ExecType et)
	{
		super(Lop.Type.RightIndex, dt, vt);
		init(input, rowL, rowU, colL, colU, dt, vt, et, false);
	}

	public RightIndex(Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU,
		DataType dt, ValueType vt, SparkAggType aggtype, ExecType et)
	{
		super(Lop.Type.RightIndex, dt, vt);
		_aggtype = aggtype;
		init(input, rowL, rowU, colL, colU, dt, vt, et, false);
	}

	private void init(Lop inputMatrix, Lop rowL, Lop rowU, Lop colL, Lop colU,
		DataType dt, ValueType vt, ExecType et, boolean forleft) 
	{
		addInput(inputMatrix);
		addInput(rowL);
		addInput(rowU);
		addInput(colL);
		addInput(colU);
		
		inputMatrix.addOutput(this);
		rowL.addOutput(this);
		rowU.addOutput(this);
		colL.addOutput(this);
		colU.addOutput(this);
		lps.setProperties(inputs, et);
		forLeftIndexing=forleft;
	}
	
	private String getOpcode() {
		if(forLeftIndexing)
			return OPCODE+"ForLeft";
		else
			return OPCODE;
	}

	@Override
	public SparkAggType getAggType() {
		return _aggtype;
	}
	
	@Override
	public String getInstructions(String input, String rowl, String rowu, String coll, String colu, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input));
		sb.append( OPERAND_DELIMITOR );
		
		// rowl, rowu
		sb.append( getInputs().get(1).prepScalarInputOperand(rowl));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepScalarInputOperand(rowu));
		sb.append( OPERAND_DELIMITOR );
		
		// coll, colu
		sb.append( getInputs().get(3).prepScalarInputOperand(coll));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(4).prepScalarInputOperand(colu));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getValueType() );
		
		//in case of spark, we also compile the optional aggregate flag into the instruction.
		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _aggtype );
		}
		
		return sb.toString();
	}

	@Override
	public String toString() {
		return getOpcode();
	}
}

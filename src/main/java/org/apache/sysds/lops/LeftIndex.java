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

 
import org.apache.sysds.lops.LopProperties.ExecType;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


public class LeftIndex extends Lop 
{
	public enum LixCacheType {
		RIGHT,
		LEFT,
		NONE
	}
	
	public static final String OPCODE = "leftIndex";
	
	private LixCacheType _type;

	public LeftIndex(
			Lop lhsInput, Lop rhsInput, Lop rowL, Lop rowU, Lop colL, Lop colU, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.LeftIndex, dt, vt);
		_type = LixCacheType.NONE;
		init(lhsInput, rhsInput, rowL, rowU, colL, colU, et);
	}
	
	public LeftIndex(
			Lop lhsInput, Lop rhsInput, Lop rowL, Lop rowU, Lop colL, Lop colU, DataType dt, ValueType vt, ExecType et, LixCacheType type) {
		super(Lop.Type.LeftIndex, dt, vt);
		_type = type;
		init(lhsInput, rhsInput, rowL, rowU, colL, colU, et);
	}
	
	/**
	 * Setup a LeftIndexing operation.
	 * Example: A[i:j, k:l] = B;
	 * 
	 * @param lhsMatrix left matrix lop
	 * @param rhsMatrix right matrix lop
	 * @param rowL row lower lop
	 * @param rowU row upper lop
	 * @param colL column lower lop
	 * @param colU column upper lop
	 * @param et execution type
	 */
	private void init(Lop lhsMatrix, Lop rhsMatrix, Lop rowL, Lop rowU, Lop colL, Lop colU, ExecType et) {
		/*
		 * A[i:j, k:l] = B;
		 * B -> rhsMatrix
		 * A -> lhsMatrix
		 * i,j -> rowL, rowU
		 * k,l -> colL, colU
		 */
		addInput(lhsMatrix);
		addInput(rhsMatrix);
		addInput(rowL);
		addInput(rowU);
		addInput(colL);
		addInput(colU);
		
		lhsMatrix.addOutput(this);		
		rhsMatrix.addOutput(this);
		rowL.addOutput(this);
		rowU.addOutput(this);
		colL.addOutput(this);
		colU.addOutput(this);
		lps.setProperties(inputs, et);
	}
	
	private String getOpcode() {
		if( _type != LixCacheType.NONE )
			return "mapLeftIndex";
		else
			return OPCODE;
	}
	
	@Override
	public String getInstructions(String lhsInput, String rhsInput, String rowl, String rowu, String coll, String colu, String output)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(lhsInput));
		sb.append( OPERAND_DELIMITOR );
		
		if ( getInputs().get(1).getDataType() == DataType.SCALAR ) {
			sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		}
		else {
			sb.append( getInputs().get(1).prepInputOperand(rhsInput));
		}
		sb.append( OPERAND_DELIMITOR );
		
		// rowl, rowu
		sb.append(getInputs().get(2).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		sb.append(getInputs().get(3).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		
		// rowl, rowu
		sb.append(getInputs().get(4).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		sb.append(getInputs().get(5).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		
		sb.append( this.prepOutputOperand(output));

		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );	
			sb.append(_type.toString());
		}
		
		return sb.toString();
	}

	@Override
	public String toString() {
		return getOpcode();
	}
}

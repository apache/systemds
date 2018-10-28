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

package org.tugraz.sysds.lops;

import org.tugraz.sysds.lops.Aggregate.OperationTypes;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;

public class CumulativeOffsetBinary extends Lop 
{
	
	private OperationTypes _op;
	private double _initValue = 0;
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, OperationTypes op, ExecType et) 
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
	
		init(data, offsets, dt, vt, et);
	}
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, double init, OperationTypes op, ExecType et)
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
		
		//in case of Spark, CumulativeOffset includes CumulativeSplit and hence needs the init value
		_initValue = init;
		
		init(data, offsets, dt, vt, et);
	}
	
	private void init(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) {
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		lps.setProperties( inputs, et);
	}

	@Override
	public String toString() {
		return "CumulativeOffsetBinary";
	}

	private static void checkSupportedOperations(OperationTypes op) {
		//sanity check for supported aggregates
		if( !( op == OperationTypes.KahanSum || op == OperationTypes.Product
			|| op == OperationTypes.SumProduct
			|| op == OperationTypes.Min || op == OperationTypes.Max) )
		{
			throw new LopsException("Unsupported aggregate operation type: "+op);
		}
	}
	
	private String getOpcode() {
		switch( _op ) {
			case KahanSum:   return "bcumoffk+";
			case Product:    return "bcumoff*";
			case SumProduct: return "bcumoff+*";
			case Min:        return "bcumoffmin";
			case Max:        return "bcumoffmax";
			default:         return null;
		}
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index) {
		return getInstructions(String.valueOf(input_index1), 
				String.valueOf(input_index2), String.valueOf(output_index));
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2) );
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output) );
		
		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _initValue );	
		}
		
		return sb.toString();
	}
}

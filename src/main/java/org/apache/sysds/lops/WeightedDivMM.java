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


import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class WeightedDivMM extends Lop 
{
	public static final String OPCODE = "mapwdivmm";
	public static final String OPCODE_CP = Opcodes.WDIVMM.toString();
	private int _numThreads = 1;

	public enum WDivMMType {
		DIV_LEFT,			//t(t(U) %*% (W / U%*%t(V)))
		DIV_RIGHT,			//(W / U%*%t(V)) %*% V
		DIV_LEFT_EPS,		//t(t(U) %*% (W / (U%*%t(V) + x)))
		DIV_RIGHT_EPS,		//(W / (U%*%t(V) + x)) %*% V
		MULT_BASIC,			//(W * U%*%t(V))
		MULT_LEFT,			//t(t(U) %*% (W * U%*%t(V)))
		MULT_RIGHT,			//(W * U%*%t(V)) %*% V
		MULT_MINUS_LEFT,	//t(t(U) %*% ((X!=0) * (U%*%t(V) - X)))
		MULT_MINUS_RIGHT,	//((X!=0) * (U%*%t(V) - X)) %*% V
		MULT_MINUS_4_LEFT,	//t(t(U) %*% (W * (U%*%t(V) - X)))
		MULT_MINUS_4_RIGHT;	//(W * (U%*%t(V) - X)) %*% V
		
		
		public boolean isBasic(){
			return (this == MULT_BASIC);
		}
		public boolean isLeft() {
			return (this == DIV_LEFT || this == DIV_LEFT_EPS || this == MULT_LEFT 
					|| this == MULT_MINUS_LEFT || this == MULT_MINUS_4_LEFT);
		}
		public boolean isRight() {
			return !(isLeft() || isBasic());
		}
		public boolean isMult() {
			return (this == MULT_LEFT || this == MULT_RIGHT || this == MULT_MINUS_LEFT || this == MULT_MINUS_RIGHT
					|| this == MULT_MINUS_4_LEFT || this == MULT_MINUS_4_RIGHT);
		}		
		public boolean isMinus(){
			return (this == MULT_MINUS_LEFT || this == MULT_MINUS_RIGHT 
					|| this == MULT_MINUS_4_LEFT || this == MULT_MINUS_4_RIGHT);
		}
		public boolean hasFourInputs() {
			return (this == MULT_MINUS_4_LEFT || this == MULT_MINUS_4_RIGHT 
					|| this == DIV_LEFT_EPS || this == DIV_RIGHT_EPS);
		}
		public boolean hasScalar() {
			return (this == DIV_LEFT_EPS || this == DIV_RIGHT_EPS);
		}
		
		public MatrixCharacteristics computeOutputCharacteristics(long Xrlen, long Xclen, long rank) {
			if( isBasic() )
				return new MatrixCharacteristics( Xrlen, Xclen, -1, -1);
			else	
				return new MatrixCharacteristics(isLeft()?Xclen:Xrlen, rank, -1, -1);
		}
	}
	
	private WDivMMType _weightsType = null;
	
	public WeightedDivMM(Lop input1, Lop input2, Lop input3, Lop input4, DataType dt, ValueType vt, WDivMMType wt, ExecType et) {
		super(Lop.Type.WeightedDivMM, dt, vt);
		addInput(input1); //W
		addInput(input2); //U
		addInput(input3); //V
		addInput(input4); //X (optional)
		input1.addOutput(this); 
		input2.addOutput(this);
		input3.addOutput(this);
		input4.addOutput(this);
		
		_weightsType = wt;
		setupLopProperties(et);
	}
	
	@Override
	public String toString() {
		return "Operation = WeightedDivMM";
	}

	/* CP/SPARK instruction generation */
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output)
	{
		StringBuilder sb = InstructionUtils.getStringBuilder();
		
		final ExecType et = getExecType();
		
		sb.append(et);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		if( et == ExecType.CP )
			sb.append(OPCODE_CP);
		else
			sb.append(OPCODE);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(0).prepInputOperand(input1));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(1).prepInputOperand(input2));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(2).prepInputOperand(input3));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(3).prepInputOperand(input4));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( prepOutputOperand(output));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_weightsType);
		
		//append degree of parallelism
		if( et == ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
		}
		
		return sb.toString();
	}
	
	public void setNumThreads(int k) {
		_numThreads = k;
	}
}

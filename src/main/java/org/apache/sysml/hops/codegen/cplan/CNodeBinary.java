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

package org.apache.sysml.hops.codegen.cplan;

import java.util.Arrays;

import org.apache.sysml.parser.Expression.DataType;


public class CNodeBinary extends CNode
{
	public enum BinType {
		DOT_PRODUCT,
		VECT_MULT_ADD, VECT_DIV_ADD,
		VECT_MULT_SCALAR, VECT_DIV_SCALAR, 
		MULT, DIV, PLUS, MINUS, MODULUS, INTDIV, 
		LESS, LESSEQUAL, GREATER, GREATEREQUAL, EQUAL,NOTEQUAL,
		MIN, MAX, AND, OR, LOG, POW,
		MINUS1_MULT;

		public static boolean contains(String value) {
			for( BinType bt : values()  )
				if( bt.name().equals(value) )
					return true;
			return false;
		}
		
		public boolean isCommutative() {
			return ( this == EQUAL || this == NOTEQUAL 
				|| this == PLUS || this == MULT 
				|| this == MIN || this == MAX );
		}
		
		public String getTemplate(boolean sparse) {
			switch (this) {
				case DOT_PRODUCT:   
					return sparse ? "    double %TMP% = LibSpoofPrimitives.dotProduct(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, %LEN%);\n" :
									"    double %TMP% = LibSpoofPrimitives.dotProduct(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
			
				case VECT_MULT_ADD: 
					return sparse ? "    LibSpoofPrimitives.vectMultiplyAdd(%IN1%, %IN2v%, %OUT%, %IN2i%, %POS2%, %POSOUT%, %LEN%);\n" : 
									"    LibSpoofPrimitives.vectMultiplyAdd(%IN1%, %IN2%, %OUT%, %POS2%, %POSOUT%, %LEN%);\n";
				
				case VECT_DIV_ADD: 
					return sparse ? "    LibSpoofPrimitives.vectDivAdd(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POSOUT%, %LEN%);\n" : 
									"    LibSpoofPrimitives.vectDivAdd(%IN1%, %IN2%, %OUT%, %POS1%, %POSOUT%, %LEN%);\n";
				
				case VECT_DIV_SCALAR: 
					return sparse ? "    LibSpoofPrimitives.vectDivWrite(%IN1v%, %IN1i%, %IN2%,  %OUT%, %POS1%, %POSOUT%, %LEN%);\n" : 
									"    LibSpoofPrimitives.vectDivWrite(%IN1%, %IN2%, %OUT%, %POS1%, %POSOUT%, %LEN%);\n";
				
				case VECT_MULT_SCALAR: 
					return "    LibSpoofPrimitives.vectMultiplyWrite(%IN2%, %IN1%, %POS1%, %OUT%, 0, %LEN%);\n";
							
				
				/*Can be replaced by function objects*/
				case MULT:
					return "    double %TMP% = %IN1% * %IN2%;\n" ;
				
				case DIV:
					return "    double %TMP% = %IN1% / %IN2%;\n" ;
				case PLUS:
					return "    double %TMP% = %IN1% + %IN2%;\n" ;
				case MINUS:
					return "    double %TMP% = %IN1% - %IN2%;\n" ;
				case MODULUS:
					return "    double %TMP% = %IN1% % %IN2%;\n" ;
				case INTDIV: 
					return "    double %TMP% = (int) %IN1% / %IN2%;\n" ;
				case LESS:
					return "    double %TMP% = (%IN1% < %IN2%) ? 1 : 0;\n" ;
				case LESSEQUAL:
					return "    double %TMP% = (%IN1% <= %IN2%) ? 1 : 0;\n" ;
				case GREATER:
					return "    double %TMP% = (%IN1% > %IN2%) ? 1 : 0;\n" ;
				case GREATEREQUAL: 
					return "    double %TMP% = (%IN1% >= %IN2%) ? 1 : 0;\n" ;
				case EQUAL:
					return "    double %TMP% = (%IN1% == %IN2%) ? 1 : 0;\n" ;
				case NOTEQUAL: 
					return "    double %TMP% = (%IN1% != %IN2%) ? 1 : 0;\n" ;
				
				case MIN:
					return "    double %TMP% = Math.min(%IN1%, %IN2%);\n" ;
				case MAX:
					return "    double %TMP% = Math.max(%IN1%, %IN2%);\n" ;
				case LOG:
					return "    double %TMP% = Math.log(%IN1%)/Math.log(%IN2%);\n" ;
				case POW:
					return "    double %TMP% = Math.pow(%IN1%, %IN2%);\n" ;
				case MINUS1_MULT:
					return "    double %TMP% = 1 - %IN1% * %IN2%;\n" ;
					
				default: 
					throw new RuntimeException("Invalid binary type: "+this.toString());
			}
		}
	}
	
	private final BinType _type;
	
	public CNodeBinary( CNode in1, CNode in2, BinType type ) {
		//canonicalize commutative matrix-scalar operations
		//to increase reuse potential
		if( type.isCommutative() && in1 instanceof CNodeData 
			&& in1.getDataType()==DataType.SCALAR ) {
			CNode tmp = in1;
			in1 = in2; 
			in2 = tmp;
		}
		
		_inputs.add(in1);
		_inputs.add(in2);
		_type = type;
		setOutputDims();
	}

	public BinType getType() {
		return _type;
	}
	
	@Override
	public String codegen(boolean sparse) {
		if( _generated )
			return "";
			
		StringBuilder sb = new StringBuilder();
		
		//generate children
		sb.append(_inputs.get(0).codegen(sparse));
		sb.append(_inputs.get(1).codegen(sparse));
		
		//generate binary operation
		String var = createVarname();
		String tmp = _type.getTemplate(sparse);
		tmp = tmp.replaceAll("%TMP%", var);
		for( int j=1; j<=2; j++ ) {
			String varj = _inputs.get(j-1).getVarname();
			if( sparse && !tmp.contains("%IN"+j+"%") ) {
				tmp = tmp.replaceAll("%IN"+j+"v%", varj+"vals");
				tmp = tmp.replaceAll("%IN"+j+"i%", varj+"ix");
			}
			else
				tmp = tmp.replaceAll("%IN"+j+"%", varj );
			
			if(varj.startsWith("b")  ) //i.e. b.get(index)
				tmp = tmp.replaceAll("%POS"+j+"%", "bi");
			else
				tmp = tmp.replaceAll("%POS"+j+"%", varj+"i");
		}
		sb.append(tmp);
		
		//mark as generated
		_generated = true;
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		switch(_type) {
			case DOT_PRODUCT: return "b(dot)";
			case VECT_MULT_ADD: return "b(vma)";
			case VECT_DIV_ADD: return "b(vda)";
			case MULT: return "b(*)";
			case DIV: return "b(/)";
			case PLUS: return "b(+)";
			case MINUS: return "b(-)";
			case MODULUS: return "b(%%)";
			case INTDIV: return "b(%/%)";
			case LESS: return "b(<)";
			case LESSEQUAL: return "b(<=)";
			case GREATER: return "b(>)";
			case GREATEREQUAL: return "b(>=)";
			case EQUAL: return "b(==)";
			case NOTEQUAL: return "b(!=)";
			case MINUS1_MULT: return "b(1-*)";
			case VECT_DIV_SCALAR:  return "b(vector/)";
			case VECT_MULT_SCALAR:  return "b(vector*)";
			default: return "b("+_type.name()+")";
		}
	}
	
	@Override
	public void setOutputDims()
	{
		switch(_type) {
			//VECT
			case VECT_MULT_ADD: 
			case VECT_DIV_ADD:
				_rows = _inputs.get(1)._rows;
				_cols = _inputs.get(1)._cols;
				_dataType= DataType.MATRIX;
				break;
				
			case VECT_DIV_SCALAR: 	
			case VECT_MULT_SCALAR:
				_rows = _inputs.get(0)._rows;
				_cols = _inputs.get(0)._cols;
				_dataType= DataType.MATRIX;
				break;
				
		
			case DOT_PRODUCT: 
			
			//SCALAR Arithmetic
			case MULT: 
			case DIV: 
			case PLUS: 
			case MINUS: 
			case MINUS1_MULT:	
			case MODULUS: 
			case INTDIV: 	
			//SCALAR Comparison
			case LESS: 
			case LESSEQUAL: 
			case GREATER: 
			case GREATEREQUAL: 
			case EQUAL: 
			case NOTEQUAL: 
			//SCALAR LOGIC
			case MIN: 
			case MAX: 
			case AND: 
			case OR: 			
			case LOG: 
			case POW: 
				_rows = 0;
				_cols = 0;
				_dataType= DataType.SCALAR;
				break;	
		}
	}
	
	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			int h1 = super.hashCode();
			int h2 = _type.hashCode();
			_hash = Arrays.hashCode(new int[]{h1,h2});
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		if( !(o instanceof CNodeBinary) )
			return false;
		
		CNodeBinary that = (CNodeBinary) o;
		return super.equals(that)
			&& _type == that._type;
	}
}

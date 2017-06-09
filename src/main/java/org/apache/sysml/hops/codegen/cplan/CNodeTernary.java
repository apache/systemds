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


public class CNodeTernary extends CNode
{
	public enum TernaryType {
		PLUS_MULT, MINUS_MULT,
		REPLACE, REPLACE_NAN,
		LOOKUP_RC1;
		
		public static boolean contains(String value) {
			for( TernaryType tt : values()  )
				if( tt.name().equals(value) )
					return true;
			return false;
		}
		
		public String getTemplate(boolean sparse) {
			switch (this) {
				case PLUS_MULT:
					return "    double %TMP% = %IN1% + %IN2% * %IN3%;\n";
				
				case MINUS_MULT:
					return "    double %TMP% = %IN1% - %IN2% * %IN3%;\n";
					
				case REPLACE:
					return "    double %TMP% = (%IN1% == %IN2% || (Double.isNaN(%IN1%) "
							+ "&& Double.isNaN(%IN2%))) ? %IN3% : %IN1%;\n";
				
				case REPLACE_NAN:
					return "    double %TMP% = Double.isNaN(%IN1%) ? %IN3% : %IN1%;\n";
					
				case LOOKUP_RC1:
					return sparse ?
							"    double %TMP% = getValue(%IN1v%, %IN2%, rowIndex, %IN3%-1);\n" :	
							"    double %TMP% = getValue(%IN1%, %IN2%, rowIndex, %IN3%-1);\n";	
					
				default: 
					throw new RuntimeException("Invalid ternary type: "+this.toString());
			}
		}
	}
	
	private final TernaryType _type;
	
	public CNodeTernary( CNode in1, CNode in2, CNode in3, TernaryType type ) {
		_inputs.add(in1);
		_inputs.add(in2);
		_inputs.add(in3);
		_type = type;
		setOutputDims();
	}

	public TernaryType getType() {
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
		sb.append(_inputs.get(2).codegen(sparse));
		
		//generate binary operation
		String var = createVarname();
		String tmp = _type.getTemplate(sparse);
		tmp = tmp.replaceAll("%TMP%", var);
		for( int j=1; j<=3; j++ ) {
			String varj = _inputs.get(j-1).getVarname();
			//replace sparse and dense inputs
			tmp = tmp.replaceAll("%IN"+j+"v%", 
				varj+(varj.startsWith("b")?"":"vals") );
			tmp = tmp.replaceAll("%IN"+j+"%", varj );
		}
		sb.append(tmp);
		
		//mark as generated
		_generated = true;
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		switch(_type) {
			case PLUS_MULT: return "t(+*)";
			case MINUS_MULT: return "t(-*)";
			case REPLACE: 
			case REPLACE_NAN: return "t(rplc)";
			case LOOKUP_RC1: return "u(ixrc1)";
			default:
				return super.toString();	
		}
	}
	
	@Override
	public void setOutputDims() {
		switch(_type) {
			case PLUS_MULT: 
			case MINUS_MULT:
			case REPLACE:
			case REPLACE_NAN:
			case LOOKUP_RC1:
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
		if( !(o instanceof CNodeTernary) )
			return false;
		
		CNodeTernary that = (CNodeTernary) o;
		return super.equals(that)
			&& _type == that._type;
	}
}

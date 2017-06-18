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

import org.apache.commons.lang.StringUtils;
import org.apache.sysml.hops.codegen.template.TemplateUtils;
import org.apache.sysml.parser.Expression.DataType;


public class CNodeBinary extends CNode
{
	public enum BinType {
		DOT_PRODUCT,
		//vector-scalar-add operations
		VECT_MULT_ADD, VECT_DIV_ADD, VECT_MINUS_ADD, VECT_PLUS_ADD,
		VECT_POW_ADD, VECT_MIN_ADD, VECT_MAX_ADD,
		VECT_EQUAL_ADD, VECT_NOTEQUAL_ADD, VECT_LESS_ADD, 
		VECT_LESSEQUAL_ADD, VECT_GREATER_ADD, VECT_GREATEREQUAL_ADD,
		//vector-scalar operations
		VECT_MULT_SCALAR, VECT_DIV_SCALAR, VECT_MINUS_SCALAR, VECT_PLUS_SCALAR,
		VECT_POW_SCALAR, VECT_MIN_SCALAR, VECT_MAX_SCALAR,
		VECT_EQUAL_SCALAR, VECT_NOTEQUAL_SCALAR, VECT_LESS_SCALAR, 
		VECT_LESSEQUAL_SCALAR, VECT_GREATER_SCALAR, VECT_GREATEREQUAL_SCALAR,
		//vector-vector operations
		VECT_MULT, VECT_DIV, VECT_MINUS, VECT_PLUS, VECT_MIN, VECT_MAX, VECT_EQUAL, 
		VECT_NOTEQUAL, VECT_LESS, VECT_LESSEQUAL, VECT_GREATER, VECT_GREATEREQUAL,
		//scalar-scalar operations
		MULT, DIV, PLUS, MINUS, MODULUS, INTDIV, 
		LESS, LESSEQUAL, GREATER, GREATEREQUAL, EQUAL,NOTEQUAL,
		MIN, MAX, AND, OR, LOG, LOG_NZ, POW,
		MINUS1_MULT, MINUS_NZ;
		
		public static boolean contains(String value) {
			for( BinType bt : values()  )
				if( bt.name().equals(value) )
					return true;
			return false;
		}
		
		public boolean isCommutative() {
			boolean ssComm = (this==EQUAL || this==NOTEQUAL 
				|| this==PLUS || this==MULT || this==MIN || this==MAX);
			boolean vsComm = (this==VECT_EQUAL_SCALAR || this==VECT_NOTEQUAL_SCALAR 
					|| this==VECT_PLUS_SCALAR || this==VECT_MULT_SCALAR 
					|| this==VECT_MIN_SCALAR || this==VECT_MAX_SCALAR);
			boolean vvComm = (this==VECT_EQUAL || this==VECT_NOTEQUAL 
					|| this==VECT_PLUS || this==VECT_MULT || this==VECT_MIN || this==VECT_MAX);
			return ssComm || vsComm || vvComm;
		}
		
		public String getTemplate(boolean sparse, boolean scalarVector) {
			switch (this) {
				case DOT_PRODUCT:   
					return sparse ? "    double %TMP% = LibSpoofPrimitives.dotProduct(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, %LEN%);\n" :
									"    double %TMP% = LibSpoofPrimitives.dotProduct(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
				
				//vector-scalar-add operations
				case VECT_MULT_ADD:
				case VECT_DIV_ADD:
				case VECT_MINUS_ADD:
				case VECT_PLUS_ADD:
				case VECT_POW_ADD:
				case VECT_MIN_ADD:
				case VECT_MAX_ADD:	
				case VECT_EQUAL_ADD:
				case VECT_NOTEQUAL_ADD:
				case VECT_LESS_ADD:
				case VECT_LESSEQUAL_ADD:
				case VECT_GREATER_ADD:
				case VECT_GREATEREQUAL_ADD: {
					String vectName = getVectorPrimitiveName();
					if( scalarVector )
						return sparse ? "    LibSpoofPrimitives.vect"+vectName+"Add(%IN1%, %IN2v%, %OUT%, %IN2i%, %POS2%, %POSOUT%, %LEN%);\n" : 
										"    LibSpoofPrimitives.vect"+vectName+"Add(%IN1%, %IN2%, %OUT%, %POS2%, %POSOUT%, %LEN%);\n";
					else	
						return sparse ? "    LibSpoofPrimitives.vect"+vectName+"Add(%IN1v%, %IN2%, %OUT%, %IN1i%, %POS1%, %POSOUT%, %LEN%);\n" : 
										"    LibSpoofPrimitives.vect"+vectName+"Add(%IN1%, %IN2%, %OUT%, %POS1%, %POSOUT%, %LEN%);\n";
				}
				
				//vector-scalar operations
				case VECT_MULT_SCALAR:
				case VECT_DIV_SCALAR:
				case VECT_MINUS_SCALAR:
				case VECT_PLUS_SCALAR:
				case VECT_POW_SCALAR:
				case VECT_MIN_SCALAR:
				case VECT_MAX_SCALAR:	
				case VECT_EQUAL_SCALAR:
				case VECT_NOTEQUAL_SCALAR:
				case VECT_LESS_SCALAR:
				case VECT_LESSEQUAL_SCALAR:
				case VECT_GREATER_SCALAR:
				case VECT_GREATEREQUAL_SCALAR: {
					String vectName = getVectorPrimitiveName();
					if( scalarVector )
						return sparse ? "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2v%, %IN2i%, %POS2%, %LEN%);\n" : 
										"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS2%, %LEN%);\n";
					else	
						return sparse ? "    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN2%, %IN1i%, %POS1%, %LEN%);\n" : 
										"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %LEN%);\n";
				}
				
				//vector-vector operations
				case VECT_MULT:
				case VECT_DIV:
				case VECT_MINUS:
				case VECT_PLUS:
				case VECT_MIN:
				case VECT_MAX:	
				case VECT_EQUAL:
				case VECT_NOTEQUAL:
				case VECT_LESS:
				case VECT_LESSEQUAL:
				case VECT_GREATER:
				case VECT_GREATEREQUAL: {
					String vectName = getVectorPrimitiveName();
					return sparse ? 
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN2%, %IN1i%, %POS1%, %POS2%, %LEN%);\n" : 
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %IN2%, %POS1%, %POS2%, %LEN%);\n";
				}
				
				//scalar-scalar operations
				case MULT:
					return "    double %TMP% = %IN1% * %IN2%;\n";
				
				case DIV:
					return "    double %TMP% = %IN1% / %IN2%;\n";
				case PLUS:
					return "    double %TMP% = %IN1% + %IN2%;\n";
				case MINUS:
					return "    double %TMP% = %IN1% - %IN2%;\n";
				case MODULUS:
					return "    double %TMP% = LibSpoofPrimitives.mod(%IN1%, %IN2%);\n";
				case INTDIV: 
					return "    double %TMP% = LibSpoofPrimitives.intDiv(%IN1%, %IN2%);\n";
				case LESS:
					return "    double %TMP% = (%IN1% < %IN2%) ? 1 : 0;\n";
				case LESSEQUAL:
					return "    double %TMP% = (%IN1% <= %IN2%) ? 1 : 0;\n";
				case GREATER:
					return "    double %TMP% = (%IN1% > %IN2%) ? 1 : 0;\n";
				case GREATEREQUAL: 
					return "    double %TMP% = (%IN1% >= %IN2%) ? 1 : 0;\n";
				case EQUAL:
					return "    double %TMP% = (%IN1% == %IN2%) ? 1 : 0;\n";
				case NOTEQUAL: 
					return "    double %TMP% = (%IN1% != %IN2%) ? 1 : 0;\n";
				
				case MIN:
					return "    double %TMP% = (%IN1% <= %IN2%) ? %IN1% : %IN2%;\n";
				case MAX:
					return "    double %TMP% = (%IN1% >= %IN2%) ? %IN1% : %IN2%;\n";
				case LOG:
					return "    double %TMP% = FastMath.log(%IN1%)/FastMath.log(%IN2%);\n";
				case LOG_NZ:
					return "    double %TMP% = (%IN1% == 0) ? 0 : FastMath.log(%IN1%)/FastMath.log(%IN2%);\n";	
				case POW:
					return "    double %TMP% = Math.pow(%IN1%, %IN2%);\n";
				case MINUS1_MULT:
					return "    double %TMP% = 1 - %IN1% * %IN2%;\n";
				case MINUS_NZ:
					return "    double %TMP% = (%IN1% != 0) ? %IN1% - %IN2% : 0;\n";
					
				default: 
					throw new RuntimeException("Invalid binary type: "+this.toString());
			}
		}
		public boolean isVectorScalarPrimitive() {
			return this == VECT_DIV_SCALAR || this == VECT_MULT_SCALAR 
				|| this == VECT_MINUS_SCALAR || this == VECT_PLUS_SCALAR
				|| this == VECT_POW_SCALAR 
				|| this == VECT_MIN_SCALAR || this == VECT_MAX_SCALAR
				|| this == VECT_EQUAL_SCALAR || this == VECT_NOTEQUAL_SCALAR
				|| this == VECT_LESS_SCALAR || this == VECT_LESSEQUAL_SCALAR
				|| this == VECT_GREATER_SCALAR || this == VECT_GREATEREQUAL_SCALAR;
		}
		public boolean isVectorVectorPrimitive() {
			return this == VECT_DIV || this == VECT_MULT 
				|| this == VECT_MINUS || this == VECT_PLUS
				|| this == VECT_MIN || this == VECT_MAX
				|| this == VECT_EQUAL || this == VECT_NOTEQUAL
				|| this == VECT_LESS || this == VECT_LESSEQUAL
				|| this == VECT_GREATER || this == VECT_GREATEREQUAL;
		}
		public BinType getVectorAddPrimitive() {
			return BinType.valueOf("VECT_"+getVectorPrimitiveName().toUpperCase()+"_ADD");
		}
		public String getVectorPrimitiveName() {
			String [] tmp = this.name().split("_");
			return StringUtils.capitalize(tmp[1].toLowerCase());
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
		
		//generate binary operation (use sparse template, if data input)
		boolean lsparse = sparse && (_inputs.get(0) instanceof CNodeData 
			&& !_inputs.get(0).getVarname().startsWith("b")
			&& !_inputs.get(0).isLiteral());
		boolean scalarVector = (_inputs.get(0).getDataType().isScalar()
			&& _inputs.get(1).getDataType().isMatrix());
		String var = createVarname();
		String tmp = _type.getTemplate(lsparse, scalarVector);
		tmp = tmp.replaceAll("%TMP%", var);
		
		//replace input references and start indexes
		for( int j=1; j<=2; j++ ) {
			String varj = _inputs.get(j-1).getVarname();
			
			//replace sparse and dense inputs
			tmp = tmp.replaceAll("%IN"+j+"v%", varj+"vals");
			tmp = tmp.replaceAll("%IN"+j+"i%", varj+"ix");
			tmp = tmp.replaceAll("%IN"+j+"%", varj );
			
			//replace start position of main input
			tmp = tmp.replaceAll("%POS"+j+"%", (_inputs.get(j-1) instanceof CNodeData 
				&& _inputs.get(j-1).getDataType().isMatrix()) ? (!varj.startsWith("b")) ? 
				varj+"i" : TemplateUtils.isMatrix(_inputs.get(j-1)) ? "rowIndex*len" : "0" : "0");
		}
		sb.append(tmp);
		
		//mark as generated
		_generated = true;
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		switch(_type) {
			case DOT_PRODUCT:              return "b(dot)";
			case VECT_MULT_ADD:            return "b(vma)";
			case VECT_DIV_ADD:             return "b(vda)";
			case VECT_MINUS_ADD:           return "b(vmia)";
			case VECT_PLUS_ADD:            return "b(vpa)";
			case VECT_POW_ADD:             return "b(vpowa)";
			case VECT_MIN_ADD:             return "b(vmina)";
			case VECT_MAX_ADD:             return "b(vmaxa)";
			case VECT_EQUAL_ADD:           return "b(veqa)";
			case VECT_NOTEQUAL_ADD:        return "b(vneqa)";
			case VECT_LESS_ADD:            return "b(vlta)";
			case VECT_LESSEQUAL_ADD:       return "b(vltea)";
			case VECT_GREATEREQUAL_ADD:    return "b(vgtea)";
			case VECT_GREATER_ADD:         return "b(vgta)";
			case VECT_MULT_SCALAR:         return "b(vm)";
			case VECT_DIV_SCALAR:          return "b(vd)";
			case VECT_MINUS_SCALAR:        return "b(vmi)";
			case VECT_PLUS_SCALAR:         return "b(vp)";
			case VECT_POW_SCALAR:          return "b(vpow)";
			case VECT_MIN_SCALAR:          return "b(vmin)";
			case VECT_MAX_SCALAR:          return "b(vmax)";
			case VECT_EQUAL_SCALAR:        return "b(veq)";
			case VECT_NOTEQUAL_SCALAR:     return "b(vneq)";
			case VECT_LESS_SCALAR:         return "b(vlt)";
			case VECT_LESSEQUAL_SCALAR:    return "b(vlte)";
			case VECT_GREATEREQUAL_SCALAR: return "b(vgte)";
			case VECT_GREATER_SCALAR:      return "b(vgt)";
			case VECT_MULT:                return "b(v2m)";
			case VECT_DIV:                 return "b(v2d)";
			case VECT_MINUS:               return "b(v2mi)";
			case VECT_PLUS:                return "b(v2p)";
			case VECT_MIN:                 return "b(v2min)";
			case VECT_MAX:                 return "b(v2max)";
			case VECT_EQUAL:               return "b(v2eq)";
			case VECT_NOTEQUAL:            return "b(v2neq)";
			case VECT_LESS:                return "b(v2lt)";
			case VECT_LESSEQUAL:           return "b(v2lte)";
			case VECT_GREATEREQUAL:        return "b(v2gte)";
			case VECT_GREATER:             return "b(v2gt)";
			case MULT:                     return "b(*)";
			case DIV:                      return "b(/)";
			case PLUS:                     return "b(+)";
			case MINUS:                    return "b(-)";
			case POW:                      return "b(^)";
			case MODULUS:                  return "b(%%)";
			case INTDIV:                   return "b(%/%)";
			case LESS:                     return "b(<)";
			case LESSEQUAL:                return "b(<=)";
			case GREATER:                  return "b(>)";
			case GREATEREQUAL:             return "b(>=)";
			case EQUAL:                    return "b(==)";
			case NOTEQUAL:                 return "b(!=)";
			case OR:                       return "b(|)";
			case AND:                      return "b(&)";
			case MINUS1_MULT:              return "b(1-*)";
			case MINUS_NZ:                 return "b(-nz)";
			default: return "b("+_type.name().toLowerCase()+")";
		}
	}
	
	@Override
	public void setOutputDims()
	{
		switch(_type) {
			//VECT
			case VECT_MULT_ADD: 
			case VECT_DIV_ADD:
			case VECT_MINUS_ADD:
			case VECT_PLUS_ADD:
			case VECT_POW_ADD:
			case VECT_MIN_ADD:
			case VECT_MAX_ADD:
			case VECT_EQUAL_ADD: 
			case VECT_NOTEQUAL_ADD: 
			case VECT_LESS_ADD: 
			case VECT_LESSEQUAL_ADD: 
			case VECT_GREATER_ADD: 
			case VECT_GREATEREQUAL_ADD:
				boolean vectorScalar = _inputs.get(1).getDataType()==DataType.SCALAR;
				_rows = _inputs.get(vectorScalar ? 0 : 1)._rows;
				_cols = _inputs.get(vectorScalar ? 0 : 1)._cols;
				_dataType= DataType.MATRIX;
				break;
				
			case VECT_DIV_SCALAR: 	
			case VECT_MULT_SCALAR:
			case VECT_MINUS_SCALAR:
			case VECT_PLUS_SCALAR:
			case VECT_POW_SCALAR:
			case VECT_MIN_SCALAR:
			case VECT_MAX_SCALAR:
			case VECT_EQUAL_SCALAR: 
			case VECT_NOTEQUAL_SCALAR: 
			case VECT_LESS_SCALAR: 
			case VECT_LESSEQUAL_SCALAR: 
			case VECT_GREATER_SCALAR: 
			case VECT_GREATEREQUAL_SCALAR:
			
			case VECT_DIV: 	
			case VECT_MULT:
			case VECT_MINUS:
			case VECT_PLUS:
			case VECT_MIN:
			case VECT_MAX:
			case VECT_EQUAL: 
			case VECT_NOTEQUAL: 
			case VECT_LESS: 
			case VECT_LESSEQUAL: 
			case VECT_GREATER: 
			case VECT_GREATEREQUAL:	
				boolean scalarVector = (_inputs.get(0).getDataType()==DataType.SCALAR);
				_rows = _inputs.get(scalarVector ? 1 : 0)._rows;
				_cols = _inputs.get(scalarVector ? 1 : 0)._cols;
				_dataType= DataType.MATRIX;
				break;
				
		
			case DOT_PRODUCT: 
			
			//SCALAR Arithmetic
			case MULT: 
			case DIV: 
			case PLUS: 
			case MINUS: 
			case MINUS1_MULT:
			case MINUS_NZ:
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
			case LOG_NZ:	
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

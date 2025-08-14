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

package org.apache.sysds.hops.codegen.cplan;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;

public class CNodeUnary extends CNode
{
	public enum UnaryType {
		LOOKUP_R, LOOKUP_C, LOOKUP_RC, LOOKUP0, //codegen specific
		ROW_SUMS, ROW_SUMSQS, ROW_COUNTNNZS, //codegen specific
		ROW_MEANS, ROW_MINS, ROW_MAXS, ROW_VARS,
		VECT_EXP, VECT_POW2, VECT_MULT2, VECT_SQRT, VECT_LOG,
		VECT_ABS, VECT_ROUND, VECT_CEIL, VECT_FLOOR, VECT_SIGN, 
		VECT_SIN, VECT_COS, VECT_TAN, VECT_ASIN, VECT_ACOS, VECT_ATAN, 
		VECT_SINH, VECT_COSH, VECT_TANH,
		VECT_CUMSUM, VECT_CUMMIN, VECT_CUMMAX,
		VECT_SPROP, VECT_SIGMOID,
		EXP, POW2, MULT2, SQRT, LOG, LOG_NZ,
		ABS, ROUND, CEIL, FLOOR, SIGN, 
		SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH,
		SPROP, SIGMOID; 
		
		public static boolean contains(String value) {
			return Arrays.stream(values()).anyMatch(ut -> ut.name().equals(value));
		}

		public boolean isVectorScalarPrimitive() {
			return this == VECT_EXP || this == VECT_POW2
				|| this == VECT_MULT2 || this == VECT_SQRT
				|| this == VECT_LOG || this == VECT_ABS
				|| this == VECT_ROUND || this == VECT_CEIL
				|| this == VECT_FLOOR || this == VECT_SIGN
				|| this == VECT_SIN || this == VECT_COS || this == VECT_TAN
				|| this == VECT_ASIN || this == VECT_ACOS || this == VECT_ATAN
				|| this == VECT_SINH || this == VECT_COSH || this == VECT_TANH
				|| this == VECT_CUMSUM || this == VECT_CUMMIN || this == VECT_CUMMAX
				|| this == VECT_SPROP || this == VECT_SIGMOID;
		}
		public UnaryType getVectorAddPrimitive() {
			return UnaryType.valueOf("VECT_"+getVectorPrimitiveName().toUpperCase()+"_ADD");
		}
		public String getVectorPrimitiveName() {
			String [] tmp = this.name().split("_");
			return StringUtils.capitalize(tmp[1].toLowerCase());
		}
		public boolean isScalarLookup() {
			return ArrayUtils.contains(new UnaryType[]{
				LOOKUP0, LOOKUP_R, LOOKUP_C, LOOKUP_RC}, this);
		}
		public boolean isSparseSafeScalar() {
			return ArrayUtils.contains(new UnaryType[]{
				POW2, MULT2, ABS, ROUND, CEIL, FLOOR, SIGN, 
				SIN, TAN, SPROP}, this);
		}
		
		public boolean isNotSupportedBySpoofCUDA() {
			return this == VECT_CUMSUM || this == VECT_CUMMIN || this == VECT_CUMMAX|| 
					this == VECT_SPROP || this == VECT_SIGMOID;
		}
	}
	
	private UnaryType _type;
	private boolean sparseTemplate;

	public CNodeUnary( CNode in1, UnaryType type ) {
		_inputs.add(in1);
		_type = type;
		setOutputDims();
	}

	public CNodeUnary( CNode in1, UnaryType type, double sparsity ) {
		_inputs.add(in1);
		_type = type;
		setOutputDims();
		sparseTemplate = getTemplateType(sparsity);
	}

	public UnaryType getType() {
		return _type;
	}
	
	public void setType(UnaryType type) {
		_type = type;
	}

	@Override
	public String codegen(boolean sparse, GeneratorAPI api) {
		if( isGenerated() )
			return "";
		
		StringBuilder sb = new StringBuilder();
		
		//generate children
		sb.append(_inputs.get(0).codegen(sparse, api));
		
		//generate unary operation
		boolean lsparse = sparse &&
			((_inputs.get(0) instanceof CNodeData
				&& _inputs.get(0).getVarname().startsWith("a")
				&& !_inputs.get(0).isLiteral())
				|| _inputs.get(0).getVarname().startsWith("STMP"));
		String var = createVarname(sparseTemplate && lsparse && getOutputType());
		String tmp = getLanguageTemplateClass(this, api).getTemplate(_type, lsparse, sparseTemplate);
		tmp = tmp.replaceAll("%TMP%", var);
		
		//replace sparse and dense inputs
		String varj = _inputs.get(0).getVarname();
		boolean vectIn = varj.startsWith("b") && !_type.isScalarLookup();
		tmp = replaceUnaryPlaceholders(tmp, varj, vectIn, api);
		
		sb.append(tmp);
		
		//mark as generated
		_generated = true;
		
		return sb.toString();
	}

	public boolean getTemplateType(double sparsity) {
		if(!DMLScript.SPARSE_INTERMEDIATE)
			return false;
		else {
			switch(_type) {
				case VECT_SQRT:
				case VECT_ABS:
				case VECT_ROUND:
				case VECT_CEIL:
				case VECT_FLOOR:
				case VECT_SIN:
				case VECT_TAN:
				case VECT_ASIN:
				case VECT_ATAN:
				case VECT_SINH:
				case VECT_TANH:
				case VECT_SIGN: return sparsity <= 0.3;
				default: return false;
			}
		}
	}

	public boolean getOutputType() {
		switch(_type) {
			case VECT_SQRT:
			case VECT_ABS:
			case VECT_ROUND:
			case VECT_CEIL:
			case VECT_FLOOR:
			case VECT_SIN:
			case VECT_TAN:
			case VECT_ASIN:
			case VECT_ATAN:
			case VECT_SINH:
			case VECT_TANH:
			case VECT_SIGN: return true;
			default: return false;
		}
	}
	
	@Override
	public String toString() {
		switch(_type) {
			case ROW_SUMS:   return "u(R+)";
			case ROW_SUMSQS: return "u(Rsq+)";
			case ROW_MINS:   return "u(Rmin)";
			case ROW_MAXS:   return "u(Rmax)";
			case ROW_MEANS:  return "u(Rmean)";
			case ROW_VARS:	 return "u(Rvar)";
			case ROW_COUNTNNZS: return "u(Rnnz)";
			case VECT_EXP:
			case VECT_POW2:
			case VECT_MULT2:
			case VECT_SQRT:
			case VECT_LOG:
			case VECT_ABS:
			case VECT_ROUND:
			case VECT_CEIL:
			case VECT_SIN:
			case VECT_COS:
			case VECT_TAN:
			case VECT_ASIN:
			case VECT_ACOS:
			case VECT_ATAN:
			case VECT_SINH:
			case VECT_COSH:
			case VECT_TANH:
			case VECT_FLOOR:
			case VECT_CUMSUM:
			case VECT_CUMMIN:
			case VECT_CUMMAX:
			case VECT_SIGN:
			case VECT_SIGMOID:
			case VECT_SPROP:return "u(v"+_type.name().toLowerCase()+")";
			case LOOKUP_R:  return "u(ixr)";
			case LOOKUP_C:  return "u(ixc)";
			case LOOKUP_RC: return "u(ixrc)";
			case LOOKUP0:   return "u(ix0)";
			case POW2:      return Opcodes.POW2.toString();
			default:        return "u("+_type.name().toLowerCase()+")";
		}
	}

	@Override
	public void setOutputDims() {
		switch(_type) {
			case VECT_EXP:
			case VECT_POW2:
			case VECT_MULT2: 
			case VECT_SQRT: 
			case VECT_LOG:
			case VECT_ABS:
			case VECT_ROUND:
			case VECT_CEIL:
			case VECT_FLOOR:
			case VECT_SIGN:
			case VECT_SIN:
			case VECT_COS:
			case VECT_TAN:
			case VECT_ASIN:
			case VECT_ACOS:
			case VECT_ATAN:
			case VECT_SINH:
			case VECT_COSH:
			case VECT_TANH:
			case VECT_CUMSUM:
			case VECT_CUMMIN:
			case VECT_CUMMAX:
			case VECT_SPROP:
			case VECT_SIGMOID:
				_rows = _inputs.get(0)._rows;
				_cols = _inputs.get(0)._cols;
				_dataType= DataType.MATRIX;
				break;
			
			case ROW_SUMS:
			case ROW_SUMSQS:
			case ROW_MINS:
			case ROW_MAXS:
			case ROW_MEANS:
			case ROW_VARS:
			case ROW_COUNTNNZS:
			case EXP:
			case LOOKUP_R:
			case LOOKUP_C:
			case LOOKUP_RC:
			case LOOKUP0:
			case POW2:
			case MULT2:
			case ABS:  
			case SIN:
			case COS: 
			case TAN:
			case ASIN:
			case ACOS:
			case ATAN:
			case SINH:
			case COSH:
			case TANH:
			case SIGN:
			case SQRT:
			case LOG:
			case ROUND:
			case CEIL:
			case FLOOR:
			case SPROP:
			case SIGMOID:
			case LOG_NZ:
				_rows = 0;
				_cols = 0;
				_dataType= DataType.SCALAR;
				break;
			default:
				throw new RuntimeException("Operation " + _type.toString() + " has no "
					+ "output dimensions, dimensions needs to be specified for the CNode " );
		}
		
	}
	
	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			_hash = UtilFunctions.intHashCode(
				super.hashCode(), _type.hashCode());
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		if( !(o instanceof CNodeUnary) )
			return false;
		
		CNodeUnary that = (CNodeUnary) o;
		return super.equals(that)
			&& _type == that._type;
	}
			
	@Override
	public boolean isSupported(GeneratorAPI api) {
		boolean is_supported = (api == GeneratorAPI.CUDA || api == GeneratorAPI.JAVA);
		
		// ToDo: support these
		if(api == GeneratorAPI.CUDA)
			is_supported = !_type.isNotSupportedBySpoofCUDA();
		
		int i = 0;
		while(is_supported && i < _inputs.size()) {
			CNode in = _inputs.get(i++);
			is_supported = in.isSupported(api);
		}
		return  is_supported;
	}
}

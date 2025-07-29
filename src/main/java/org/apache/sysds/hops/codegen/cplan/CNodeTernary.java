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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;

public class CNodeTernary extends CNode
{
	public enum TernaryType {
		PLUS_MULT, MINUS_MULT,
		BIASADD, BIASMULT,
		REPLACE, REPLACE_NAN, IFELSE,
		LOOKUP_RC1, LOOKUP_RVECT1;
	
		public static boolean contains(String value) {
			return Arrays.stream(values()).anyMatch(tt -> tt.name().equals(value));
		}
		

		public boolean isVectorPrimitive() {
			return (this == LOOKUP_RVECT1);
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
	public String codegen(boolean sparse, GeneratorAPI api) {
		if( isGenerated() )
			return "";
			
		StringBuilder sb = new StringBuilder();
		
		//generate children
		sb.append(_inputs.get(0).codegen(sparse, api));
		sb.append(_inputs.get(1).codegen(sparse, api));
		sb.append(_inputs.get(2).codegen(sparse, api));
		
		//generate binary operation
		boolean lsparse = sparse && (_inputs.get(0) instanceof CNodeData
			&& _inputs.get(0).getVarname().startsWith("a")
			&& !_inputs.get(0).isLiteral());
		String var = createVarname();
//		String tmp = _type.getTemplate(lsparse, api, lang);
		String tmp = getLanguageTemplateClass(this, api).getTemplate(_type, lsparse);

		tmp = tmp.replace("%TMP%", var);
		for( int j=1; j<=3; j++ ) {
			String varj = _inputs.get(j-1).getVarname();
			//replace sparse and dense inputs
			tmp = tmp.replace("%IN"+j+"v%", 
				varj+(varj.startsWith("a")?"vals" : varj.startsWith("STMP") ? ".values()" :"") );
			tmp = tmp.replace("%IN"+j+"i%", 
				varj+(varj.startsWith("a")?"ix": varj.startsWith("STMP") ? ".indexes()" :"") );
			tmp = tmp.replace("%IN"+j+"%", varj );
			tmp = tmp.replace("%POS%", varj.startsWith("a") ? varj+"i" : varj.startsWith("STMP") ? "0" : "");
			tmp = tmp.replace("%LEN%",
				varj.startsWith("a") ? "alen" : varj.startsWith("STMP") ? varj+".size()" : "");
		}
		sb.append(tmp);
		
		//mark as generated
		_generated = true;
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		switch(_type) {
			case PLUS_MULT:     return "t(+*)";
			case MINUS_MULT:    return "t(-*)";
			case BIASADD:       return "t(bias+)";
			case BIASMULT:      return "t(bias*)";
			case REPLACE:
			case REPLACE_NAN:   return "t(rplc)";
			case IFELSE:        return "t(ifelse)";
			case LOOKUP_RC1:    return "u(ixrc1)";
			case LOOKUP_RVECT1: return "u(ixrv1)";
			default:            return super.toString();
		}
	}
	
	@Override
	public void setOutputDims() {
		switch(_type) {
			case PLUS_MULT: 
			case MINUS_MULT:
			case BIASADD:
			case BIASMULT:
			case REPLACE:
			case REPLACE_NAN:
			case IFELSE:
			case LOOKUP_RC1:
				_rows = 0;
				_cols = 0;
				_dataType= DataType.SCALAR;
				break;
			case LOOKUP_RVECT1:
				_rows = 1;
				_cols = _inputs.get(0)._cols;
				_dataType= DataType.MATRIX;
				break;
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
		if( !(o instanceof CNodeTernary) )
			return false;
		
		CNodeTernary that = (CNodeTernary) o;
		return super.equals(that)
			&& _type == that._type;
	}
	@Override
	public boolean isSupported(GeneratorAPI api) {
		boolean is_supported = (api == GeneratorAPI.CUDA || api == GeneratorAPI.JAVA);
		int i = 0;
		while(is_supported && i < _inputs.size()) {
			CNode in = _inputs.get(i++);
			is_supported = in.isSupported(api);
		}
		return  is_supported;
	}
}

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

import java.util.ArrayList;

import org.apache.sysml.hops.codegen.template.TemplateUtils;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.util.UtilFunctions;

public class CNodeNary extends CNode
{
	public enum NaryType {
		VECT_CBIND;
		public static boolean contains(String value) {
			for( NaryType bt : values() )
				if( bt.name().equals(value) )
					return true;
			return false;
		}
		public String getTemplate(boolean sparseGen, long len, ArrayList<CNode> inputs) {
			switch (this) {
				case VECT_CBIND:
					StringBuilder sb = new StringBuilder();
					sb.append("    double[] %TMP% = LibSpoofPrimitives.allocVector("+len+", true); //nary cbind\n");
					for( int i=0, off=0; i<inputs.size(); i++ ) {
						CNode input = inputs.get(i);
						boolean sparseInput = sparseGen && input instanceof CNodeData
							&& input.getVarname().startsWith("a");
						String varj = input.getVarname();
						String pos = (input instanceof CNodeData && input.getDataType().isMatrix()) ? 
								(!varj.startsWith("b")) ? varj+"i" : TemplateUtils.isMatrix(input) ? 
								varj + ".pos(rix)" : "0" : "0";
						sb.append( sparseInput ?
							"    LibSpoofPrimitives.vectWrite("+varj+"vals, %TMP%, "
								+varj+"ix, "+pos+", "+off+", "+input._cols+");\n" :
							"    LibSpoofPrimitives.vectWrite("+(varj.startsWith("b")?varj+".values(rix)":varj)
								+", %TMP%, "+pos+", "+off+", "+input._cols+");\n");
						off += input._cols;
					}
					return sb.toString();
				default:
					throw new RuntimeException("Invalid nary type: "+this.toString());
			}
		}
		public boolean isVectorPrimitive() {
			return this == VECT_CBIND;
		}
	}
	
	private final NaryType _type;
	
	public CNodeNary( CNode[] inputs, NaryType type ) {
		for( CNode in : inputs )
			_inputs.add(in);
		_type = type;
		setOutputDims();
	}

	public NaryType getType() {
		return _type;
	}
	
	@Override
	public String codegen(boolean sparse) {
		if( isGenerated() )
			return "";
		
		StringBuilder sb = new StringBuilder();
		
		//generate children
		for(CNode in : _inputs)
			sb.append(in.codegen(sparse));
		
		//generate nary operation (use sparse template, if data input)
		String var = createVarname();
		String tmp = _type.getTemplate(sparse, _cols, _inputs);
		tmp = tmp.replace("%TMP%", var);
		
		sb.append(tmp);
		
		//mark as generated
		_generated = true;
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		switch(_type) {
			case VECT_CBIND: return "n(cbind)";
			default:
				return "m("+_type.name().toLowerCase()+")";
		}
	}
	
	@Override
	public void setOutputDims() {
		switch(_type) {
			case VECT_CBIND:
				_rows = _inputs.get(0)._rows;
				_cols = 0;
				for(CNode in : _inputs)
					_cols += in._cols;
				_dataType = DataType.MATRIX;
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
		if( !(o instanceof CNodeNary) )
			return false;
		
		CNodeNary that = (CNodeNary) o;
		return super.equals(that)
			&& _type == that._type;
	}
}

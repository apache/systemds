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

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.apache.sysds.hops.codegen.template.TemplateUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.util.DnnUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;

public class CNodeNary extends CNode
{
	public enum NaryType {
		VECT_CBIND,
		VECT_MAX_POOL,
		VECT_AVG_POOL,
		VECT_IM2COL,
		VECT_CONV2DMM;
		
		public static boolean contains(String value) {
			for( NaryType bt : values() )
				if( bt.name().equals(value) )
					return true;
			return false;
		}
		public String getTemplate(boolean sparseGen, long len, ArrayList<CNode> inputs, GeneratorAPI api) {
			switch (this) {
				case VECT_CBIND:
					StringBuilder sb = new StringBuilder();
					sb.append("    double[] %TMP% = LibSpoofPrimitives.allocVector("+len+", true); //nary cbind\n");
					for( int i=0, off=0; i<inputs.size(); i++ ) {
						CNode input = inputs.get(i);
						boolean sparseInput = sparseGen && input instanceof CNodeData
							&& input.getVarname().startsWith("a");
						String varj = input.getVarname();
						if( input.getDataType()==DataType.MATRIX ) {
							String pos = (input instanceof CNodeData) ?
								!varj.startsWith("b") ? varj+"i" : varj + ".pos(rix)" : "0";
							sb.append( sparseInput ?
								"    LibSpoofPrimitives.vectWrite("+varj+"vals, %TMP%, "
									+varj+"ix, "+pos+", "+off+", "+input._cols+");\n" :
								varj.startsWith("STMP") ?
									"    LibSpoofPrimitives.vectWrite("+input._cols+", "+varj+".values(), %TMP%, "
										+varj+".indexes(), "+pos+", "+off+", "+varj+".size());\n" :
									"    LibSpoofPrimitives.vectWrite("+(varj.startsWith("b")?varj+".values(rix)":varj)
									+", %TMP%, "+pos+", "+off+", "+input._cols+");\n");
							off += input._cols;	
						}
						else { //e.g., col vectors -> scalars
							sb.append("    %TMP%["+off+"] = "+varj+";\n");
							off ++;
						}
					}
					return sb.toString();
				case VECT_MAX_POOL:
				case VECT_AVG_POOL: {
					String vectName = (this==VECT_MAX_POOL) ? "Maxpool" : "Avgpool";
					String paramStr = getDnnParameterString(inputs, true);
					return sparseGen ?
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1v%, %IN1i%, %POS1%, alen, len, "+paramStr+");\n" : 
						"    double[] %TMP% = LibSpoofPrimitives.vect"+vectName+"Write(%IN1%, %POS1%, %LEN%, "+paramStr+");\n";
				}
				case VECT_IM2COL: {
					String paramStr = getDnnParameterString(inputs, true);
					return sparseGen ?
						"    double[] %TMP% = LibSpoofPrimitives.vectIm2colWrite(%IN1v%, %IN1i%, %POS1%, alen, len, "+paramStr+");\n" : 
						"    double[] %TMP% = LibSpoofPrimitives.vectIm2colWrite(%IN1%, %POS1%, %LEN%, "+paramStr+");\n";
				}
				case VECT_CONV2DMM: {
					return "    double[] %TMP% = LibSpoofPrimitives.vectConv2dmmWrite(%IN2%, %IN1%, %POS2%, %POS1%, %LEN%, "
						+ getDnnParameterString(inputs, false) +");\n";
				}
				default:
					throw new RuntimeException("Invalid nary type: "+this.toString());
			}
		}
		public boolean isVectorPrimitive() {
			return this == VECT_CBIND || this == VECT_MAX_POOL || this == VECT_AVG_POOL
				|| this == VECT_IM2COL || this == NaryType.VECT_CONV2DMM;
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
	public String codegen(boolean sparse, GeneratorAPI api) {
		if( isGenerated() )
			return "";
				
		StringBuilder sb = new StringBuilder();
		
		//generate children
		for(CNode in : _inputs)
			sb.append(in.codegen(sparse, api));
		
		//generate nary operation (use sparse template, if data input)
		boolean lsparse = sparse && (_inputs.get(0) instanceof CNodeData
			&& _inputs.get(0).getVarname().startsWith("a")
			&& !_inputs.get(0).isLiteral());
		String var = createVarname();
		String tmp = _type.getTemplate(lsparse, _cols, _inputs, api);
		tmp = tmp.replace("%TMP%", var);
		
		//replace sparse and dense inputs
		String varj1 = _inputs.get(0).getVarname();
		String varj2 = _inputs.get(1).getVarname();
		tmp = (_type == NaryType.VECT_CONV2DMM) ?
			replaceBinaryPlaceholders(tmp, new String[]{varj1,varj2}, false, api) :
			replaceUnaryPlaceholders(tmp, varj1, false, api);
		
		sb.append(tmp);
		
		//mark as generated
		_generated = true;
		
		return sb.toString();
	}
	
	@Override
	public String toString() {
		switch(_type) {
			case VECT_CBIND:    return "n(cbind)";
			case VECT_MAX_POOL: return "n(maxpool)";
			case VECT_AVG_POOL: return "n(avgpool)";
			case VECT_IM2COL:   return "n(im2col)";
			case VECT_CONV2DMM: return "n(conv2dmm)";
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
			case VECT_MAX_POOL:
			case VECT_AVG_POOL: { //only stride 1, pad 0
				int C = Integer.parseInt(_inputs.get(6).getVarname());
				int H = Integer.parseInt(_inputs.get(7).getVarname());
				int W = Integer.parseInt(_inputs.get(8).getVarname());
				int R = Integer.parseInt(_inputs.get(11).getVarname());
				int S = Integer.parseInt(_inputs.get(12).getVarname());
				long P = DnnUtils.getP(H, R, 1, 0);
				long Q = DnnUtils.getQ(W, S, 1, 0);
				_rows = _inputs.get(0)._rows; //N
				_cols =  C * P * Q;
				_dataType = DataType.MATRIX;
				break;
			}
			case VECT_IM2COL:
				_rows = 1;
				_cols = -1;
				_dataType = DataType.MATRIX;
				break;
			case VECT_CONV2DMM: {
				int H = Integer.parseInt(_inputs.get(8).getVarname());
				int W = Integer.parseInt(_inputs.get(9).getVarname());
				int K = Integer.parseInt(_inputs.get(10).getVarname());
				int R = Integer.parseInt(_inputs.get(12).getVarname());
				int S = Integer.parseInt(_inputs.get(13).getVarname());
				long P = DnnUtils.getP(H, R, 1, 0);
				long Q = DnnUtils.getQ(W, S, 1, 0);
				_rows = _inputs.get(0)._rows; //N
				_cols = K * P * Q;
				_dataType = DataType.MATRIX;
			}
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

	@Override
	public boolean isSupported(GeneratorAPI api) {
		boolean is_supported = (api == GeneratorAPI.JAVA);
		int i = 0;
		while(is_supported && i < _inputs.size()) {
			CNode in = _inputs.get(i++);
			is_supported = in.isSupported(api);
		}
		return  is_supported;
	}

	private static String getDnnParameterString(List<CNode> inputs, boolean unary) {
		int off = unary ? 0 : 1;
		
		//extract and derive individual parameters
		int C = Integer.parseInt(inputs.get(off+6).getVarname());
		int H = Integer.parseInt(inputs.get(off+7).getVarname());
		int W = Integer.parseInt(inputs.get(off+8).getVarname());
		int K = Integer.parseInt(inputs.get(off+9).getVarname());
		int R = Integer.parseInt(inputs.get(off+11).getVarname());
		int S = Integer.parseInt(inputs.get(off+12).getVarname());
		int P = (int) DnnUtils.getP(H, R, 1, 0);
		int Q = (int) DnnUtils.getQ(W, S, 1, 0);
		
		//construct parameter string
		return "rix, " + StringUtils.join(
			new int[]{C, P, Q, K, R, S, H, W}, ',');
	}
	

	private String replaceBinaryPlaceholders(String tmp, String[] vars, boolean vectIn, GeneratorAPI api) {
		//replace sparse and dense inputs
		for( int j=0; j<2; j++ ) {
			String varj = vars[j];
			
			//replace sparse and dense inputs
			tmp = tmp.replace("%IN"+(j+1)+"v%", varj+"vals");
			tmp = tmp.replace("%IN"+(j+1)+"i%", varj+"ix");
//			tmp = tmp.replace("%IN"+(j+1)+"%", 
//				varj.startsWith("b") ? varj + ".values(rix)" : varj );
			tmp = tmp.replace("%IN"+(j+1)+"%",
				varj.startsWith("b") ? ((api == GeneratorAPI.JAVA) ? varj + ".values(rix)" :
					varj + ".vals(0)") : varj);
			
			//replace start position of main input
			tmp = tmp.replace("%POS"+(j+1)+"%", (_inputs.get(j) instanceof CNodeData 
				&& _inputs.get(j).getDataType().isMatrix()) ? !varj.startsWith("b") ? varj+"i" : 
				(TemplateUtils.isMatrix(_inputs.get(j)) && _type!=NaryType.VECT_CONV2DMM) ? varj + ".pos(rix)" : "0" : "0");
		}
		
		//replace length
		if( _inputs.get(0).getDataType().isMatrix() )
			tmp = tmp.replace("%LEN%", _inputs.get(0).getVectorLength(api));
		
		return tmp;
	}
}

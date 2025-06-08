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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.util.UtilFunctions;

import static org.apache.sysds.runtime.matrix.data.LibMatrixNative.isSinglePrecision;

public class CNodeData extends CNode 
{
	protected final long _hopID;
	protected String _name;
	private boolean _strictEquals;
	
	public CNodeData(Hop hop) {
		this(hop, hop.getDim1(), hop.getDim2(), hop.getDataType());
	}
	
	public CNodeData(Hop hop, long rows, long cols, DataType dt) {
		//note: previous rewrites might have created hops with equal name
		//hence, we also keep the hopID to uniquely identify inputs
		super();
		_name = hop.getName();
		_hopID = hop.getHopID();
		_rows = rows;
		_cols = cols;
		_dataType = dt;
	}
	
	public CNodeData(CNodeData node, String newName) {
		_name = newName;
		_hopID = node.getHopID();
		_rows = node.getNumRows();
		_cols = node.getNumCols();
		_dataType = node.getDataType();
	}

	public CNodeData(String name, long hopID, long rows, long cols, DataType dataType) {
		_name = name;
		_hopID = hopID;
		_rows = rows;
		_cols = cols;
		_dataType = dataType;
	}
	
	@Override
	public String getVarname() {
		if ("NaN".equals(_name))
			return "Double.NaN";
		else if ("Infinity".equals(_name))
			return "Double.POSITIVE_INFINITY";
		else if ("-Infinity".equals(_name))
			return "Double.NEGATIVE_INFINITY";
		else if ("true".equals(_name) || "false".equals(_name))
			return "true".equals(_name) ? "1d" : "0d";
		else
			return _name;
	}

	public String getVarname(GeneratorAPI api) {
		if(api == GeneratorAPI.JAVA) {
			if ("NaN".equals(_name))
				return "Double.NaN";
			else if ("Infinity".equals(_name))
				return "Double.POSITIVE_INFINITY";
			else if ("-Infinity".equals(_name))
				return "Double.NEGATIVE_INFINITY";
			else if ("true".equals(_name) || "false".equals(_name))
				return "true".equals(_name) ? "1d" : "0d";
			else
				return _name;
		}
		else if(api == GeneratorAPI.CUDA) {
			if ("NaN".equals(_name))
				return isSinglePrecision() ? "CUDART_NAN_F" : "CUDART_NAN";
			else if ("Infinity".equals(_name))
				return isSinglePrecision() ? "CUDART_INF_F" : "CUDART_INF";
			else if ("-Infinity".equals(_name))
				return isSinglePrecision() ? "-CUDART_INF_F" : "-CUDART_INF";
			else if ("true".equals(_name) || "false".equals(_name))
				return "true".equals(_name) ? "1" : "0";
			else if (CodegenUtils.isNumeric(_name))
				return isSinglePrecision() ? _name + ".0f" : _name + ".0";
			else
				return _name;
		}
		else
			throw new RuntimeException("Unknown GeneratorAPI: " + SpoofCompiler.API);
	}
	
	public long getHopID() {
		return _hopID;
	}
	
	public void setName(String name) {
		_name = name;
	}
	
	public void setStrictEquals(boolean flag) {
		_strictEquals = flag;
		_hash = 0;
	}
	
	@Override
	public String codegen(boolean sparse, GeneratorAPI api) {
		return "";
	}

	@Override
	public void setOutputDims() {
		
	}
	
	@Override
	public String toString() {
		return "data("+_name+", hopid="+_hopID+")";
	}
	
	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			_hash = UtilFunctions.intHashCode(
				super.hashCode(), (isLiteral() || !_strictEquals) ? 
				_name.hashCode() : Long.hashCode(_hopID));
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object o) {
		return (o instanceof CNodeData 
			&& super.equals(o)
			&& isLiteral() == ((CNode)o).isLiteral()
			&& ((isLiteral() || !_strictEquals) ? 
				_name.equals(((CNodeData)o)._name) : 
				_hopID == ((CNodeData)o)._hopID));
	}
	@Override
	public boolean isSupported(GeneratorAPI api) {
		return true;
	}
}

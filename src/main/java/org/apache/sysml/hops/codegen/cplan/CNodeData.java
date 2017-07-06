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

import org.apache.sysml.hops.Hop;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.util.UtilFunctions;

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
		
	@Override
	public String getVarname() {
		if( "NaN".equals(_name) )
			return "Double.NaN";
		else if( "Infinity".equals(_name) )
			return "Double.POSITIVE_INFINITY";
		else if( "-Infinity".equals(_name) )
			return "Double.NEGATIVE_INFINITY";
		else
			return _name;
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
	public String codegen(boolean sparse) {
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
			&& isLiteral() == ((CNodeData)o).isLiteral()
			&& (isLiteral() || !_strictEquals) ? 
				_name.equals(((CNodeData)o)._name) : 
				_hopID == ((CNodeData)o)._hopID);
	}
}

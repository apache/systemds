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

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.util.UtilFunctions;

public abstract class CNode
{
	private static final IDSequence _seqVar = new IDSequence();
	private static final IDSequence _seqID = new IDSequence();
	
	protected final long _ID; 
	protected ArrayList<CNode> _inputs = null; 
	protected CNode _output = null; 
	protected boolean _visited = false;
	protected boolean _generated = false;
	protected String _genVar = null;
	protected long _rows = -1;
	protected long _cols = -1;
	protected DataType _dataType;
	protected boolean _literal = false;
	
	//cached hash to allow memoization in DAG structures and repeated 
	//recursive hash computation over all inputs (w/ reset on updates)
	protected int _hash = 0;
	
	public CNode() {
		_ID = _seqID.getNextID();
		_inputs = new ArrayList<CNode>();
		_generated = false;
	}
	
	public long getID() {
		return _ID;
	}

	public ArrayList<CNode> getInput() {
		return _inputs;
	}
	
	public boolean isGenerated() {
		return _generated;
	}

	public void resetGenerated() {
		if( isGenerated() )
			for( CNode cn : _inputs )
				cn.resetGenerated();
		_generated = false;
	}
	
	public String createVarname() {
		_genVar = "TMP"+_seqVar.getNextID();
		return _genVar; 
	}
	
	protected static String getCurrentVarName() {
		return "TMP"+(_seqVar.getCurrentID()-1);
	}
	
	public String getVarname() {
		return _genVar;
	}
	
	public String getVectorLength() {
		if( getVarname().startsWith("a") )
			return "len";
		else if( getVarname().startsWith("b") )
			return getVarname()+".clen";
		else if( _dataType==DataType.MATRIX )
			return getVarname()+".length";
		return "";
	}
	
	public String getClassname() {
		return getVarname();
	}
	
	public void resetHash() {
		_hash = 0;
	}
	
	public void setNumRows(long rows) {
		_rows = rows;
	}
	
	public long getNumRows() {
		return _rows;
	}
	
	public void setNumCols(long cols) {
		_cols = cols;
	}
	
	public long getNumCols() {
		return _cols;
	}
	
	public DataType getDataType() {
		return _dataType;
	}
	
	public void setDataType(DataType dt) {
		_dataType = dt;
		_hash = 0;
	}
	
	public boolean isLiteral() {
		return _literal;
	}
	
	public void setLiteral(boolean literal) {
		_literal = literal;
		_hash = 0;
	}
	
	public CNode getOutput() {
		return _output;
	}
	
	public void setOutput(CNode output) {
		_output = output;
		_hash = 0;
	}
	
	public boolean isVisited() {
		return _visited; 
	}
	
	public void setVisited() {
		setVisited(true);
	}
	
	public void setVisited(boolean flag) {
		_visited = flag;
	}
	
	public void resetVisitStatus() {
		if( !isVisited() )
			return;
		for( CNode h : getInput() )
			h.resetVisitStatus();
		setVisited(false);
	}
	
	public abstract String codegen(boolean sparse);
	
	public abstract void setOutputDims();
	
	///////////////////////////////////////
	// Functionality for plan cache
	
	//note: genvar/generated changed on codegen and not considered,
	//rows and cols also not include to increase reuse potential
	
	@Override
	public int hashCode() {
		if( _hash == 0 ) {
			//include inputs, partitioned by matrices and scalars to increase 
			//reuse in case of interleaved inputs (see CNodeTpl.renameInputs)
			int h = 1;
			for( CNode c : _inputs )
				if( c.getDataType()==DataType.MATRIX )
					h = UtilFunctions.intHashCode(h, c.hashCode());
			for( CNode c : _inputs )
				if( c.getDataType()!=DataType.MATRIX )
					h = UtilFunctions.intHashCode(h, c.hashCode());
			h = UtilFunctions.intHashCode(h, (_output!=null)?_output.hashCode():0);
			h = UtilFunctions.intHashCode(h, (_dataType!=null)?_dataType.hashCode():0);
			h = UtilFunctions.intHashCode(h, Boolean.hashCode(_literal));
			_hash = h;
		}
		return _hash;
	}
	
	@Override 
	public boolean equals(Object that) {
		if( !(that instanceof CNode) )
			return false;
		
		CNode cthat = (CNode) that;
		boolean ret = _inputs.size() == cthat._inputs.size();
		for( int i=0; i<_inputs.size() && ret; i++ )
			ret &= _inputs.get(i).equals(cthat._inputs.get(i));
		return ret 
			&& (_output == cthat._output || _output.equals(cthat._output))
			&& _dataType == cthat._dataType
			&& _literal == cthat._literal;
	}
}

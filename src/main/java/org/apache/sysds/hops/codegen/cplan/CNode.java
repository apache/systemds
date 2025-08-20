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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.codegen.SpoofCompiler.GeneratorAPI;
import org.apache.sysds.hops.codegen.template.TemplateUtils;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.util.ArrayList;

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
		_inputs = new ArrayList<>();
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
		if(_genVar == null)
			_genVar = "TMP"+_seqVar.getNextID();
		return _genVar; 
	}

	public String createVarname(boolean sparse) {
		if(!sparse) {
			return createVarname();
		} else {
			return _genVar = "S" + createVarname();
		}
	}
	
	public String getVarname() {
		return _genVar;
	}

	public String getVarname(GeneratorAPI api) { return getVarname(); }

	public String getVectorLength(GeneratorAPI api) {
		if(api == GeneratorAPI.CUDA) {
			if( getVarname().startsWith("a") )
				return "a.cols()";
			if(getVarname().startsWith("b"))
				return getVarname()+".cols()";
			else
				return getVarname()+".length";
		}
		else {
			if( getVarname().startsWith("a") )
				return "len";
			if(getVarname().startsWith("b"))
				return getVarname() + ".clen";
			else if(getVarname().startsWith("STMP"))
				return "len";
			else if(_dataType == DataType.MATRIX)
				return getVarname() + ".length";
		}
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
	
	public abstract String codegen(boolean sparse, GeneratorAPI api);
	
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
	
	protected String replaceUnaryPlaceholders(String tmp, String varj, boolean vectIn, GeneratorAPI api) {
		//replace sparse and dense inputs
		if(DMLScript.SPARSE_INTERMEDIATE) {
			tmp = tmp.replace("%IN1v%", varj.startsWith("STMP") ? varj+".values()" : varj+"vals");
			tmp = tmp.replace("%IN1i%", varj.startsWith("STMP") ? varj+".indexes()" :varj+"ix");
		} else {
			tmp = tmp.replace("%IN1v%", varj+"vals");
			tmp = tmp.replace("%IN1i%", varj+"ix");
		}
		tmp = tmp.replace("%IN1%", 
			(vectIn && TemplateUtils.isMatrix(_inputs.get(0))) ? 
				((api == GeneratorAPI.JAVA) ? varj + ".values(rix)" : varj + ".vals(0)" ) :
				(vectIn && TemplateUtils.isRowVector(_inputs.get(0)) ? 
					((api == GeneratorAPI.JAVA) ? varj + ".values(0)" : varj + ".val(0)") :
						(varj.startsWith("a") || TemplateUtils.isMatrix(_inputs.get(0))) ?
								(api == GeneratorAPI.JAVA ? varj : varj + ".vals(0)") : varj));
		
		//replace start position of main input
		String spos = (_inputs.get(0) instanceof CNodeData 
			&& _inputs.get(0).getDataType().isMatrix()) ? !varj.startsWith("b") ? 
			varj+"i" : TemplateUtils.isMatrix(_inputs.get(0)) ? varj + ".pos(rix)" : "0" : "0";
		
		tmp = tmp.replace("%POS1%", spos);
		tmp = tmp.replace("%POS2%", spos);
		
		//replace length
		if( _inputs.get(0).getDataType().isMatrix() )
			tmp = tmp.replace("%LEN%", _inputs.get(0).getVectorLength(api));
		
		return tmp;
	}

	protected CodeTemplate getLanguageTemplateClass(CNode caller, GeneratorAPI api) {
		switch (api) {
			case CUDA:
				if(caller instanceof CNodeBinary)
					return new org.apache.sysds.hops.codegen.cplan.cuda.Binary();
				else if(caller instanceof CNodeTernary)
					return new org.apache.sysds.hops.codegen.cplan.cuda.Ternary();
				else if(caller instanceof CNodeUnary)
					return new org.apache.sysds.hops.codegen.cplan.cuda.Unary();
				else return null;
			case JAVA: 
				if(caller instanceof CNodeBinary)
					return new org.apache.sysds.hops.codegen.cplan.java.Binary();
				else if(caller instanceof CNodeTernary)
					return new org.apache.sysds.hops.codegen.cplan.java.Ternary();
				else if(caller instanceof CNodeUnary)
					return new org.apache.sysds.hops.codegen.cplan.java.Unary();
				else return null;
			default:
				throw new RuntimeException("API not supported by code generator: " + api.toString());
		}
	}
	
	protected String getLanguageTemplate(CNode caller, GeneratorAPI api) {
		switch (api) {
			case CUDA:
				if(caller instanceof CNodeCell)
					return CodeTemplate.getTemplate("/cuda/spoof/cellwise.cu");
				else if(caller instanceof CNodeRow)
					return CodeTemplate.getTemplate("/cuda/spoof/rowwise.cu");
				else return null;
			case JAVA:
				if(caller instanceof CNodeCell)
					return CNodeCell.JAVA_TEMPLATE;
				else if(caller instanceof CNodeRow)
					return CNodeRow.JAVA_TEMPLATE;
				else return null;
			default:
				throw new RuntimeException("API not supported by code generator: " + api.toString());
		}
	}
	
	public abstract boolean isSupported(GeneratorAPI api);
	
	public void setVarName(String name) { _genVar = name; }
}

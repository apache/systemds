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

package org.apache.sysds.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.PartitionFormat;

/**
 * Internal representation of a plan alternative for program blocks and instructions 
 * in order to enable efficient and simple recursive enumeration and plan changes.
 * This is only used within the optimizer and therefore not visible to any other component.
 * 
 */
public class OptNode 
{
	public enum NodeType{
		GENERIC,
		FUNCCALL,
		IF,
		WHILE,
		FOR,
		PARFOR,
		INST,
		HOP;
		public boolean isLoop() {
			return this == WHILE ||
				this == FOR || this == PARFOR;
		}
	}
	
	public enum ExecType { 
		CP,
		SPARK;

		public ParForProgramBlock.PExecMode toParForExecMode() {
			switch( this ) {
				case CP:    return ParForProgramBlock.PExecMode.LOCAL;
				case SPARK: return ParForProgramBlock.PExecMode.REMOTE_SPARK;
			}
			
			return null;
		}
	}
	
	public enum ParamType{
		OPSTRING,
		TASK_PARTITIONER,
		TASK_SIZE,
		DATA_PARTITIONER,
		DATA_PARTITION_FORMAT,
		DATA_PARTITION_COND,
		DATA_PARTITION_COND_MEM,
		RESULT_MERGE,
		NUM_ITERATIONS,
		RECURSIVE_CALL
	}

	//child nodes
	private ArrayList<OptNode>        _childs  = null;

	//node configuration 
	private long                      _id      = -1;
	private NodeType                  _ntype   = null;
	private ExecType                  _etype   = null;
	private int                       _k       = -1;
	private HashMap<ParamType,String> _params  = null;
	
	//line numbers (for explain)
	private int                       _beginLine = -1;
	private int                       _endLine = -1;
	
	public OptNode( NodeType type ) {
		this(type, null);
	}

	public OptNode( NodeType ntype, ExecType etype ) {
		_ntype = ntype;
		_etype = etype;
		_k = 1;
	}
	
	///////
	//getters and setters
	
	public NodeType getNodeType() {
		return _ntype;
	}
	
	public void setNodeType(NodeType type) {
		_ntype = type;
	}
	
	public boolean isNodeType(NodeType... types) {
		return ArrayUtils.contains(types, _ntype);
	}
	
	public ExecType getExecType() {
		return _etype;
	}
	
	public void setExecType(ExecType type) {
		_etype = type;
	}
	
	public void setID( long id ) {
		_id = id;
	}
	
	public long getID( ) {
		return _id;
	}
	
	public void addParam(ParamType ptype, String val) {
		if( _params == null )
			_params = new HashMap<>();
		_params.put(ptype, val);
	}

	public void setParams( HashMap<ParamType,String> params ) {
		_params = params;
	}
	
	public String getParam( ParamType type ) {
		return (_params != null) ?
			_params.get(type) : null;
	}
	
	public int getBeginLine() {
		return _beginLine;
	}
	
	public void setBeginLine( int line ) {
		_beginLine = line;
	}
	
	public int getEndLine() {
		return _endLine;
	}
	
	public void setEndLine( int line ) {
		_endLine = line;
	}

	public void setLineNumbers( int begin, int end ) {
		setBeginLine( begin );
		setEndLine( end );
	}
	
	public void addChild( OptNode child ) {
		if( _childs==null )
			_childs = new ArrayList<>();
		_childs.add( child );
	}
	
	public void addChilds( ArrayList<OptNode> childs ) {
		if( _childs==null )
			_childs = new ArrayList<>();
		_childs.addAll( childs );
	}
	
	public void setChilds(ArrayList<OptNode> childs) {
		_childs = childs;
	}
	
	public ArrayList<OptNode> getChilds() {
		return _childs;
	}
	
	public int getK() {
		return _k;
	}

	public void setK(int k) {
		_k = k;
	}

	public boolean exchangeChild(OptNode oldNode, OptNode newNode) {
		if( isLeaf() )
			return false;
		
		boolean ret = false;
		for( int i=0; i<_childs.size(); i++ )
			if( _childs.get(i) == oldNode ) {
				_childs.set(i, newNode);
				ret = true;
			}
		return ret;
	}

	public boolean isLeaf() {
		return ( _childs == null || _childs.isEmpty() );
	}

	public boolean hasOnlySimpleChilds() {
		if( isLeaf() )
			return true;
		
		boolean ret = true;
		for( OptNode n : _childs ) {
			if( n.getNodeType()==NodeType.GENERIC )
				ret &= n.hasOnlySimpleChilds();
			//functions, loops, branches
			else if( n.getNodeType()!=NodeType.HOP )
				return false;
		}
		return ret;
	}

	public String getInstructionName() {
		return String.valueOf(_etype) + Lop.OPERAND_DELIMITOR + getParam(ParamType.OPSTRING);
	}

	public boolean isRecursive() {
		String rec = getParam(ParamType.RECURSIVE_CALL);
		return (rec != null) ?
			Boolean.parseBoolean(rec) : false;
	}
	

	///////
	//recursive methods

	public Collection<OptNode> getNodeList() {
		Collection<OptNode> nodes = new ArrayList<>();
		if(!isLeaf())
			for( OptNode n : _childs )
				nodes.addAll( n.getNodeList() );
		nodes.add(this);
		return nodes;
	}

	public Collection<OptNode> getNodeList( ExecType et ) {
		Collection<OptNode> nodes = new ArrayList<>();
		if(!isLeaf())
			for( OptNode n : _childs )
				nodes.addAll( n.getNodeList( et ) );
		if( _etype == et )
			nodes.add(this);
		return nodes;
	}

	public Collection<OptNode> getRelevantNodeList() {
		Collection<OptNode> nodes = new ArrayList<>();
		if( !isLeaf() )
			for( OptNode n : _childs )
				nodes.addAll( n.getRelevantNodeList() );
		if( _ntype == NodeType.PARFOR || _ntype == NodeType.HOP )
			nodes.add(this);
		return nodes;
	}
	
	
	/**
	 * Set the plan to a parallel degree of 1 (serial execution).
	 */
	public void setSerialParFor() {
		//process parfor nodes
		if( _ntype == NodeType.PARFOR ) {
			_k = 1;
			_etype = ExecType.CP;
		}
		
		//process childs
		if( !isLeaf() )
			for( OptNode n : _childs )
				n.setSerialParFor();
	}

	/**
	 * Gets the number of plan nodes.
	 * 
	 * @return number of plan nodes
	 */
	public int size() {
		int count = 1; //self
		if( !isLeaf() )
			for( OptNode n : _childs )
				count += n.size();
		return count;
	}
	
	/**
	 * Determines if all program blocks and instructions exhibit 
	 * the execution type CP. 
	 * 
	 * @return true of all program blocks and instructions execute on CP
	 */
	public boolean isCPOnly() {
		boolean ret = (_etype == ExecType.CP);
		if( !isLeaf() )
			for( OptNode n : _childs ) {
				if( !ret ) break; //early abort if already false
				ret &= n.isCPOnly();
			}
		return ret;
	}

	public int getTotalK() {
		int k = 1;
		if( !isLeaf() )
			for( OptNode n : _childs )
				k = Math.max(k, n.getTotalK() );
		
		if( _ntype == NodeType.PARFOR ) {
			if( _etype==ExecType.CP )
				k = _k * k;
			else //MR
				k = 1;
		}
		
		return k;
	}

	public long getMaxC( long N ) {
		long maxc = N;
		if( !isLeaf() )
			for( OptNode n : _childs )
				maxc = Math.min(maxc, n.getMaxC( N ));
		
		if( _ntype == NodeType.HOP ) {
			String ts = getParam(ParamType.TASK_SIZE);
			if( ts != null )
				maxc = Math.min(maxc, Integer.parseInt(ts));
		}
		
		if( _ntype == NodeType.PARFOR && _etype == ExecType.CP)
			maxc = maxc / _k; //intdiv
		
		return maxc;
	}

	public boolean hasNestedParallelism( boolean flagNested ) {
		boolean ret = false;
		//check for parfor in nested context
		if( _ntype == NodeType.PARFOR ) {
			if( flagNested ) 
				return true;
			flagNested = true;
		}
		
		//recursively process children
		if( !isLeaf() )
			for( int i=0; i<_childs.size() && !ret; i++ )
				ret |= _childs.get(i).hasNestedParallelism( flagNested );
		return ret;
	}

	public boolean hasNestedPartitionReads( boolean flagNested ) {
		if( isLeaf() ) {
			//partitioned read identified by selected partition format
			String tmp = getParam(ParamType.DATA_PARTITION_FORMAT);
			return ( tmp !=null && flagNested
				&& PartitionFormat.valueOf(tmp)._dpf!=PDataPartitionFormat.NONE);
		}
		
		boolean ret = false;
		for( int i=0; i<_childs.size() && !ret; i++ ) {
			OptNode n = _childs.get(i);
			if( n._ntype.isLoop() )
				flagNested = true;
			ret |= n.hasNestedPartitionReads( flagNested );
		}
		return ret;
	}

	public void checkAndCleanupLeafNodes() {
		if( isLeaf() )
			return;
		Iterator<OptNode> iter = _childs.iterator();
		while( iter.hasNext() ) {
			OptNode n = iter.next();
			n.checkAndCleanupLeafNodes();
			if( n.isLeaf() && n._ntype != NodeType.HOP && n._ntype != NodeType.INST 
				&& n._ntype != NodeType.FUNCCALL ) {
				iter.remove();
			}
		}
	}

	public void checkAndCleanupRecursiveFunc(Set<String> stack) {
		//recursive invocation
		if( !isLeaf() )
			for( OptNode n : _childs )
				n.checkAndCleanupRecursiveFunc( stack );
	
		//collect and update func info
		if(_ntype == NodeType.FUNCCALL) {
			String rec = getParam(ParamType.RECURSIVE_CALL);
			String fname = getParam(ParamType.OPSTRING);
			if( rec != null && Boolean.parseBoolean(rec) ) 
				stack.add(fname); //collect
			else if( stack.contains(fname) )
				addParam(ParamType.RECURSIVE_CALL, "true");
		}
	}
	
	/**
	 * Explain tool: prints the hierarchical plan to <code>stdout</code>.
	 * 
	 * @param level depth to print?
	 * @param withDetails if true, explain details
	 * @return string explanation
	 */
	public String explain(int level, boolean withDetails) 
	{
		StringBuilder sb = new StringBuilder();
		for( int i=0; i<level; i++ )
			sb.append("--");	
		if( _ntype == NodeType.INST || _ntype == NodeType.HOP ) //leaf nodes
		{
			sb.append(_params.get(ParamType.OPSTRING));
		}
		else //non-leaf nodes
		{
			sb.append(_ntype);
			if( _beginLine>0 && _endLine>0 ) { //known lines
				sb.append(" (lines ");
				sb.append(_beginLine);
				sb.append("-");
				sb.append(_endLine);
				sb.append(")");
			}
		}
		sb.append(", exec=");
		sb.append(_etype);
		sb.append(", k=");
		sb.append(_k);
		switch( _ntype ) //specific details
		{	
			case PARFOR: {
				sb.append(", dp="); //data partitioner
				sb.append(_params.get(ParamType.DATA_PARTITIONER));
				sb.append(", tp="); //task partitioner
				sb.append(_params.get(ParamType.TASK_PARTITIONER));
				sb.append(", rm="); //result merge
				sb.append(_params.get(ParamType.RESULT_MERGE));
				break;
			}
			case FUNCCALL: {
				sb.append(", name=");
				sb.append(_params.get(ParamType.OPSTRING));
				if( _params.get(ParamType.RECURSIVE_CALL)!=null && Boolean.parseBoolean(_params.get(ParamType.RECURSIVE_CALL)) )
					sb.append(", recursive");
				break;
			}	
			default:
				//do nothing
		}
		sb.append("\n");
		
		if( _childs != null )
			for( OptNode n : _childs )
				sb.append( n.explain(level+1, withDetails) );
		
		return sb.toString();
	}

	/**
	 * Determines the maximum problem size, in terms of the maximum
	 * total number of inner loop iterations, of the entire subtree.
	 * 
	 * @return maximum problem size
	 */
	public long getMaxProblemSize() {
		//recursively process children
		long max = 1;
		if( !isLeaf() )
			for( OptNode n : _childs )
				max = Math.max(max, n.getMaxProblemSize());
		
		//scale problem size by number of loop iterations
		if( _ntype.isLoop() && !isLeaf() ) {
			String numIter = getParam(ParamType.NUM_ITERATIONS);
			max *= (numIter != null) ? Long.parseLong(numIter) :
				CostEstimator.FACTOR_NUM_ITERATIONS;
		}
		return max;
	}
}

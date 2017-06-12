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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysml.hops.codegen.SpoofFusedOp.SpoofOutputDimsType;
import org.apache.sysml.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysml.hops.codegen.template.TemplateUtils;
import org.apache.sysml.parser.Expression.DataType;

public abstract class CNodeTpl extends CNode implements Cloneable
{
	private int _beginLine = -1;
	
	public CNodeTpl(ArrayList<CNode> inputs, CNode output ) {
		if(inputs.size() < 1)
			throw new RuntimeException("Cannot pass empty inputs to the CNodeTpl");

		for(CNode input : inputs)
			addInput(input);
		_output = output;	
	}
	
	public void addInput(CNode in) {
		//check for duplicate entries or literals
		if( containsInput(in) || in.isLiteral() )
			return;
		
		_inputs.add(in);
	}
	
	public void cleanupInputs(HashSet<Long> filter) {
		ArrayList<CNode> tmp = new ArrayList<CNode>();
		for( CNode in : _inputs )
			if( in instanceof CNodeData && filter.contains(((CNodeData) in).getHopID()) )
				tmp.add(in);
		_inputs = tmp;
	}
	
	public String[] getInputNames() {
		String[] ret = new String[_inputs.size()];
		for( int i=0; i<_inputs.size(); i++ )
			ret[i] = _inputs.get(i).getVarname();
		return ret;
	}
	
	public void resetVisitStatusOutputs() {
		getOutput().resetVisitStatus();
	}
	
	public String codegen() {
		return codegen(false);
	}
	
	public abstract CNodeTpl clone();
	
	public abstract SpoofOutputDimsType getOutputDimType();
	
	public abstract String getTemplateInfo();
	
	protected void renameInputs(ArrayList<CNode> inputs, int startIndex) {
		renameInputs(Collections.singletonList(_output), inputs, startIndex);
	}
	
	protected void renameInputs(List<CNode> outputs, ArrayList<CNode> inputs, int startIndex) {
		//create map of hopID to data nodes with new names, used for CSE
		HashMap<Long, CNode> nodes = new HashMap<Long, CNode>();
		for(int i=startIndex, sPos=0, mPos=0; i < inputs.size(); i++) {
			CNode cnode = inputs.get(i);
			if( cnode instanceof CNodeData && ((CNodeData)cnode).isLiteral() )
				continue;
			CNodeData cdata = (CNodeData)cnode;
			if( cdata.getDataType() == DataType.SCALAR  || ( cdata.getNumCols() == 0 && cdata.getNumRows() == 0) ) 
				nodes.put(cdata.getHopID(), new CNodeData(cdata, "scalars["+ mPos++ +"]"));
			else
				nodes.put(cdata.getHopID(), new CNodeData(cdata, "b["+ sPos++ +"]"));
		}
		
		//single pass to replace all names
		for( CNode output : outputs )
			rReplaceDataNode(output, nodes, new HashMap<Long, CNode>());
	}
	
	protected void rReplaceDataNode( CNode root, CNode input, String newName ) {
		if( !(input instanceof CNodeData) )
			return;
			
		//create temporary name mapping
		HashMap<Long, CNode> names = new HashMap<Long, CNode>();
		CNodeData tmp = (CNodeData)input;
		names.put(tmp.getHopID(), new CNodeData(tmp, newName));
		
		rReplaceDataNode(root, names, new HashMap<Long,CNode>());
	}
	
	protected void rReplaceDataNode( ArrayList<CNode> roots, CNode input, String newName ) {
		if( !(input instanceof CNodeData) )
			return;
			
		//create temporary name mapping
		HashMap<Long, CNode> names = new HashMap<Long, CNode>();
		CNodeData tmp = (CNodeData)input;
		names.put(tmp.getHopID(), new CNodeData(tmp, newName));
		
		for( CNode root : roots )
			rReplaceDataNode(root, names, new HashMap<Long,CNode>());
	}
	
	/**
	 * Recursively searches for data nodes and replaces them if found.
	 * 
	 * @param node current node in recursive descend
	 * @param dnodes prepared data nodes, identified by own hop id
	 * @param lnodes memoized lookup nodes, identified by data node hop id
	 */
	protected void rReplaceDataNode( CNode node, HashMap<Long, CNode> dnodes, HashMap<Long, CNode> lnodes ) 
	{	
		for( int i=0; i<node._inputs.size(); i++ ) {
			//recursively process children
			rReplaceDataNode(node._inputs.get(i), dnodes, lnodes);
			
			//replace leaf data node
			if( node._inputs.get(i) instanceof CNodeData ) {
				CNodeData tmp = (CNodeData)node._inputs.get(i);
				if( dnodes.containsKey(tmp.getHopID()) )
					node._inputs.set(i, dnodes.get(tmp.getHopID()));
			}
			
			//replace lookup on top of leaf data node (for CSE only)
			if( node._inputs.get(i) instanceof CNodeUnary
				&& node._inputs.get(i)._inputs.get(0) instanceof CNodeData
				&& (((CNodeUnary)node._inputs.get(i)).getType()==UnaryType.LOOKUP_R
				|| ((CNodeUnary)node._inputs.get(i)).getType()==UnaryType.LOOKUP_RC)) {
				CNodeData tmp = (CNodeData)node._inputs.get(i)._inputs.get(0);
				if( !lnodes.containsKey(tmp.getHopID()) )
					lnodes.put(tmp.getHopID(), node._inputs.get(i));
				else
					node._inputs.set(i, lnodes.get(tmp.getHopID()));	
			}
		}
	}
	
	public void rReplaceDataNode( CNode node, long hopID, CNode newNode ) 
	{
		for( int i=0; i<node._inputs.size(); i++ ) {
			//replace leaf node
			if( node._inputs.get(i) instanceof CNodeData ) {
				CNodeData tmp = (CNodeData)node._inputs.get(i);
				if( tmp.getHopID() == hopID )
					node._inputs.set(i, newNode);
			}
			//recursively process children
			rReplaceDataNode(node._inputs.get(i), hopID, newNode);
			
			//remove unnecessary lookups
			if( node._inputs.get(i) instanceof CNodeUnary 
				&& (((CNodeUnary)node._inputs.get(i)).getType()==UnaryType.LOOKUP_R
				|| ((CNodeUnary)node._inputs.get(i)).getType()==UnaryType.LOOKUP_RC)
				&& node._inputs.get(i)._inputs.get(0).getDataType()==DataType.SCALAR)
				node._inputs.set(i, node._inputs.get(i)._inputs.get(0));
		}
	}
	
	public void rInsertLookupNode( CNode node, long hopID, HashMap<Long, CNode> memo, UnaryType lookupType ) 
	{
		for( int i=0; i<node._inputs.size(); i++ ) {
			//recursively process children
			rInsertLookupNode(node._inputs.get(i), hopID, memo, lookupType);
			
			//replace leaf node
			if( node._inputs.get(i) instanceof CNodeData ) {
				CNodeData tmp = (CNodeData)node._inputs.get(i);
				if( tmp.getHopID() == hopID ) {
					//use memo structure to retain DAG structure
					CNode lookup = memo.get(hopID);
					if( lookup == null && !TemplateUtils.isLookup(node) ) {
						lookup = new CNodeUnary(tmp, lookupType);
						memo.put(hopID, lookup);
					}
					else if( TemplateUtils.isLookup(node) )
						((CNodeUnary)node).setType(lookupType);
					else
						node._inputs.set(i, lookup);
				}
			}
		}
	}
	
	public void rReorderCommutativeBinaryOps(CNode node, long mainHopID) {
		if( isVisited() )
			return;
		for( CNode c : node.getInput() )
			rReorderCommutativeBinaryOps(c, mainHopID);
		if( node instanceof CNodeBinary && node.getInput().get(1) instanceof CNodeData
			&& ((CNodeData)node.getInput().get(1)).getHopID() == mainHopID
			&& ((CNodeBinary)node).getType().isCommutative() ) {
			CNode tmp = node.getInput().get(0);
			node.getInput().set(0, node.getInput().get(1));
			node.getInput().set(1, tmp);
		}
		setVisited();
	}
	
	/**
	 * Checks for duplicates (object ref or varname).
	 * 
	 * @param input new input node
	 * @return true if duplicate, false otherwise
	 */
	private boolean containsInput(CNode input) {
		if( !(input instanceof CNodeData) )
			return false;
		
		CNodeData input2 = (CNodeData)input;
		for( CNode cnode : _inputs ) {
			if( !(cnode instanceof CNodeData) )
				continue;
			CNodeData cnode2 = (CNodeData)cnode;
			if( cnode2._name.equals(input2._name) && cnode2._hopID==input2._hopID )
				return true;
		}
		
		return false;
	}
	
	public void setBeginLine(int line) {
		_beginLine = line;
	}
	
	public int getBeginLine() {
		return _beginLine;
	}
	
	@Override
	public int hashCode() {
		return super.hashCode();
	}
	
	@Override 
	public boolean equals(Object o) {
		return (o instanceof CNodeTpl
			&& super.equals(o));
	}
	
	protected static boolean equalInputReferences(CNode current1, CNode current2, ArrayList<CNode> input1, ArrayList<CNode> input2) {
		boolean ret = (current1.getInput().size() == current2.getInput().size());
		
		//process childs recursively
		for( int i=0; ret && i<current1.getInput().size(); i++ )
			ret &= equalInputReferences(
					current1.getInput().get(i), current2.getInput().get(i), input1, input2);
		
		if( ret && current1 instanceof CNodeData ) {
			ret &= current2 instanceof CNodeData
				&& indexOf(input1, (CNodeData)current1)
				== indexOf(input2, (CNodeData)current2);
		}
		
		return ret;
	}
	
	protected static boolean equalInputReferences(ArrayList<CNode> current1, ArrayList<CNode> current2, ArrayList<CNode> input1, ArrayList<CNode> input2) {
		boolean ret = (current1.size() == current2.size());
		for( int i=0; ret && i<current1.size(); i++ )
			ret &= equalInputReferences(current1.get(i), current2.get(i), input1, input2);
		return ret;
	}
	
	private static int indexOf(ArrayList<CNode> inputs, CNodeData probe) {
		for( int i=0; i<inputs.size(); i++ ) {
			CNodeData cd = ((CNodeData)inputs.get(i));
			if( cd.getHopID()==probe.getHopID() )
				return i;
		}
		return -1;
	}
}

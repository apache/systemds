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
	
	public String[] getInputNames() {
		String[] ret = new String[_inputs.size()];
		for( int i=0; i<_inputs.size(); i++ )
			ret[i] = _inputs.get(i).getVarname();
		return ret;
	}
	
	public HashSet<Long> getInputHopIDs(boolean inclLiterals) {
		HashSet<Long> ret = new HashSet<Long>();
		for( CNode input : _inputs )
			if( !input.isLiteral() || inclLiterals )
				ret.add(((CNodeData)input).getHopID());
		return ret;
	}
	
	public void resetVisitStatusOutputs() {
		getOutput().resetVisitStatus();
	}
	
	public static void resetVisitStatus(List<CNode> outputs) {
		for( CNode output : outputs )
			output.resetVisitStatus();
	}
	
	public String codegen() {
		return codegen(false);
	}
	
	public abstract CNodeTpl clone();
	
	public abstract SpoofOutputDimsType getOutputDimType();
	
	public abstract String getTemplateInfo();
	
	public abstract void renameInputs();
	
	protected void renameInputs(ArrayList<CNode> inputs, int startIndex) {
		renameInputs(Collections.singletonList(_output), inputs, startIndex);
	}
	
	protected void renameInputs(List<CNode> outputs, ArrayList<CNode> inputs, int startIndex) {
		//create map of hopID to new names used for code generation
		HashMap<Long, String> newNames = new HashMap<Long, String>();
		for(int i=startIndex, sPos=0, mPos=0; i < inputs.size(); i++) {
			CNode cnode = inputs.get(i);
			if( cnode instanceof CNodeData && ((CNodeData)cnode).isLiteral() )
				continue;
			newNames.put(((CNodeData)cnode).getHopID(), cnode.getDataType().isScalar() ?
				"scalars["+ mPos++ +"]" : "b["+ sPos++ +"]");
		}
		
		//single pass to replace all names
		resetVisitStatus(outputs);
		for( CNode output : outputs )
			rRenameDataNodes(output, newNames);
	}
	
	protected void rRenameDataNode( CNode root, CNode input, String newName ) {
		rRenameDataNode(Collections.singletonList(root), input, newName);
	}
	
	protected void rRenameDataNode( List<CNode> roots, CNode input, String newName ) {
		if( !(input instanceof CNodeData) )
			return;
		
		//create temporary name mapping
		HashMap<Long, String> newNames = new HashMap<Long, String>();
		newNames.put(((CNodeData)input).getHopID(), newName);
		
		//single pass to replace all names
		resetVisitStatus(roots);
		for( CNode root : roots )
			rRenameDataNodes(root, newNames);
	}
	
	protected void rRenameDataNodes( CNode node, HashMap<Long, String> newNames ) {
		if( node.isVisited() )
			return;
		
		//recursively process children
		for( CNode c : node.getInput() )
			rRenameDataNodes(c, newNames);
		
		//rename data node
		if( node instanceof CNodeData ) {
			CNodeData dnode = (CNodeData) node;
			if( newNames.containsKey(dnode.getHopID()) )
				dnode.setName(newNames.get(dnode.getHopID()));
		}
		
		node.resetHash();
		node.setVisited();
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

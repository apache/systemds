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

package org.apache.sysml.hops.codegen.template;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysml.hops.codegen.cplan.CNode;
import org.apache.sysml.hops.codegen.cplan.CNodeData;
import org.apache.sysml.hops.codegen.cplan.CNodeMultiAgg;
import org.apache.sysml.hops.codegen.cplan.CNodeTpl;

public class CPlanCSERewriter 
{
	public CNodeTpl eliminateCommonSubexpressions(CNodeTpl tpl) 
	{
		//Note: Compared to our traditional common subexpression elimination, on cplans, 
		//we don't have any parent references, and hence cannot use a collect-merge approach. 
		//In contrast, we exploit the hash signatures of cnodes as used in the plan cache. 
		//However, note that these signatures ignore input hops by default (for better plan 
		//cache hit rates), but are temporarily set to strict evaluation for this rewrite. 
		
		List<CNode> outputs = (tpl instanceof CNodeMultiAgg) ? 
			((CNodeMultiAgg)tpl).getOutputs() : 
			Arrays.asList(tpl.getOutput());
		
		//step 1: set data nodes to strict comparison
		HashSet<Long> memo = new HashSet<Long>();
		for( CNode out : outputs )
			rSetStrictDataNodeComparision(out, memo, true);
		
		//step 2: perform common subexpression elimination
		HashMap<CNode,CNode> cseSet = new HashMap<CNode,CNode>();
		memo.clear();
		for( CNode out : outputs )
			rEliminateCommonSubexpression(out, cseSet, memo);
		
		//step 3: reset data nodes to imprecise comparison
		memo.clear();
		for( CNode out : outputs )
			rSetStrictDataNodeComparision(out, memo, true);
		
		return tpl;
	}
	
	private void rEliminateCommonSubexpression(CNode current, HashMap<CNode,CNode> cseSet, HashSet<Long> memo) {
		//avoid redundant re-evaluation
		if( memo.contains(current.getID()) )
			return;
		
		//replace input with existing common subexpression
		for( int i=0; i<current.getInput().size(); i++ ) {
			CNode input = current.getInput().get(i);
			if( cseSet.containsKey(input) )
				current.getInput().set(i, cseSet.get(input));
		}
		
		//process inputs recursively
		for( CNode input : current.getInput() )
			rEliminateCommonSubexpression(input, cseSet, memo);
		
		//process node itself
		cseSet.put(current, current);
		memo.add(current.getID());
	}
	
	private void rSetStrictDataNodeComparision(CNode current, HashSet<Long> memo, boolean flag) {
		//avoid redundant re-evaluation
		if( memo.contains(current.getID()) )
			return;
		
		//process inputs recursively and node itself
		for( CNode input : current.getInput() ) {
			rSetStrictDataNodeComparision(input, memo, flag);
			input.resetHash();
		}
		if( current instanceof CNodeData )
			((CNodeData)current).setStrictEquals(flag);
		memo.add(current.getID());	
	}
}

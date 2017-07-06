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

import java.util.Collections;
import java.util.HashMap;
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
			Collections.singletonList(tpl.getOutput());
		
		//step 1: set data nodes to strict comparison
		tpl.resetVisitStatusOutputs();
		for( CNode out : outputs )
			rSetStrictDataNodeComparision(out, true);
		
		//step 2: perform common subexpression elimination
		HashMap<CNode,CNode> cseSet = new HashMap<CNode,CNode>();
		tpl.resetVisitStatusOutputs();
		for( CNode out : outputs )
			rEliminateCommonSubexpression(out, cseSet);
		
		//step 3: reset data nodes to imprecise comparison
		tpl.resetVisitStatusOutputs();
		for( CNode out : outputs )
			rSetStrictDataNodeComparision(out, true);
		tpl.resetVisitStatusOutputs();
		
		return tpl;
	}
	
	private void rEliminateCommonSubexpression(CNode current, HashMap<CNode,CNode> cseSet) {
		//avoid redundant re-evaluation
		if( current.isVisited() )
			return;
		
		//replace input with existing common subexpression
		for( int i=0; i<current.getInput().size(); i++ ) {
			CNode input = current.getInput().get(i);
			if( cseSet.containsKey(input) )
				current.getInput().set(i, cseSet.get(input));
		}
		
		//process inputs recursively
		for( CNode input : current.getInput() )
			rEliminateCommonSubexpression(input, cseSet);
		
		//process node itself
		cseSet.put(current, current);
		current.setVisited();
	}
	
	private void rSetStrictDataNodeComparision(CNode current, boolean flag) {
		//avoid redundant re-evaluation
		if( current.isVisited() )
			return;
		
		//process inputs recursively and node itself
		for( CNode input : current.getInput() ) {
			rSetStrictDataNodeComparision(input, flag);
			input.resetHash();
		}
		if( current instanceof CNodeData )
			((CNodeData)current).setStrictEquals(flag);
		current.setVisited();
	}
}

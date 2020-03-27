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

package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Rule: CommonSubexpressionElimination. For all statement blocks, 
 * eliminate common subexpressions within dags by merging equivalent
 * operators (same input, equal parameters) bottom-up. For the moment,
 * this only applies within a dag, later this should be extended across
 * statements block (global, inter-procedure). 
 */
public class RewriteCommonSubexpressionElimination extends HopRewriteRule
{
	private final boolean _mergeLeafs;
	
	public RewriteCommonSubexpressionElimination() {
		this( true ); //default full CSE
	}
	
	public RewriteCommonSubexpressionElimination( boolean mergeLeafs ) {
		_mergeLeafs = mergeLeafs;
	}
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
	{
		if( roots == null )
			return null;
		
		//CSE pass 1: merge leaf nodes by name
		int cseMerged = 0;
		if( _mergeLeafs ) {
			HashMap<String, Hop> dataops = new HashMap<>();
			HashMap<LiteralKey, Hop> literalops = new HashMap<>();
			for (Hop h : roots)
				cseMerged += rule_CommonSubexpressionElimination_MergeLeafs(h, dataops, literalops);
			Hop.resetVisitStatus(roots);
		}
		
		//CSE pass 2: bottom-up merge of inner nodes
		for (Hop h : roots) 
			cseMerged += rule_CommonSubexpressionElimination(h);
		
		if( cseMerged > 0 )
			LOG.debug("Common Subexpression Elimination - removed "+cseMerged+" operators.");
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
	{
		if( root == null )
			return null;
		
		//CSE pass 1: merge leaf nodes by name
		int cseMerged = 0;
		if( _mergeLeafs ) {
			HashMap<String, Hop> dataops = new HashMap<>();
			HashMap<LiteralKey, Hop> literalops = new HashMap<>();
			cseMerged += rule_CommonSubexpressionElimination_MergeLeafs(root, dataops, literalops);
			root.resetVisitStatus();
		}
		
		//CSE pass 2: bottom-up merge of inner nodes
		cseMerged += rule_CommonSubexpressionElimination(root);
		
		if( cseMerged > 0 )
			LOG.debug("Common Subexpression Elimination - removed "+cseMerged+" operators.");
		
		return root;
	}
	
	private int rule_CommonSubexpressionElimination_MergeLeafs( Hop hop,
		HashMap<String, Hop> dataops, HashMap<LiteralKey, Hop> literalops ) 
	{
		if( hop.isVisited() )
			return 0;
		
		int ret = 0;
		if( hop.getInput().isEmpty() //LEAF NODE
			|| HopRewriteUtils.isData(hop, OpOpData.TRANSIENTREAD) )
		{
			if( hop instanceof LiteralOp ) {
				LiteralKey key = new LiteralKey(hop.getValueType(), hop.getName());
				if( !literalops.containsKey(key) )
					literalops.put(key, hop);
			}
			else if( hop instanceof DataOp && ((DataOp)hop).isRead()
				&& !dataops.containsKey(hop.getName())) {
				dataops.put(hop.getName(), hop);
			} 
		}
		else //INNER NODE
		{
			//merge leaf nodes (data, literal)
			for( int i=0; i<hop.getInput().size(); i++ )
			{
				Hop hi = hop.getInput().get(i);
				LiteralKey litKey = new LiteralKey(hi.getValueType(), hi.getName());
				if( hi instanceof DataOp && ((DataOp)hi).isRead() && dataops.containsKey(hi.getName()) ) {
					//replace child node ref
					Hop tmp = dataops.get(hi.getName());
					if( tmp != hi ) { //if required
						tmp.getParent().add(hop);
						tmp.setVisited();
						hop.getInput().set(i, tmp);
						ret++;
					}
				}
				else if( hi instanceof LiteralOp && literalops.containsKey(litKey) ) {
					Hop tmp = literalops.get(litKey);
					//replace child node ref
					if( tmp != hi ){ //if required
						tmp.getParent().add(hop);
						tmp.setVisited();
						hop.getInput().set(i, tmp);
						ret++;
					}
				}
				
				//recursive invocation (direct return on merged nodes)
				ret += rule_CommonSubexpressionElimination_MergeLeafs(hi, dataops, literalops);
			}
		}
		hop.setVisited();
		return ret;
	}

	private int rule_CommonSubexpressionElimination( Hop hop ) 
	{
		if( hop.isVisited() )
			return 0;
		
		//step 1: merge childs recursively first
		int ret = 0;
		for(Hop hi : hop.getInput())
			ret += rule_CommonSubexpressionElimination(hi);	
		
		//step 2: merge parent nodes
		if( hop.getParent().size()>1 ) //multiple consumers
		{
			//for all pairs 
			for( int i=0; i<hop.getParent().size()-1; i++ )
				for( int j=i+1; j<hop.getParent().size(); j++ ) {
					Hop h1 = hop.getParent().get(i);
					Hop h2 = hop.getParent().get(j);
					
					if( h1==h2 ) {
						//do nothing, note: we should not remove redundant parent links
						//(otherwise rewrites would need to take this property into account) 
						
						//remove redundant h2 from parent list
						//hop.getParent().remove(j);
						//j--;
					}
					else if( h1.compare(h2) ) { //merge h2 into h1
						//remove h2 from parent list
						hop.getParent().remove(j);
						
						//replace h2 w/ h1 in h2-parent inputs
						ArrayList<Hop> parent = h2.getParent();
						for( Hop p : parent )
							for( int k=0; k<p.getInput().size(); k++ )
								if( p.getInput().get(k)==h2 ) {
									p.getInput().set(k, h1);
									h1.getParent().add(p);
									h1.setVisited();
								}
						
						//replace h2 w/ h1 in h2-input parents
						for( Hop in : h2.getInput() )
							in.getParent().remove(h2);
						
						ret++;
						j--;
					}
				}
		}
		
		hop.setVisited();
		return ret;
	}
	
	protected static class LiteralKey {
		private final int _vtType;
		private final String _name;
		
		public LiteralKey(ValueType vt, String name) {
			_vtType = vt.ordinal();
			_name = name;
		}
		@Override
		public int hashCode() {
			return UtilFunctions.longHashCode(_vtType, _name.hashCode());
		}
		@Override 
		public boolean equals(Object o) {
			return (o instanceof LiteralKey
				&& _vtType == ((LiteralKey)o)._vtType
				&& _name.equals(((LiteralKey)o)._name));
		}
	}
}

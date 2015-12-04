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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;

/**
 * Rule: CommonSubexpressionElimination. For all statement blocks, 
 * eliminate common subexpressions within dags by merging equivalent
 * operators (same input, equal parameters) bottom-up. For the moment,
 * this only applies within a dag, later this should be extended across
 * statements block (global, inter-procedure). 
 */
public class RewriteCommonSubexpressionElimination extends HopRewriteRule
{
	
	private boolean _mergeLeafs = true;
	
	public RewriteCommonSubexpressionElimination()
	{
		this( true ); //default full CSE
	}
	
	public RewriteCommonSubexpressionElimination( boolean mergeLeafs )
	{
		_mergeLeafs = mergeLeafs;
	}
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) 
		throws HopsException
	{
		if( roots == null )
			return null;
		
		HashMap<String, Hop> dataops = new HashMap<String, Hop>();
		HashMap<String, Hop> literalops = new HashMap<String, Hop>(); //key: <VALUETYPE>_<LITERAL>
		for (Hop h : roots) 
		{
			int cseMerged = 0;
			if( _mergeLeafs ) {
				cseMerged += rule_CommonSubexpressionElimination_MergeLeafs(h, dataops, literalops);
				h.resetVisitStatus();		
			}
			cseMerged += rule_CommonSubexpressionElimination(h);
				
			if( cseMerged > 0 )
				LOG.debug("Common Subexpression Elimination - removed "+cseMerged+" operators.");
		}
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException 
	{
		if( root == null )
			return null;
		
		HashMap<String, Hop> dataops = new HashMap<String, Hop>();
		HashMap<String, Hop> literalops = new HashMap<String, Hop>(); //key: <VALUETYPE>_<LITERAL>
		int cseMerged = 0;
		if( _mergeLeafs ) {
			cseMerged += rule_CommonSubexpressionElimination_MergeLeafs(root, dataops, literalops);
			root.resetVisitStatus();
		}
		cseMerged += rule_CommonSubexpressionElimination(root);
		
		if( cseMerged > 0 )
			LOG.debug("Common Subexpression Elimination - removed "+cseMerged+" operators.");
		
		return root;
	}
	
	/**
	 * 
	 * @param dataops
	 * @param literalops
	 * @return
	 * @throws HopsException
	 */
	private int rule_CommonSubexpressionElimination_MergeLeafs( Hop hop, HashMap<String, Hop> dataops, HashMap<String, Hop> literalops ) 
		throws HopsException 
	{
		int ret = 0;
		if( hop.getVisited() == Hop.VisitStatus.DONE )
			return ret;

		if( hop.getInput().isEmpty() ) //LEAF NODE
		{
			if( hop instanceof LiteralOp )
			{
				String key = hop.getValueType()+"_"+hop.getName();
				if( !literalops.containsKey(key) )
					literalops.put(key, hop);
			}
			else if( hop instanceof DataOp && ((DataOp)hop).isRead())
			{
				if(!dataops.containsKey(hop.getName()) )
					dataops.put(hop.getName(), hop);
			} 
		}
		else //INNER NODE
		{
			//merge leaf nodes (data, literal)
			for( int i=0; i<hop.getInput().size(); i++ )
			{
				Hop hi = hop.getInput().get(i);
				String litKey = hi.getValueType()+"_"+hi.getName();
				if( hi instanceof DataOp && ((DataOp)hi).isRead() && dataops.containsKey(hi.getName()) )
				{
					
					//replace child node ref
					Hop tmp = dataops.get(hi.getName());
					if( tmp != hi ) { //if required
						tmp.getParent().add(hop);
						hop.getInput().set(i, tmp);
						ret++;
					}
				}
				else if( hi instanceof LiteralOp && literalops.containsKey(litKey) )
				{
					Hop tmp = literalops.get(litKey);
					
					//replace child node ref
					if( tmp != hi ){ //if required
						tmp.getParent().add(hop);
						hop.getInput().set(i, tmp);
						ret++;
					}
				}
				
				//recursive invocation (direct return on merged nodes)
				ret += rule_CommonSubexpressionElimination_MergeLeafs(hi, dataops, literalops);		
			}	
		}
		
		hop.setVisited(Hop.VisitStatus.DONE);
		return ret;
	}

	/**
	 * 
	 * @param dataops
	 * @param literalops
	 * @return
	 * @throws HopsException
	 */
	private int rule_CommonSubexpressionElimination( Hop hop ) 
		throws HopsException 
	{
		int ret = 0;
		if( hop.getVisited() == Hop.VisitStatus.DONE )
			return ret;

		//step 1: merge childs recursively first
		for(Hop hi : hop.getInput())
			ret += rule_CommonSubexpressionElimination(hi);	
		
		
		//step 2: merge parent nodes
		if( hop.getParent().size()>1 ) //multiple consumers
		{
			//for all pairs 
			for( int i=0; i<hop.getParent().size()-1; i++ )
				for( int j=i+1; j<hop.getParent().size(); j++ )
				{
					Hop h1 = hop.getParent().get(i);
					Hop h2 = hop.getParent().get(j);
					
					if( h1==h2 )
					{
						//do nothing, note: we should not remove redundant parent links
						//(otherwise rewrites would need to take this property into account) 
						
						//remove redundant h2 from parent list
						//hop.getParent().remove(j);
						//j--;
					}
					else if( h1.compare(h2) ) //merge h2 into h1
					{
						//remove h2 from parent list
						hop.getParent().remove(j);
						
						//replace h2 w/ h1 in h2-parent inputs
						ArrayList<Hop> parent = h2.getParent();
						for( Hop p : parent )
							for( int k=0; k<p.getInput().size(); k++ )
								if( p.getInput().get(k)==h2 )
								{
									p.getInput().set(k, h1);
									h1.getParent().add(p);
								}
						
						//replace h2 w/ h1 in h2-input parents
						for( Hop in : h2.getInput() )
							in.getParent().remove(h2);
						
						ret++;
						j--;
					}
				}
		}
		
		hop.setVisited(Hop.VisitStatus.DONE);

		return ret;
	}

}

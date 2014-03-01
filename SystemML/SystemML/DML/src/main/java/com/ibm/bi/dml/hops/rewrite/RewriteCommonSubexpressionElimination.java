/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;

/**
 * Rule: CommonSubexpressionElimination. For all statement blocks, 
 * eliminate common subexpressions within dags by merging equivalent
 * operators (same input, equal parameters) bottom-up. For the moment,
 * this only applies within a dag, later this should be extended across
 * statements block (global, inter-procedure). 
 */
public class RewriteCommonSubexpressionElimination extends HopRewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots) 
		throws HopsException
	{
		if( roots == null )
			return null;
		
		HashMap<String, Hop> dataops = new HashMap<String, Hop>();
		HashMap<String, Hop> literalops = new HashMap<String, Hop>();
		for (Hop h : roots) 
		{
			int cseMerged = rule_CommonSubexpressionElimination_MergeLeafs(h, dataops, literalops);
			h.resetVisitStatus();
			cseMerged = rule_CommonSubexpressionElimination(h, dataops, literalops);
			LOG.debug("Common Subexpression Elimination - removed "+cseMerged+" operators.");
		}
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root) 
		throws HopsException 
	{
		if( root == null )
			return null;
		
		HashMap<String, Hop> dataops = new HashMap<String, Hop>();
		HashMap<String, Hop> literalops = new HashMap<String, Hop>();
		int cseMerged = rule_CommonSubexpressionElimination_MergeLeafs(root, dataops, literalops);
		root.resetVisitStatus();
		cseMerged = rule_CommonSubexpressionElimination(root, dataops, literalops);
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
		if( hop.get_visited() == Hop.VISIT_STATUS.DONE )
			return ret;

		if( hop.getInput().size()==0 ) //LEAF NODE
		{
			if( hop instanceof LiteralOp )
			{
				if( !literalops.containsKey(hop.get_name()) )
					literalops.put(hop.get_name(), hop);
			}
			else if( hop instanceof DataOp && ((DataOp)hop).isRead())
			{
				if(!dataops.containsKey(hop.get_name()) )
					dataops.put(hop.get_name(), hop);
			} 
		}
		else //INNER NODE
		{
			//merge leaf nodes (data, literal)
			for( int i=0; i<hop.getInput().size(); i++ )
			{
				Hop hi = hop.getInput().get(i);
				if( hi instanceof DataOp && ((DataOp)hi).isRead() && dataops.containsKey(hi.get_name()) )
				{
					//replace child node ref
					Hop tmp = dataops.get(hi.get_name());
					tmp.getParent().add(hop);
					hop.getInput().set(i, tmp);
					ret++; 
				}
				else if( hi instanceof LiteralOp && literalops.containsKey(hi.get_name()) )
				{
					Hop tmp = literalops.get(hi.get_name());
					if(hi.get_valueType()==tmp.get_valueType())
					{	
						//replace child node ref
						tmp.getParent().add(hop);
						hop.getInput().set(i, tmp);
						ret++;
					}
				}
				
				//recursive invocation (direct return on merged nodes)
				ret += rule_CommonSubexpressionElimination_MergeLeafs(hi, dataops, literalops);		
			}	
		}
		
		hop.set_visited(Hop.VISIT_STATUS.DONE);
		return ret;
	}

	/**
	 * 
	 * @param dataops
	 * @param literalops
	 * @return
	 * @throws HopsException
	 */
	private int rule_CommonSubexpressionElimination( Hop hop, HashMap<String, Hop> dataops, HashMap<String, Hop> literalops ) 
		throws HopsException 
	{
		int ret = 0;
		if( hop.get_visited() == Hop.VISIT_STATUS.DONE )
			return ret;

		//step 1: merge childs recursively first
		for(Hop hi : hop.getInput())
			ret += rule_CommonSubexpressionElimination(hi, dataops, literalops);	
		
		
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
						//remove redundant h2 from parent list
						hop.getParent().remove(j);
						j--;
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
		
		hop.set_visited(Hop.VISIT_STATUS.DONE);

		return ret;
	}

}

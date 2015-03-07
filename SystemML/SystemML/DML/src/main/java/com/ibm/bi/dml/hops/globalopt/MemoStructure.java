/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode;

/**
 * This MemoStructure is the central location for storing enumerated plans (configurations) and serves
 * two purposes:
 * 
 * 1) Plan Memoization: Due to the DAG structure (where a single node is reachable over alternative 
 * paths), our top-down, recursive optimization procedure might visit an operator multiple times. 
 * The memo structure memoizes and reuses already generated plans.
 * 
 * 2) config set cache
 * 
 * The internal structure is as follows:
 * Memo Structure := | LONG GDFNodeID | MemoEntry | *
 * Memo Entry := | InterestingPropertySet | DOUBLE COST | * 
 * 
 */
public class MemoStructure 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private HashMap<Long, PlanSet> _entries = null; //NODEID | PLANSET
	private HashMap<Long, Long>    _nodeIDs = null; //NODEID | HOPID
	
	public MemoStructure()
	{
		_entries = new HashMap<Long, PlanSet>();
		_nodeIDs = new HashMap<Long, Long>();
	}
	
	///////////////////////////////////////////////////
	// basic access to memo structure entries
	
	public boolean constainsEntry( GDFNode node )
	{
		return _entries.containsKey( node.getID() );
	}
	
	public void putEntry( GDFNode node, PlanSet entry )
	{
		_entries.put( node.getID(), entry );
		if( node.getHop()!=null )
			_nodeIDs.put( node.getID(), node.getHop().getHopID() );
		else
			_nodeIDs.put( node.getID(), -1L );	
	}
	
	public PlanSet getEntry( GDFNode node )
	{
		return _entries.get( node.getID()) ;
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("------------------------------------\n");
		sb.append("MEMO STRUCTURE (gdfnodeid, plans):  \n");
		sb.append("------------------------------------\n");
		
		for( Entry<Long, PlanSet> e : _entries.entrySet() )
		{
			sb.append("------------------------------------\n");
			sb.append("Node "+e.getKey()+" (hop "+_nodeIDs.get(e.getKey())+"):\n");
			for( Plan p : e.getValue().getPlans() ) {
				sb.append(p.toString());
				sb.append("\n");
			}
		}
		
		return sb.toString();
	}
}

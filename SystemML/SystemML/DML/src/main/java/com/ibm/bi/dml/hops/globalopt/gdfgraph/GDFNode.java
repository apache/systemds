/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfgraph;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

/**
 * The reason of a custom graph structure is to unify both within DAG
 * and cross DAG enumeration. Conceptually, we would only need interesting
 * properties of transient reads and could compile locally. 
 * 
 * Furthermore, having a global graph structure also allows for more advanced
 * algebraic simplification rewrites because the semantics of transient read
 * inputs are always available.
 * 
 */
public class GDFNode 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//references to original program and hop dag
	protected Hop _hop = null;
	protected ProgramBlock _pb = null;
	
	//input nodes
	protected ArrayList<GDFNode> _inputs = null;
	
	public GDFNode()
	{
		
	}
	
	public GDFNode( Hop hop, ProgramBlock pb, ArrayList<GDFNode> inputs )
	{
		_hop = hop;
		_pb = pb;
		_inputs = inputs;
	}
	
	public Hop getHop()
	{
		return _hop;
	}

	public String explain(int level) 
	{
		StringBuilder sb = new StringBuilder();
		
		//create level indentation
		for( int i=0; i<level*2; i++ )
			sb.append("-");
		
		//current node details
		if( _hop!=null )
			sb.append(" Node ["+_hop.getHopID()+", "+_hop.getOpString()+"]\n");
		else
			sb.append(" Node [null]\n");
		
		//recursively explain childs
		for( GDFNode c : _inputs ) {
			sb.append(c.explain(level+1));
		}
		
		return sb.toString();
	}
}

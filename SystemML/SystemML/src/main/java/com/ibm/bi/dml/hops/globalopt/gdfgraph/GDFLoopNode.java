/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfgraph;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

public class GDFLoopNode extends GDFNode
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private GDFNode _predicate = null; 
	private HashMap<String,GDFNode> _linputs = null;
	private HashMap<String,GDFNode> _loutputs = null;
	
	public GDFLoopNode( ProgramBlock pb, GDFNode predicate, HashMap<String, GDFNode> inputs, HashMap<String,GDFNode> outputs )
	{
		super(null, pb, new ArrayList<GDFNode>(inputs.values()));
		_type = NodeType.LOOP_NODE;
		_predicate = predicate;
		_linputs = inputs;
		_loutputs = outputs;
	}
	
	public String explain(int level) 
	{
		StringBuilder sb = new StringBuilder();
		
		//create level indentation
		for( int i=0; i<level*2; i++ )
			sb.append("-");
		
		//current node details
		sb.append(" LoopNode ["+_linputs.keySet().toString()+","+_loutputs.keySet().toString()+"]\n");
		
		//recursively explain childs
		if( _predicate != null )
			sb.append(_predicate.explain(level+1));	
		for( GDFNode c : _linputs.values() ) {
			sb.append(c.explain(level+1));
		}
		
		
		return sb.toString();
	}
}

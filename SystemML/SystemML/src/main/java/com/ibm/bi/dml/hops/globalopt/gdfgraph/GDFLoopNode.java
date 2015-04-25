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
	private HashMap<String,GDFNode> _linputs = null; //all read variables
	private HashMap<String,GDFNode> _loutputs = null; //all updated variables, not necessarily liveout
	
	public GDFLoopNode( ProgramBlock pb, GDFNode predicate, HashMap<String, GDFNode> inputs, HashMap<String,GDFNode> outputs )
	{
		super(null, pb, new ArrayList<GDFNode>(inputs.values()));
		_type = NodeType.LOOP_NODE;
		_predicate = predicate;
		_linputs = inputs;
		_loutputs = outputs;
	}
	
	public HashMap<String, GDFNode> getLoopInputs()
	{
		return _linputs;
	}
	
	public HashMap<String, GDFNode> getLoopOutputs()
	{
		return _loutputs;
	}
	
	public GDFNode getLoopPredicate()
	{
		return _predicate;
	}
	
	@Override
	public String explain(String deps) 
	{
		String ldeps = (deps!=null) ? deps : "";
	
		//current node details
		return "LoopNode "+ldeps+" ["+_linputs.keySet().toString()+","+_loutputs.keySet().toString()+"]";
	}
}

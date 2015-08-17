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
 * Crossblock operators represent 
 * 
 */
public class GDFCrossBlockNode extends GDFNode
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	public enum CrossBlockNodeType {
		PLAIN,
		MERGE,
	}
	
	private CrossBlockNodeType _cbtype = null;
	private String _name = null;
	
	/**
	 * Constructor PLAIN crossblocknode
	 * 
	 * @param hop
	 * @param pb
	 * @param input
	 * @param name
	 */
	public GDFCrossBlockNode( Hop hop, ProgramBlock pb, GDFNode input, String name )
	{
		super(hop, pb, null);
		_type = NodeType.CROSS_BLOCK_NODE;
		_inputs = new ArrayList<GDFNode>();
		_inputs.add( input );
		
		_cbtype = CrossBlockNodeType.PLAIN;
		_name = name;
	}
	
	/**
	 * Constructor MERGE crossblocknode
	 * 
	 * @param hop
	 * @param pb
	 * @param input1
	 * @param input2
	 * @param name
	 */
	public GDFCrossBlockNode( Hop hop, ProgramBlock pb, GDFNode input1, GDFNode input2, String name )
	{
		super(hop, pb, null);
		_type = NodeType.CROSS_BLOCK_NODE;
		_inputs = new ArrayList<GDFNode>();
		_inputs.add( input1 );
		_inputs.add( input2 );
		
		_cbtype = CrossBlockNodeType.MERGE;
		_name = name;
	}

	public String getName()
	{
		return _name;
	}
	
	@Override
	public String explain(String deps) 
	{
		String ldeps = (deps!=null) ? deps : "";
		
		return "CBNode "+ldeps+" ["+_name+", "+_cbtype.toString().toLowerCase()+"]";
	}
}

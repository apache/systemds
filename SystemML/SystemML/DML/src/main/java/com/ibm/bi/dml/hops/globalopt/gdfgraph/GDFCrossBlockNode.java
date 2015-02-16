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
	
	private CrossBlockNodeType _type = null;
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
		_inputs = new ArrayList<GDFNode>();
		_inputs.add( input );
		
		_type = CrossBlockNodeType.PLAIN;
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
		_inputs = new ArrayList<GDFNode>();
		_inputs.add( input1 );
		_inputs.add( input2 );
		
		_type = CrossBlockNodeType.MERGE;
		_name = name;
	}
	
	public String explain(int level) 
	{
		StringBuilder sb = new StringBuilder();
		
		//create level indentation
		for( int i=0; i<level*2; i++ )
			sb.append("-");
		
		//current node details
		sb.append(" CBNode ["+_name+", "+_type.toString().toLowerCase()+"]\n");
		
		//recursively explain childs
		for( GDFNode c : _inputs ) {
			sb.append(c.explain(level+1));
		}
		
		return sb.toString();
	}
}

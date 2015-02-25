/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfgraph;

import java.util.ArrayList;

import com.ibm.bi.dml.runtime.controlprogram.Program;

public class GDFGraph 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	private ArrayList<GDFNode> _roots = null;
	private Program _rtprog = null;
	
	
	public GDFGraph( Program prog, ArrayList<GDFNode> roots )
	{
		_rtprog = prog;
		_roots = roots;
	}
	
	public ArrayList<GDFNode> getGraphRootNodes()
	{
		return _roots;
	}
	
	public Program getRuntimeProgram()
	{
		return _rtprog;
	}
}

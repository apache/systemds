/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

public class ProgramRewriteStatus 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean _rmHopsCSE  = false; //removed hops during common subexpression elimination
	private boolean _rmHopsCF   = false; //removed hops during constant folding
	private boolean _rmBranches = false; //removed branches
	
	public ProgramRewriteStatus()
	{
		_rmHopsCSE = false;
		_rmHopsCF = false;
		_rmBranches = false;
	}
	
	public void setRemovedBranches(){
		_rmBranches = true;
	}
	
	public boolean getRemovedBranches(){
		return _rmBranches;
	}
}

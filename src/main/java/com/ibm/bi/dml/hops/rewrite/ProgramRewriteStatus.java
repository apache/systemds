/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

public class ProgramRewriteStatus 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//status of applied rewrites
	private boolean _rmBranches = false; //removed branches
	private int _blkSize = -1;
	private boolean _injectCheckpoints = false;
	
	//current context
	private boolean _inParforCtx = false;
	
	public ProgramRewriteStatus()
	{
		_rmBranches = false;
		_inParforCtx = false;
		_injectCheckpoints = false;
	}
	
	public void setRemovedBranches(){
		_rmBranches = true;
	}
	
	public boolean getRemovedBranches(){
		return _rmBranches;
	}
	
	public void setInParforContext(boolean flag){
		_inParforCtx = flag;
	}
	
	public boolean isInParforContext(){
		return _inParforCtx;
	}
	
	public void setBlocksize( int blkSize ){
		_blkSize = blkSize;
	}
	
	public int getBlocksize() {
		return _blkSize;
	}
	
	public void setInjectedCheckpoints(){
		_injectCheckpoints = true;
	}
	
	public boolean getInjectedCheckpoints(){
		return _injectCheckpoints;
	}
}

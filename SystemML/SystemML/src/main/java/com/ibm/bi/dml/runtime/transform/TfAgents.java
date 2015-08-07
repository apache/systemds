/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.transform;


public class TfAgents {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	OmitAgent _oa = null;
	MVImputeAgent _mia = null;
	RecodeAgent _ra = null;	
	BinAgent _ba = null;
	DummycodeAgent _da = null;
	
	public TfAgents(OmitAgent oa, MVImputeAgent mia, RecodeAgent ra, BinAgent ba, DummycodeAgent da)  {
		_oa = oa;
		_mia = mia;
		_ra = ra;
		_ba = ba;
		_da = da;
	}
	
	public OmitAgent 	  getOmitAgent() 	{ 	return _oa; }
	public MVImputeAgent  getMVImputeAgent(){ 	return _mia;}
	public RecodeAgent 	  getRecodeAgent() 	{ 	return _ra; }
	public BinAgent 	  getBinAgent() 	{ 	return _ba; }
	public DummycodeAgent getDummycodeAgent() { return _da; }
	
}

/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfresolve;

import com.ibm.bi.dml.hops.globalopt.RewriteConfig;

public class GDFMismatchHeuristicFirst extends GDFMismatchHeuristic
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public String getName(){
		return "FIRST";
	}
	
	@Override
	public boolean resolveMismatch( RewriteConfig currRc, RewriteConfig newRc ) 
	{
		//always return the current rewrite configuration (first come first served)
		return false;
	}
}

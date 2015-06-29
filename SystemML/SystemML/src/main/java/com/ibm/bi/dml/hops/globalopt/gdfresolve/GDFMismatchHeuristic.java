/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfresolve;

import com.ibm.bi.dml.hops.globalopt.RewriteConfig;

public abstract class GDFMismatchHeuristic 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public enum MismatchHeuristicType {
		FIRST,
		BLOCKSIZE_OR_FIRST,
	}
	
	/**
	 * Returns the name of the implementing mismatch heuristic.
	 * 
	 * @return
	 */
	public abstract String getName();
	
	/**
	 * Resolve the mismatch of two given rewrite configurations. This call returns true,
	 * if and only if the new configuration is chosen.
	 * 
	 * @param currRc
	 * @param newRc
	 * @return
	 */
	public abstract boolean resolveMismatch( RewriteConfig currRc, RewriteConfig newRc );
}

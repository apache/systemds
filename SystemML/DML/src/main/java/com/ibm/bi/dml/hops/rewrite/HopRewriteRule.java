/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;

/**
 * Base class for all hop rewrites in order to enable generic
 * application of all rules.
 * 
 */
public abstract class HopRewriteRule 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	protected static final Log LOG = LogFactory.getLog(HopRewriteRule.class.getName());
		
	/**
	 * Handle a generic (last-level) hop DAG with multiple roots.
	 * 
	 * @param root
	 * @throws HopsException 
	 */
	public abstract ArrayList<Hop> rewriteHopDAGs( ArrayList<Hop> roots ) 
		throws HopsException;
	
	/**
	 * Handle a predicate hop DAG with exactly one root.
	 * 
	 * @param root
	 * @throws HopsException 
	 */
	public abstract Hop rewriteHopDAG( Hop root ) 
		throws HopsException;
}

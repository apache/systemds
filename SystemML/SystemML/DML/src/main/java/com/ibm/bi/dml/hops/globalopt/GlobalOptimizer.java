/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFGraph;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.runtime.DMLRuntimeException;

/**
 * Super class for all optimizers (e.g., transformation-based, and enumeration-based)
 * 
 */
public abstract class GlobalOptimizer 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * Core optimizer call, to be implemented by an instance of a global
	 * data flow optimizer.
	 * 
	 * @param prog
	 * @param rtprog
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 */
	public abstract GDFGraph optimize( GDFGraph gdfgraph )
		throws DMLRuntimeException, HopsException, LopsException;
}

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

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.parser.StatementBlock;

/**
 * Base class for all hop rewrites in order to enable generic
 * application of all rules.
 * 
 */
public abstract class StatementBlockRewriteRule 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	protected static final Log LOG = LogFactory.getLog(StatementBlockRewriteRule.class.getName());
		
	/**
	 * Handle an arbitrary statement block. Specific type constraints have to be ensured
	 * within the individual rewrites.
	 * 
	 * @param sb
	 * @param sate
	 * @return
	 * @throws HopsException
	 */
	public abstract ArrayList<StatementBlock> rewriteStatementBlock( StatementBlock sb, ProgramRewriteStatus sate ) 
		throws HopsException;
	
}

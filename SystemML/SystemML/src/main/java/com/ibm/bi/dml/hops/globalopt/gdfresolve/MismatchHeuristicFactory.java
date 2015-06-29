/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.gdfresolve;

import com.ibm.bi.dml.hops.globalopt.gdfresolve.GDFMismatchHeuristic.MismatchHeuristicType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;

public abstract class MismatchHeuristicFactory 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	/**
	 * 
	 * @param type
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static GDFMismatchHeuristic createMismatchHeuristic( MismatchHeuristicType type ) 
		throws DMLRuntimeException
	{
		switch( type ) {
			case FIRST: 
				return new GDFMismatchHeuristicFirst(); 
			case BLOCKSIZE_OR_FIRST: 
				return new GDFMismatchHeuristicBlocksizeOrFirst();
				
			default:
				throw new DMLRuntimeException("Unsupported mismatch heuristic: "+type);
		}	
	}
}

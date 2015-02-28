/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.context;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.sql.sqlcontrolprogram.NetezzaConnector;

public class ExecutionContextFactory 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * 
	 * @return
	 */
	public static ExecutionContext createContext()
	{
		return createContext( null );
	}
	
	public static ExecutionContext createContext( Program prog )
	{
		return createContext(true, prog);
	}
	
	/**
	 * 
	 * @param platform
	 * @return
	 */
	public static ExecutionContext createContext( boolean allocateVars, Program prog )
	{
		ExecutionContext ec = null;
		
		switch( DMLScript.rtplatform )
		{
			case SINGLE_NODE:
			case HADOOP:
			case HYBRID:
				ec = new ExecutionContext(allocateVars, prog);
				break;
				
			case SPARK:
				ec = new SparkExecutionContext(allocateVars, prog);
				break;
				
			case NZ:
				ec = new SQLExecutionContext(allocateVars, prog);
				break;
		}
		
		return ec;
	}
	
	/**
	 * 
	 * @param conn
	 * @return
	 */
	public static SQLExecutionContext createSQLContext( NetezzaConnector conn )
	{
		return new SQLExecutionContext(conn);
	}
}

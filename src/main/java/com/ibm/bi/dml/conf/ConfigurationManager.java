/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.conf;

import org.apache.hadoop.mapred.JobConf;



/**
 * Singleton for accessing the parsed and merged system configuration.
 * 
 * NOTE: parallel execution of multiple DML scripts (in the same JVM) with different configurations  
 *       would require changes/extensions of this class. 
 */
public class ConfigurationManager 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static DMLConfig _conf = null; //read systemml configuration
	private static JobConf _rJob = null; //cached job conf for read-only operations	
	
	static{
		_rJob = new JobConf();
	}
	
	
	/**
	 * 
	 * @param conf
	 */
	public synchronized static void setConfig( DMLConfig conf )
	{
		_conf = conf;
	}
	
	/**
	 * 
	 * @return
	 */
	public synchronized static DMLConfig getConfig()
	{
		return _conf;
	}
	
    /**
     * Returns a cached JobConf object, intended for global use by all operations 
     * with read-only access to job conf. This prevents to read the hadoop conf files
     * over and over again from classpath. However, 
     * 
     * @return
     */
	public static JobConf getCachedJobConf()
	{
		return _rJob;
	}
}

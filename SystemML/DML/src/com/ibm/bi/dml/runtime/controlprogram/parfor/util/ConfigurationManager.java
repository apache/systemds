package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

import com.ibm.bi.dml.utils.configuration.DMLConfig;

/**
 * Singleton for accessing the parsed and merged system configuration.
 * 
 * NOTE: parallel execution of multiple DML scripts (in the same JVM) with different configurations  
 *       would require changes/extensions of this class. 
 */
public class ConfigurationManager 
{
	private static DMLConfig _conf = null;
	
	/**
	 * 
	 * @param conf
	 */
	public static void setConfig( DMLConfig conf )
	{
		_conf = conf;
	}
	
	/**
	 * 
	 * @return
	 */
	public static DMLConfig getConfig()
	{
		return _conf;
	}
}

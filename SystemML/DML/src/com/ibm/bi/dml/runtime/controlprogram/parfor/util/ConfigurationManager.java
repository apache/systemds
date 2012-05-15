package com.ibm.bi.dml.runtime.controlprogram.parfor.util;

import com.ibm.bi.dml.utils.configuration.DMLConfig;

/**
 * Singleton for accessing the parsed and merged system configuration.
 * 
 * TODO: would require changes if multiple DML scripts with different configurations run in the same JVM
 * 
 * 
 * @author mboehm
 *
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

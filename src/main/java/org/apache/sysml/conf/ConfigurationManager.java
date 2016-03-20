/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.conf;

import org.apache.hadoop.mapred.JobConf;
import org.apache.sysml.conf.CompilerConfig.ConfigType;



/**
 * Singleton for accessing the parsed and merged system configuration.
 * 
 * NOTE: parallel execution of multiple DML scripts (in the same JVM) with different configurations  
 *       would require changes/extensions of this class. 
 */
public class ConfigurationManager 
{
	/** Global cached job conf for read-only operations	*/
	private static JobConf _rJob = null; 
	
	/** Global DML configuration (read or defaults) */
	private static DMLConfig _dmlconf = null; 
	
	/** Local DML configuration for thread-local config updates */
	private static ThreadLocalDMLConfig _ldmlconf = new ThreadLocalDMLConfig();
	
    /** Global compiler configuration (defaults) */
    private static CompilerConfig _cconf = null;
	
    /** Local compiler configuration for thead-local config updates */
    private static ThreadLocalCompilerConfig _lcconf = new ThreadLocalCompilerConfig();
    
    //global static initialization
	static {
		_rJob = new JobConf();
		
		//initialization after job conf in order to prevent cyclic initialization issues 
		//ConfigManager -> OptimizerUtils -> InfrastructureAnalyer -> ConfigManager 
 		_dmlconf = new DMLConfig();
		_cconf = new CompilerConfig();
	}
	
	
    /**
     * Returns a cached JobConf object, intended for global use by all operations 
     * with read-only access to job conf. This prevents to read the hadoop conf files
     * over and over again from classpath. However, 
     * 
     * @return
     */
	public static JobConf getCachedJobConf() {
		return _rJob;
	}
	
	/**
	 * 
	 * @param job
	 */
	public static void setCachedJobConf(JobConf job) {
		_rJob = job;
	}
	
	/**
	 * Sets a global configuration as a basis for any thread-local configurations.
	 * NOTE: This global configuration should never be accessed directly but only
	 * through its thread-local derivatives. 
	 * 
	 * @param conf
	 */
	public synchronized static void setGlobalConfig( DMLConfig conf ) {
		_dmlconf = conf;
		
		//reinitialize thread-local dml configs w/ _dmlconf
		_ldmlconf = new ThreadLocalDMLConfig();
	}
	
	/**
	 * Sets the current thread-local dml configuration to the given config.
	 * 
	 * @param conf
	 */
	public static void setLocalConfig( DMLConfig conf ) {
		_ldmlconf.set(conf);
	}
	
	/**
	 * Gets the current thread-local dml configuration.
	 * 
	 * @return
	 */
	public static DMLConfig getDMLConfig() {
		return _ldmlconf.get();
	}
	
	/**
	 * 
	 * @param conf
	 */
	public synchronized static void setGlobalConfig( CompilerConfig conf ) {
		_cconf = conf;
		
		//reinitialize thread-local compiler configs w/ _cconf
		_lcconf = new ThreadLocalCompilerConfig();
	}
	
	/**
	 * Sets the current thread-local compiler configuration to the given config.
	 * 
	 * @param conf
	 */
	public static void setLocalConfig( CompilerConfig conf ) {
		_lcconf.set(conf);
	}
	
	/**
	 * Removes the thread-local dml and compiler configurations, leading to
	 * a reinitialization on the next get unless set in between.
	 */
	public static void clearLocalConfigs() {
		_ldmlconf.remove();
		_lcconf.remove();
	}
	
	/**
	 * Gets the current thread-local compiler configuration.
	 * 
	 * @return
	 */
	public static CompilerConfig getCompilerConfig() {
		return _lcconf.get();
	}
	
	/**
	 * Get a boolean compiler config in a robust manner,
	 * returning false if config not existing.
	 * 
	 * @param key
	 * @return
	 */
	public static boolean getCompilerConfigFlag(ConfigType key) {
		CompilerConfig cconf = getCompilerConfig();
		return (cconf!=null) ? cconf.getBool(key) : false;
	}
	
	/////////////////////////////////////
	// shorthand methods for common local configurations
	
	public static String getScratchSpace() {
		return getDMLConfig().getTextValue(DMLConfig.SCRATCH_SPACE);
	}
	
	public static int getBlocksize() {
		return getCompilerConfig().getInt(ConfigType.BLOCK_SIZE);
	}
	
	public static int getNumReducers() {
		return getDMLConfig().getIntValue(DMLConfig.NUM_REDUCERS);
	}
	
	public static boolean isDynamicRecompilation() {
		return getCompilerConfigFlag(ConfigType.ALLOW_DYN_RECOMPILATION);
	}
	
	public static boolean isParallelMatrixOperations() {
		return getCompilerConfigFlag(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS);
	}
	
	public static boolean isParallelParFor() {
		return getCompilerConfigFlag(ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR);
	}
	
	
	///////////////////////////////////////
	// Thread-local classes
	
	/**
	 * 
	 */
	private static class ThreadLocalDMLConfig extends ThreadLocal<DMLConfig> {
		@Override 
        protected DMLConfig initialValue() { 
			//currently initialize by reference to avoid unnecessary deep copy via clone.
	        if( _dmlconf != null )
	        	return _dmlconf; 
	        return null;
        }
    }
	
	/**
	 * 
	 */
	private static class ThreadLocalCompilerConfig extends ThreadLocal<CompilerConfig> {
		@Override 
		protected CompilerConfig initialValue() { 
			if( _cconf != null )
				return _cconf.clone();
			return null;
		}
    };
}

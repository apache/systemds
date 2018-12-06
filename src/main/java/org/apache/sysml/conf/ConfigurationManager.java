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
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.runtime.matrix.mapred.MRConfigurationNames;
import org.apache.sysml.runtime.matrix.mapred.MRJobConfiguration;
import org.apache.sysml.utils.lite.LiteCheck;



/**
 * Singleton for accessing the parsed and merged system configuration.
 * 
 * NOTE: parallel execution of multiple DML scripts (in the same JVM) with different configurations  
 *       would require changes/extensions of this class. 
 */
public class ConfigurationManager 
{
	/** Global cached job conf for read-only operations	*/
	private static JobConf _rJob;
	
	/** Global DML configuration (read or defaults) */
	private static DMLConfig _dmlconf;
	
	/** Local DML configuration for thread-local config updates */
	private static ThreadLocalDMLConfig _ldmlconf = new ThreadLocalDMLConfig();
	
	/** Global DML options (read or defaults) */
	private static DMLOptions _dmlOptions = DMLOptions.defaultOptions; 
	
	/** Local DML configuration for thread-local options */
	private static ThreadLocalDMLOptions _ldmlOptions = new ThreadLocalDMLOptions();
	
    /** Global compiler configuration (defaults) */
    private static CompilerConfig _cconf;
	
    /** Local compiler configuration for thead-local config updates */
    private static ThreadLocalCompilerConfig _lcconf = new ThreadLocalCompilerConfig();
    
    //global static initialization
	static {
		_rJob = new JobConf();
		
		//initialization after job conf in order to prevent cyclic initialization issues 
		//ConfigManager -> OptimizerUtils -> InfrastructureAnalyer -> ConfigManager 
 		_dmlconf = new DMLConfig();
		_cconf = new CompilerConfig();

		if (LiteCheck.isLite() && MRJobConfiguration.USE_BINARYBLOCK_SERIALIZATION) {
			// to be able to write using binary format
			// WritableSerialization -> MatrixIndexes
			// BinaryBlockSerialization -> MatrixBlock
			_rJob.set(MRConfigurationNames.IO_SERIALIZATIONS,
					"org.apache.hadoop.io.serializer.WritableSerialization,org.apache.sysml.runtime.io.BinaryBlockSerialization");
		}
	}
	
	
    /**
     * Returns a cached JobConf object, intended for global use by all operations 
     * with read-only access to job conf. This prevents to read the hadoop conf files
     * over and over again from classpath. However, 
     * 
     * @return the cached JobConf
     */
	public static JobConf getCachedJobConf() {
		return _rJob;
	}
	
	public static void setCachedJobConf(JobConf job) {
		_rJob = job;
	}
	
	/**
	 * Sets a global configuration as a basis for any thread-local configurations.
	 * NOTE: This global configuration should never be accessed directly but only
	 * through its thread-local derivatives. 
	 * 
	 * @param conf the configuration
	 */
	public synchronized static void setGlobalConfig( DMLConfig conf ) {
		_dmlconf = conf;
		
		//reinitialize thread-local dml configs w/ _dmlconf
		_ldmlconf = new ThreadLocalDMLConfig();
		
		FINEGRAINED_STATISTICS = conf.getBooleanValue(DMLConfig.EXTRA_FINEGRAINED_STATS);
	}
	
	/**
	 * Sets a global options as a basis for any thread-local configurations.
	 * NOTE: This global options should never be accessed directly but only
	 * through its thread-local derivatives. 
	 * 
	 * @param opts the dml options
	 */
	public synchronized static void setGlobalOptions( DMLOptions opts ) {
		_dmlOptions = opts;
		
		//reinitialize thread-local dml options w/ _dmlOptions
		_ldmlOptions = new ThreadLocalDMLOptions();
		
		STATISTICS = opts.stats;
	}
	
	/**
	 * Sets the current thread-local dml configuration to the given config.
	 * 
	 * @param conf the configuration
	 */
	public static void setLocalConfig( DMLConfig conf ) {
		_ldmlconf.set(conf);
		
		FINEGRAINED_STATISTICS = conf.getBooleanValue(DMLConfig.EXTRA_FINEGRAINED_STATS);
	}
	
	/**
	 * Gets the current thread-local dml configuration.
	 * 
	 * @return the dml configuration
	 */
	public static DMLConfig getDMLConfig() {
		return _ldmlconf.get();
	}
	
	/**
	 * Gets the current thread-local dml options.
	 * 
	 * @return the dml options
	 */
	public static DMLOptions getDMLOptions() {
		return _ldmlOptions.get();
	}
	
	/**
	 * Sets the current thread-local dml configuration to the given options.
	 * 
	 * @param opts the configuration
	 */
	public static void setLocalOptions( DMLOptions opts ) {
		_dmlOptions = opts;
		_ldmlOptions.set(opts);
		STATISTICS = opts.stats;
	}
	
	
	public synchronized static void setGlobalConfig( CompilerConfig conf ) {
		_cconf = conf;
		
		//reinitialize thread-local compiler configs w/ _cconf
		_lcconf = new ThreadLocalCompilerConfig();
	}
	
	/**
	 * Sets the current thread-local compiler configuration to the given config.
	 * 
	 * @param conf the compiler configuration
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
		_ldmlOptions.remove();
	}
	
	/**
	 * Gets the current thread-local compiler configuration.
	 * 
	 * @return the compiler configuration
	 */
	public static CompilerConfig getCompilerConfig() {
		return _lcconf.get();
	}
	
	/**
	 * Get a boolean compiler config in a robust manner,
	 * returning false if config not existing.
	 * 
	 * @param key config type
	 * @return compiler config flag
	 */
	public static boolean getCompilerConfigFlag(ConfigType key) {
		CompilerConfig cconf = getCompilerConfig();
		return (cconf!=null) && cconf.getBool(key);
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
	
	public static boolean isCodegenEnabled() {
		return (getDMLConfig().getBooleanValue(DMLConfig.CODEGEN)
			|| getCompilerConfigFlag(ConfigType.CODEGEN_ENABLED))
			&& !ConfigurationManager.isGPU();
		//note: until codegen is supported for the GPU backend, we globally
		//disable codegen if operations are forced to the GPU to avoid
		//a counter-productive impact on performance.
	}
	
	/**
	 * @return true if gpu is enabled
	 */
	public static boolean isGPU() {
		return _ldmlOptions.get().isGPU();
	}
	
	/**
	 * @return true if GPU is enabled in forced mode
	 */
	public static boolean isForcedGPU() {
		return _ldmlOptions.get().isForceGPU();
	}
	
	/**
	 * @return the execution Mode
	 */
	public static RUNTIME_PLATFORM getExecutionMode() {
		return _ldmlOptions.get().getExecutionMode();
	}
	
	// -------------------------------------------------------------------------------
	// This needs to be revisited in context of multi-threaded execution: JMLC.
	// Since STATISTICS and FINEGRAINED_STATISTICS are frequently used flags,
	// we use static variables here instead of _dmlOptions.stats and 
	// _dmlconf.getBooleanValue(DMLConfig.EXTRA_FINEGRAINED_STATS);
	private static boolean STATISTICS = false;
	private static boolean FINEGRAINED_STATISTICS = false;
	private static boolean JMLC_MEM_STATISTICS = false;
	
	/**
	 * @return true if statistics is enabled
	 */
	public static boolean isStatistics() {
		return STATISTICS;
	}
	
	/**
	 * @return true if finegrained statistics is enabled
	 */
	public static boolean isFinegrainedStatistics() {
		return FINEGRAINED_STATISTICS;
	}

	/**
	 * @return true if JMLC memory statistics are enabled
	 */
	public static boolean isJMLCMemStatistics() { return JMLC_MEM_STATISTICS; }

	/**
	 * Whether or not statistics about the DML/PYDML program should be output to
	 * standard output.
	 *
	 * @param enabled
	 *            {@code true} if statistics should be output, {@code false}
	 *            otherwise
	 */
	public static void setStatistics(boolean enabled) {
		STATISTICS = enabled;
	}

	/**
	 * Whether or not detailed statistics about program memory use should be output
	 * to standard output when running under JMLC
	 *
	 * @param enabled
	 *            {@code true} if statistics should be output, {@code false}
	 *            otherwise
	 */
	public static void setJMLCMemStats(boolean enabled) {
		JMLC_MEM_STATISTICS = enabled;
	}


	/**
	 * Whether or not finegrained statistics should be enabled
	 *
	 * @param enabled
	 *            {@code true} if statistics should be output, {@code false}
	 *            otherwise
	 */
	public static void setFinegrainedStatistics(boolean enabled) {
		FINEGRAINED_STATISTICS = enabled;
	}

	/**
	 * Reset the statistics flag.
	 */
	public static void resetStatistics() {
		STATISTICS = false;
	}
	
	
	// -------------------------------------------------------------------------------
	
	///////////////////////////////////////
	// Thread-local classes
	
	private static class ThreadLocalDMLOptions extends ThreadLocal<DMLOptions> {
		@Override 
		protected DMLOptions initialValue() {  
			if(_dmlOptions != null)
				return _dmlOptions;
			return DMLOptions.defaultOptions;
		}
	}
	
	private static class ThreadLocalDMLConfig extends ThreadLocal<DMLConfig> {
		@Override 
		protected DMLConfig initialValue() { 
			//currently initialize by reference to avoid unnecessary deep copy via clone.
			if( _dmlconf != null )
				return _dmlconf; 
			return null;
		}
	}
	
	private static class ThreadLocalCompilerConfig extends ThreadLocal<CompilerConfig> {
		@Override 
		protected CompilerConfig initialValue() { 
			if( _cconf != null )
				return _cconf.clone();
			return null;
		}
	}
}

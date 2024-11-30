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

package org.apache.sysds.conf;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Compression.CompressConfig;
import org.apache.sysds.lops.compile.linearization.IDagLinearizerFactory.DagLinearizer;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Singleton for accessing the parsed and merged system configuration.
 * 
 * NOTE: parallel execution of multiple DML scripts (in the same JVM) with different configurations  
 *       would require changes/extensions of this class. 
 */
public class ConfigurationManager{
	private static final Log LOG = LogFactory.getLog(ConfigurationManager.class.getName());

	/** Global cached job conf for read-only operations */
	private static JobConf _rJob = null; 

	/** Global DML configuration (read or defaults) */
	private static DMLConfig _dmlconf = null; 

	/** Local DML configuration for thread-local config updates */
	private static ThreadLocalDMLConfig _ldmlconf = new ThreadLocalDMLConfig();

	/** Global compiler configuration (defaults) */
	private static CompilerConfig _cconf = null;

	/** Local compiler configuration for thead-local config updates */
	private static ThreadLocalCompilerConfig _lcconf = new ThreadLocalCompilerConfig();

	/** Indicate if the filesystem is loaded or not */
	private static Future<?> loaded;

	//global static initialization
	static {
		_rJob = new JobConf();
		
		//initialization after job conf in order to prevent cyclic initialization issues 
		//ConfigManager -> OptimizerUtils -> InfrastructureAnalyzer -> ConfigManager 
 		_dmlconf = new DMLConfig();
		_cconf = new CompilerConfig();

		final ExecutorService pool = CommonThreadPool.get();
		try{
			loaded = pool.submit(() ->{
				try{
					IOUtilFunctions.getFileSystem(_rJob);
				}
				catch(Exception e){
					LOG.warn(e.getMessage());
				}
			});
		}
		finally{
			pool.shutdown();
		}
	}
	
	
	/**
	 * Returns a cached JobConf object, intended for global use by all operations
	 * with read-only access to job conf. This prevents to read the hadoop conf files
	 * over and over again from classpath.
	 *
	 * @return the cached JobConf
	 */
	public static JobConf getCachedJobConf() {
		try{
			loaded.get();
			return _rJob;
		}
		catch(Exception e){
			throw new RuntimeException(e);
		}
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
	}
	
	/**
	 * Sets the current thread-local dml configuration to the given config.
	 * 
	 * @param conf the configuration
	 */
	public static void setLocalConfig( DMLConfig conf ) {
		_ldmlconf.set(conf);
	}
	
	/**
	 * Gets the current thread-local dml configuration.
	 * 
	 * @return the dml configuration
	 */
	public static DMLConfig getDMLConfig() {
		return _ldmlconf.get();
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
	
	public static boolean isDynamicRecompilation() {
		return getCompilerConfigFlag(ConfigType.ALLOW_DYN_RECOMPILATION);
	}
	
	public static boolean isParallelMatrixOperations() {
		return getCompilerConfigFlag(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS);
	}
	
	public static boolean isParallelTransform() {
		return getDMLConfig().getBooleanValue(DMLConfig.PARALLEL_ENCODE);
	}

	public static boolean isParallelTokenize() {
		return getDMLConfig().getBooleanValue(DMLConfig.PARALLEL_TOKENIZE);
	}

	public static boolean isStagedParallelTransform() {
		return getDMLConfig().getBooleanValue(DMLConfig.PARALLEL_ENCODE_STAGED);
	}

	public static int getNumberTokenizeBlocks(){
		return getDMLConfig().getIntValue(DMLConfig.PARALLEL_TOKENIZE_NUM_BLOCKS);
	}

	public static int getParallelApplyBlocks(){
		return getDMLConfig().getIntValue(DMLConfig.PARALLEL_ENCODE_APPLY_BLOCKS);
	}

	public static int getParallelBuildBlocks(){
		return getDMLConfig().getIntValue(DMLConfig.PARALLEL_ENCODE_BUILD_BLOCKS);
	}
	
	public static int getNumThreads() {
		return getDMLConfig().getIntValue(DMLConfig.PARALLEL_ENCODE_NUM_THREADS);
	}

	public static boolean isParallelParFor() {
		return getCompilerConfigFlag(ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR);
	}
	
	public static boolean isCodegenEnabled() {
		return (getDMLConfig().getBooleanValue(DMLConfig.CODEGEN)
			|| getCompilerConfigFlag(ConfigType.CODEGEN_ENABLED));
	}
	
	public static boolean isFederatedRuntimePlanner() {
		return getCompilerConfigFlag(ConfigType.FEDERATED_RUNTIME);
	}

	public static boolean isCompressionEnabled(){
		CompressConfig compress = getCompressConfig();
		return compress.isEnabled();
	}

	// flag to enable encode cache
	/*public static boolean isEncodeCacheEnabled(){

		return getDMLConfig().getBooleanValue(DMLConfig.ENCODECACHE_ENABLED);
	}*/

	// flag for encode cache memory limit (fraction of heap size)
	/*public static double getEncodeCacheMemoryFraction(){

		return getDMLConfig().getDoubleValue(DMLConfig.ENCODECACHE_MEMORY_FRACTION);
	}*/

	public static CompressConfig getCompressConfig(){
		return CompressConfig.valueOf(getDMLConfig().getTextValue(DMLConfig.COMPRESSED_LINALG).toUpperCase());
	}
	
	public static int getFederatedTimeout(){
		return getDMLConfig().getIntValue(DMLConfig.FEDERATED_TIMEOUT);
	}

	public static boolean isFederatedSSL(){
		return getDMLConfig().getBooleanValue(DMLConfig.USE_SSL_FEDERATED_COMMUNICATION);
	}
	
	public static boolean isFederatedReadCacheEnabled(){
		return getDMLConfig().getBooleanValue(DMLConfig.FEDERATED_READCACHE);
	}

	public static boolean isPrefetchEnabled() {
		return (getDMLConfig().getBooleanValue(DMLConfig.ASYNC_PREFETCH)
			|| OptimizerUtils.ASYNC_PREFETCH);
	}

	public static boolean isMaxPrallelizeEnabled() {
		return (getLinearizationOrder() == DagLinearizer.MAX_PARALLELIZE
			|| OptimizerUtils.MAX_PARALLELIZE_ORDER);
	}

	public static boolean isAutoLinearizationEnabled() {
		return OptimizerUtils.COST_BASED_ORDERING;
	}

	public static boolean isParallelIOEnabled(){
		return getDMLConfig().getBooleanValue(DMLConfig.CP_PARALLEL_IO);
	}

	public static boolean isBroadcastEnabled() {
		return (getDMLConfig().getBooleanValue(DMLConfig.ASYNC_SPARK_BROADCAST)
			|| OptimizerUtils.ASYNC_BROADCAST_SPARK);
	}
	public static boolean isCheckpointEnabled() {
		return (getDMLConfig().getBooleanValue(DMLConfig.ASYNC_SPARK_CHECKPOINT)
			|| OptimizerUtils.ASYNC_CHECKPOINT_SPARK);
	}

	public static boolean isRuleBasedGPUPlacement() {
		return (DMLScript.USE_ACCELERATOR &&
			(getDMLConfig().getBooleanValue(DMLConfig.GPU_RULE_BASED_PLACEMENT)
			|| OptimizerUtils.RULE_BASED_GPU_EXEC));
	}

	public static boolean isAutoEvictionEnabled() {
		return OptimizerUtils.AUTO_GPU_CACHE_EVICTION;
	}

	public static DagLinearizer getLinearizationOrder() {
		if (OptimizerUtils.COST_BASED_ORDERING)
			return DagLinearizer.AUTO;
		else if (OptimizerUtils.MAX_PARALLELIZE_ORDER)
			return DagLinearizer.MAX_PARALLELIZE;
		else
			return DagLinearizer.valueOf(ConfigurationManager.getDMLConfig()
				.getTextValue(DMLConfig.DAG_LINEARIZATION).toUpperCase());
	}

	///////////////////////////////////////
	// Thread-local classes
	
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

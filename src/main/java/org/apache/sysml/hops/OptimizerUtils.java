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

package org.apache.sysml.hops;

import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.CompilerConfig;
import org.apache.sysml.conf.CompilerConfig.ConfigType;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.FileFormatTypes;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.lops.Checkpoint;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.Dag;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.functionobjects.IntegerDivide;
import org.apache.sysml.runtime.functionobjects.Modulus;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;

public class OptimizerUtils 
{
	private static final Log LOG = LogFactory.getLog(OptimizerUtils.class.getName());
	
	////////////////////////////////////////////////////////
	// Optimizer constants and flags (incl tuning knobs)  //
	////////////////////////////////////////////////////////
	/**
	 * Utilization factor used in deciding whether an operation to be scheduled on CP or MR. 
	 * NOTE: it is important that MEM_UTIL_FACTOR+CacheableData.CACHING_BUFFER_SIZE &lt; 1.0
	 */
	public static double MEM_UTIL_FACTOR = 0.7d;
	
	/** Default blocksize if unspecified or for testing purposes */
	public static final int DEFAULT_BLOCKSIZE = 1000;
	
	/** Default frame blocksize */
	public static final int DEFAULT_FRAME_BLOCKSIZE = 1000;
	
	/** Default optimization level if unspecified */
	public static final OptimizationLevel DEFAULT_OPTLEVEL = 
			OptimizationLevel.O2_LOCAL_MEMORY_DEFAULT;
	
	/**
	 * Default memory size, which is used if the actual estimate can not be computed 
	 * e.g., when input/output dimensions are unknown. The default is set to a large 
	 * value so that operations are scheduled on MR while avoiding overflows as well.  
	 */
	public static double DEFAULT_SIZE;	
	
	
	public static final long DOUBLE_SIZE = 8;
	public static final long INT_SIZE = 4;
	public static final long CHAR_SIZE = 1;
	public static final long BOOLEAN_SIZE = 1;
	public static final double INVALID_SIZE = -1d; // memory estimate not computed

	//constants for valid CP matrix dimension sizes / nnz (dense/sparse)
	public static final long MAX_NUMCELLS_CP_DENSE = Integer.MAX_VALUE;
	public static final long MAX_NNZ_CP_SPARSE = (MatrixBlock.DEFAULT_SPARSEBLOCK == 
			SparseBlock.Type.MCSR) ? Long.MAX_VALUE : Integer.MAX_VALUE;

	/**
	 * Enables common subexpression elimination in dags. There is however, a potential tradeoff
	 * between computation redundancy and data transfer between MR jobs. Since, we do not reason
	 * about transferred data yet, this rewrite rule is enabled by default.
	 */
	public static boolean ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = true;

	/**
	 * Enables constant folding in dags. Constant folding computes simple expressions of binary 
	 * operations and literals and replaces the hop sub-DAG with a new literal operator. 
	 */
	public static boolean ALLOW_CONSTANT_FOLDING = true;
	
	public static boolean ALLOW_ALGEBRAIC_SIMPLIFICATION = true;
	public static boolean ALLOW_OPERATOR_FUSION = true;
	
	/**
	 * Enables if-else branch removal for constant predicates (original literals or 
	 * results of constant folding). 
	 * 
	 */
	public static boolean ALLOW_BRANCH_REMOVAL = true;
	
	public static boolean ALLOW_AUTO_VECTORIZATION = true;
	
	/**
	 * Enables simple expression evaluation for datagen parameters 'rows', 'cols'. Simple
	 * expressions are defined as binary operations on literals and nrow/ncol. This applies
	 * only to exact size information.
	 */
	public static boolean ALLOW_SIZE_EXPRESSION_EVALUATION = true;

	/**
	 * Enables simple expression evaluation for datagen parameters 'rows', 'cols'. Simple
	 * expressions are defined as binary operations on literals and b(+) or b(*) on nrow/ncol.
	 * This applies also to worst-case size information. 
	 */
	public static boolean ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = true;

	public static boolean ALLOW_RAND_JOB_RECOMPILE = true;

	/**
	 * Enables parfor runtime piggybacking of MR jobs into the packed jobs for
	 * scan sharing.
	 */
	public static boolean ALLOW_RUNTIME_PIGGYBACKING = true;
	
	/**
	 * Enables interprocedural analysis between main script and functions as well as functions
	 * and other functions. This includes, for example, to propagate statistics into functions
	 * if save to do so (e.g., if called once).
	 */
	public static boolean ALLOW_INTER_PROCEDURAL_ANALYSIS = true;

	/**
	 * Enables an additional "second chance" pass of static rewrites + IPA after the initial pass of
	 * IPA.  Without this, there are many situations in which sizes will remain unknown even after
	 * recompilation, thus leading to distributed ops.  With the second chance enabled, sizes in
	 * these situations can be determined.  For example, the alternation of constant folding
	 * (static rewrite) and scalar replacement (IPA) can allow for size propagation without dynamic
	 * rewrites or recompilation due to replacement of scalars with literals during IPA, which
	 * enables constant folding of sub-DAGs of literals during static rewrites, which in turn allows
	 * for scalar propagation during IPA.
	 */
	public static boolean ALLOW_IPA_SECOND_CHANCE = true;

	/**
	 * Enables sum product rewrites such as mapmultchains. In the future, this will cover 
	 * all sum-product related rewrites.
	 */
	public static boolean ALLOW_SUM_PRODUCT_REWRITES = true;
	
	/**
	 * Enables a specific hop dag rewrite that splits hop dags after csv persistent reads with 
	 * unknown size in order to allow for recompile.
	 */
	public static boolean ALLOW_SPLIT_HOP_DAGS = true;
	
	/**
	 * Enables a specific rewrite that enables update in place for loop variables that are
	 * only read/updated via cp leftindexing.
	 */
	public static boolean ALLOW_LOOP_UPDATE_IN_PLACE = true;
	
	
	/**
	 * Specifies a multiplier computing the degree of parallelism of parallel
	 * text read/write out of the available degree of parallelism. Set it to 1.0
	 * to get a number of threads equal the number of virtual cores.
	 * 
	 */
	public static final double PARALLEL_CP_READ_PARALLELISM_MULTIPLIER = 1.0;
	public static final double PARALLEL_CP_WRITE_PARALLELISM_MULTIPLIER = 1.0;

	/**
	 * Enables the use of CombineSequenceFileInputFormat with splitsize = 2x hdfs blocksize, 
	 * if sort buffer size large enough and parallelism not hurt. This solves to issues: 
	 * (1) it combines small files (depending on producers), and (2) it reduces task
	 * latency of large jobs with many tasks by factor 2.
	 * 
	 */
	public static final boolean ALLOW_COMBINE_FILE_INPUT_FORMAT = true;
	
	
	public static long GPU_MEMORY_BUDGET = -1;
	
	//////////////////////
	// Optimizer levels //
	//////////////////////

	/**
	 * Optimization Types for Compilation
	 * 
	 *  O0 STATIC - Decisions for scheduling operations on CP/MR are based on
	 *  predefined set of rules, which check if the dimensions are below a 
	 *  fixed/static threshold (OLD Method of choosing between CP and MR). 
	 *  The optimization scope is LOCAL, i.e., per statement block.
	 *  Advanced rewrites like constant folding, common subexpression elimination,
	 *  or inter procedural analysis are NOT applied.
	 *  
	 *  O1 MEMORY_BASED - Every operation is scheduled on CP or MR, solely
	 *  based on the amount of memory required to perform that operation. 
	 *  It does NOT take the execution time into account.
	 *  The optimization scope is LOCAL, i.e., per statement block.
	 *  Advanced rewrites like constant folding, common subexpression elimination,
	 *  or inter procedural analysis are NOT applied.
	 *  
	 *  O2 MEMORY_BASED - Every operation is scheduled on CP or MR, solely
	 *  based on the amount of memory required to perform that operation. 
	 *  It does NOT take the execution time into account.
	 *  The optimization scope is LOCAL, i.e., per statement block.
	 *  All advanced rewrites are applied. This is the default optimization
	 *  level of SystemML.
	 *
	 *  O3 GLOBAL TIME_MEMORY_BASED - Operation scheduling on CP or MR as well as
	 *  many other rewrites of data flow properties such as block size, partitioning,
	 *  replication, vectorization, etc are done with the optimization objective of
	 *  minimizing execution time under hard memory constraints per operation and
	 *  execution context. The optimization scope if GLOBAL, i.e., program-wide.
	 *  All advanced rewrites are applied. This optimization level requires more 
	 *  optimization time but has higher optimization potential.
	 *  
	 *  O4 DEBUG MODE - All optimizations, global and local, which interfere with 
	 *  breakpoints are NOT applied. This optimization level is REQUIRED for the 
	 *  compiler running in debug mode.
	 */
	public enum OptimizationLevel { 
		O0_LOCAL_STATIC, 
		O1_LOCAL_MEMORY_MIN,
		O2_LOCAL_MEMORY_DEFAULT,
		O3_LOCAL_RESOURCE_TIME_MEMORY,
		O4_GLOBAL_TIME_MEMORY,
		O5_DEBUG_MODE,
	};
		
	public static OptimizationLevel getOptLevel() {
		int optlevel = ConfigurationManager.getCompilerConfig().getInt(ConfigType.OPT_LEVEL);
		return OptimizationLevel.values()[optlevel];
	}
	
	public static boolean isMemoryBasedOptLevel() {
		return (getOptLevel() != OptimizationLevel.O0_LOCAL_STATIC);
	}
	
	public static boolean isOptLevel( OptimizationLevel level ){
		return (getOptLevel() == level);
	}
	
	public static CompilerConfig constructCompilerConfig( DMLConfig dmlconf ) 
		throws DMLRuntimeException
	{
		//create default compiler configuration
		CompilerConfig cconf = new CompilerConfig();
		
		//each script sets its own block size, opt level etc
		cconf.set(ConfigType.BLOCK_SIZE, dmlconf.getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ));

		//handle optimization level
		int optlevel = dmlconf.getIntValue(DMLConfig.OPTIMIZATION_LEVEL);
		if( optlevel < 0 || optlevel > 7 )
			throw new DMLRuntimeException("Error: invalid optimization level '"+optlevel+"' (valid values: 0-5).");
	
		// This overrides any optimization level that is present in the configuration file.
		// Why ? This simplifies the calling logic: User doesnot have to maintain two config file or worse
		// edit config file everytime he/she is trying to call the debugger.
		if(DMLScript.ENABLE_DEBUG_MODE) {
			optlevel = 5;
		}
		
		switch( optlevel )
		{
			// opt level 0: static dimensionality
			case 0:
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O0_LOCAL_STATIC.ordinal());
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
				ALLOW_AUTO_VECTORIZATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				ALLOW_IPA_SECOND_CHANCE = false;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_SUM_PRODUCT_REWRITES = false;
				break;
			// opt level 1: memory-based (no advanced rewrites)	
			case 1:
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O1_LOCAL_MEMORY_MIN.ordinal());
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
				ALLOW_AUTO_VECTORIZATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				ALLOW_IPA_SECOND_CHANCE = false;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_SUM_PRODUCT_REWRITES = false;
				ALLOW_LOOP_UPDATE_IN_PLACE = false;
				break;
			// opt level 2: memory-based (all advanced rewrites)
			case 2:
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O2_LOCAL_MEMORY_DEFAULT.ordinal());
				break;
			// opt level 3: resource optimization, time- and memory-based (2 w/ resource optimizat)
			case 3:
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O3_LOCAL_RESOURCE_TIME_MEMORY.ordinal());
			break;
							
			// opt level 3: global, time- and memory-based (all advanced rewrites)
			case 4:
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O4_GLOBAL_TIME_MEMORY.ordinal());
				break;
			// opt level 4: debug mode (no interfering rewrites)
			case 5:				
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O5_DEBUG_MODE.ordinal());
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_SIZE_EXPRESSION_EVALUATION = false;
				ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = false;
				ALLOW_RAND_JOB_RECOMPILE = false;
				ALLOW_SUM_PRODUCT_REWRITES = false;
				ALLOW_SPLIT_HOP_DAGS = false;
				cconf.set(ConfigType.ALLOW_DYN_RECOMPILATION, false);
				cconf.set(ConfigType.ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS, false);
				break;
			
			// opt level 6 and7: SPOOF w/o fused operators, otherwise same as O2
			// (hidden optimization levels not documented on purpose, as they will
			// be removed once SPOOF is production ready)	
			case 6:
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O2_LOCAL_MEMORY_DEFAULT.ordinal());
				ALLOW_AUTO_VECTORIZATION = false;
				break;
			case 7:				
				cconf.set(ConfigType.OPT_LEVEL, OptimizationLevel.O2_LOCAL_MEMORY_DEFAULT.ordinal());
				ALLOW_OPERATOR_FUSION = false;
				ALLOW_AUTO_VECTORIZATION = false;
				ALLOW_SUM_PRODUCT_REWRITES = false;
				break;	
		}
		
		//handle parallel text io (incl awareness of thread contention in <jdk8)
		if (!dmlconf.getBooleanValue(DMLConfig.CP_PARALLEL_TEXTIO)) {
			cconf.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, false);
			cconf.set(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS, false);
			cconf.set(ConfigType.PARALLEL_CP_READ_BINARYFORMATS, false);
			cconf.set(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS, false);
		}
		else if(   InfrastructureAnalyzer.isJavaVersionLessThanJDK8() 
			    && InfrastructureAnalyzer.getLocalParallelism() > 1   )
		{
			LOG.warn("Auto-disable multi-threaded text read for 'text' and 'csv' due to thread contention on JRE < 1.8"
					+ " (java.version="+ System.getProperty("java.version")+").");			
			cconf.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, false);
		}

		//handle parallel matrix mult / rand configuration
		if (!dmlconf.getBooleanValue(DMLConfig.CP_PARALLEL_MATRIXMULT)) {
			cconf.set(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS, false);
		}	
		
		return cconf;
	}

	public static long getDefaultSize() {
		//we need to set default_size larger than any execution context
		//memory budget, however, it should not produce overflows on sum
		return Math.max( InfrastructureAnalyzer.getLocalMaxMemory(),
					Math.max(InfrastructureAnalyzer.getRemoteMaxMemoryMap(),
				          InfrastructureAnalyzer.getRemoteMaxMemoryReduce()));
	}
	
	public static void resetDefaultSize() {
		DEFAULT_SIZE = getDefaultSize();
	}
	
	
	public static int getDefaultFrameSize()
	{
		return DEFAULT_FRAME_BLOCKSIZE;
	}
	
	/**
	 * Returns memory budget (according to util factor) in bytes
	 * 
	 * @return local memory budget
	 */
	public static double getLocalMemBudget()
	{
		double ret = InfrastructureAnalyzer.getLocalMaxMemory();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	public static double getRemoteMemBudgetMap()
	{
		return getRemoteMemBudgetMap(false);
	}
	
	public static double getRemoteMemBudgetMap(boolean substractSortBuffer)
	{
		double ret = InfrastructureAnalyzer.getRemoteMaxMemoryMap();
		if( substractSortBuffer )
			ret -= InfrastructureAnalyzer.getRemoteMaxMemorySortBuffer();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}

	public static double getRemoteMemBudgetReduce()
	{
		double ret = InfrastructureAnalyzer.getRemoteMaxMemoryReduce();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}

	public static boolean checkSparkBroadcastMemoryBudget( double size )
	{
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		//basic requirement: the broadcast needs to to fit once in the remote broadcast memory 
		//and twice into the local memory budget because we have to create a partitioned broadcast
		//memory and hand it over to the spark context as in-memory object
		return ( size < memBudgetExec && 2*size < memBudgetLocal );
	}

	public static boolean checkSparkBroadcastMemoryBudget( long rlen, long clen, long brlen, long bclen, long nnz )
	{
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		double sp = getSparsity(rlen, clen, nnz);
		double size = estimateSizeExactSparsity(rlen, clen, sp);
		double sizeP = estimatePartitionedSizeExactSparsity(rlen, clen, brlen, bclen, sp);
		
		//basic requirement: the broadcast needs to to fit once in the remote broadcast memory 
		//and twice into the local memory budget because we have to create a partitioned broadcast
		//memory and hand it over to the spark context as in-memory object
		return (   OptimizerUtils.isValidCPDimensions(rlen, clen)
				&& sizeP < memBudgetExec && size+sizeP < memBudgetLocal );
	}

	public static boolean checkSparkCollectMemoryBudget( MatrixCharacteristics mc, long memPinned )
	{
		return checkSparkCollectMemoryBudget(
				mc.getRows(), 
				mc.getCols(),
				mc.getRowsPerBlock(),
				mc.getColsPerBlock(),
				mc.getNonZeros(), memPinned);
	}
	
	public static boolean checkSparkCollectMemoryBudget( long rlen, long clen, int brlen, int bclen, long nnz, long memPinned )
	{
		//compute size of output matrix and its blocked representation
		double sp = getSparsity(rlen, clen, nnz);
		double memMatrix = estimateSizeExactSparsity(rlen, clen, sp);
		double memPMatrix = estimatePartitionedSizeExactSparsity(rlen, clen, brlen, bclen, sp);
		
		//check if both output matrix and partitioned matrix fit into local mem budget
		return (memPinned + memMatrix + memPMatrix < getLocalMemBudget());
	}

	public static boolean checkSparseBlockCSRConversion( MatrixCharacteristics mcIn ) {
		return Checkpoint.CHECKPOINT_SPARSE_CSR
			&& OptimizerUtils.getSparsity(mcIn) < MatrixBlock.SPARSITY_TURN_POINT;
	}
	
	/**
	 * Returns the number of reducers that potentially run in parallel.
	 * This is either just the configured value (SystemML config) or
	 * the minimum of configured value and available reduce slots. 
	 * 
	 * @param configOnly true if configured value
	 * @return number of reducers
	 */
	public static int getNumReducers( boolean configOnly )
	{
		if( isSparkExecutionMode() )
			return SparkExecutionContext.getDefaultParallelism(false);
		
		int ret = ConfigurationManager.getNumReducers();
		if( !configOnly ) {
			ret = Math.min(ret,InfrastructureAnalyzer.getRemoteParallelReduceTasks());
			
			//correction max number of reducers on yarn clusters
			if( InfrastructureAnalyzer.isYarnEnabled() )
				ret = (int)Math.max( ret, YarnClusterAnalyzer.getNumCores()/2 );
		}
		
		return ret;
	}

	public static int getNumMappers()
	{
		if( isSparkExecutionMode() )
			return SparkExecutionContext.getDefaultParallelism(false);
		
		int ret = InfrastructureAnalyzer.getRemoteParallelMapTasks();
			
		//correction max number of reducers on yarn clusters
		if( InfrastructureAnalyzer.isYarnEnabled() )
			ret = (int)Math.max( ret, YarnClusterAnalyzer.getNumCores() );
		
		return ret;
	}

	public static RUNTIME_PLATFORM getDefaultExecutionMode() {
		//default execution type is hybrid (cp+mr)
		RUNTIME_PLATFORM ret = RUNTIME_PLATFORM.HYBRID;
		
		//switch default to hybrid_spark (cp+spark) if in spark driver
		String sparkenv = System.getenv().get("SPARK_ENV_LOADED");
		if( sparkenv != null && sparkenv.equals("1") )
			ret = RUNTIME_PLATFORM.HYBRID_SPARK;
		
		return ret;
	}

	public static boolean isSparkExecutionMode() {
		return (   DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK
				|| DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK);
	}

	public static boolean isHadoopExecutionMode() {
		return (   DMLScript.rtplatform == RUNTIME_PLATFORM.HADOOP
				|| DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID);
	}

	public static boolean isHybridExecutionMode() {
		return (  DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID 
			   || DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK );
	}
	
	/**
	 * Returns the degree of parallelism used for parallel text read. 
	 * This is computed as the number of virtual cores scales by the 
	 * PARALLEL_READ_PARALLELISM_MULTIPLIER. If PARALLEL_READ_TEXTFORMATS
	 * is disabled, this method returns 1.
	 * 
	 * @return degree of parallelism
	 */
	public static int getParallelTextReadParallelism()
	{
		if( !ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_TEXTFORMATS) )
			return 1; // sequential execution
			
		//compute degree of parallelism for parallel text read
		double dop = InfrastructureAnalyzer.getLocalParallelism()
				     * PARALLEL_CP_READ_PARALLELISM_MULTIPLIER;
		return (int) Math.round(dop);
	}

	public static int getParallelBinaryReadParallelism()
	{
		if( !ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_BINARYFORMATS) )
			return 1; // sequential execution
			
		//compute degree of parallelism for parallel text read
		double dop = InfrastructureAnalyzer.getLocalParallelism()
				     * PARALLEL_CP_READ_PARALLELISM_MULTIPLIER;
		return (int) Math.round(dop);
	}
	
	/**
	 * Returns the degree of parallelism used for parallel text write. 
	 * This is computed as the number of virtual cores scales by the 
	 * PARALLEL_WRITE_PARALLELISM_MULTIPLIER. If PARALLEL_WRITE_TEXTFORMATS
	 * is disabled, this method returns 1.
	 * 
	 * @return degree of parallelism
	 */
	public static int getParallelTextWriteParallelism()
	{
		if( !ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
			return 1; // sequential execution

		//compute degree of parallelism for parallel text read
		double dop = InfrastructureAnalyzer.getLocalParallelism()
				     * PARALLEL_CP_WRITE_PARALLELISM_MULTIPLIER;
		return (int) Math.round(dop);
	}

	public static int getParallelBinaryWriteParallelism()
	{
		if( !ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS) )
			return 1; // sequential execution

		//compute degree of parallelism for parallel text read
		double dop = InfrastructureAnalyzer.getLocalParallelism()
				     * PARALLEL_CP_WRITE_PARALLELISM_MULTIPLIER;
		return (int) Math.round(dop);
	}
	
	////////////////////////
	// Memory Estimates   //
	////////////////////////
	
	public static long estimateSize(MatrixCharacteristics mc) {
		return estimateSizeExactSparsity(mc);
	}

	public static long estimateSizeExactSparsity(MatrixCharacteristics mc)
	{
		return estimateSizeExactSparsity(
				mc.getRows(),
				mc.getCols(),
				mc.getNonZeros());
	}
	
	/**
	 * Estimates the footprint (in bytes) for an in-memory representation of a
	 * matrix with dimensions=(nrows,ncols) and and number of non-zeros nnz.
	 * 
	 * @param nrows number of rows
	 * @param ncols number of cols
	 * @param nnz number of non-zeros
	 * @return memory footprint
	 */
	public static long estimateSizeExactSparsity(long nrows, long ncols, long nnz) 
	{
		double sp = getSparsity(nrows, ncols, nnz);
		return estimateSizeExactSparsity(nrows, ncols, sp);
	}
	
	/**
	 * Estimates the footprint (in bytes) for an in-memory representation of a
	 * matrix with dimensions=(nrows,ncols) and sparsity=sp.
	 * 
	 * This function can be used directly in Hops, when the actual sparsity is
	 * known i.e., <code>sp</code> is guaranteed to give worst-case estimate
	 * (e.g., Rand with a fixed sparsity). In all other cases, estimateSize()
	 * must be used so that worst-case estimates are computed, whenever
	 * applicable.
	 * 
	 * @param nrows number of rows
	 * @param ncols number of cols
	 * @param sp sparsity
	 * @return memory footprint
	 */
	public static long estimateSizeExactSparsity(long nrows, long ncols, double sp) 
	{
		return MatrixBlock.estimateSizeInMemory(nrows,ncols,sp);
	}
	
	/**
	 * Estimates the footprint (in bytes) for a partitioned in-memory representation of a
	 * matrix with the given matrix characteristics
	 * 
	 * @param mc matrix characteristics
	 * @return memory estimate
	 */
	public static long estimatePartitionedSizeExactSparsity(MatrixCharacteristics mc)
	{
		return estimatePartitionedSizeExactSparsity(
				mc.getRows(), 
				mc.getCols(), 
				mc.getRowsPerBlock(), 
				mc.getColsPerBlock(), 
				mc.getNonZeros());
	}
	
	/**
	 * Estimates the footprint (in bytes) for a partitioned in-memory representation of a
	 * matrix with dimensions=(nrows,ncols) and number of non-zeros nnz.
	 * 
	 * @param rlen number of rows
	 * @param clen number of cols
	 * @param brlen rows per block
	 * @param bclen cols per block
	 * @param nnz number of non-zeros
	 * @return memory estimate
	 */
	public static long estimatePartitionedSizeExactSparsity(long rlen, long clen, long brlen, long bclen, long nnz) 
	{
		double sp = getSparsity(rlen, clen, nnz);
		return estimatePartitionedSizeExactSparsity(rlen, clen, brlen, bclen, sp);
	}
	
	/**
	 * Estimates the footprint (in bytes) for a partitioned in-memory representation of a
	 * matrix with dimensions=(nrows,ncols) and sparsity=sp.
	 * 
	 * @param rlen number of rows
	 * @param clen number of cols
	 * @param brlen rows per block
	 * @param bclen cols per block
	 * @param sp sparsity
	 * @return memory estimate
	 */
	public static long estimatePartitionedSizeExactSparsity(long rlen, long clen, long brlen, long bclen, double sp) 
	{
		long ret = 0;

		//check for guaranteed existence of empty blocks (less nnz than total number of blocks)
		long tnrblks = (long)Math.ceil((double)rlen/brlen);
		long tncblks = (long)Math.ceil((double)clen/bclen);
		long nnz = (long) Math.ceil(sp * rlen * clen);		
		if( nnz < tnrblks * tncblks ) {
			long lrlen = Math.min(rlen, brlen);
			long lclen = Math.min(clen, bclen);
			return nnz * estimateSizeExactSparsity(lrlen, lclen, 1)
				 + (tnrblks * tncblks - nnz) * estimateSizeEmptyBlock(lrlen, lclen);
		}
		
		//estimate size of full brlen x bclen blocks
		long nrblks = rlen / brlen;
		long ncblks = clen / bclen;
		if( nrblks * ncblks > 0 )
			ret += nrblks * ncblks * estimateSizeExactSparsity(brlen, bclen, sp);

		//estimate size of bottom boundary blocks 
		long lrlen = rlen % brlen;
		if( ncblks > 0 && lrlen > 0 )
			ret += ncblks * estimateSizeExactSparsity(lrlen, bclen, sp);
		
		//estimate size of right boundary blocks
		long lclen = clen % bclen;
		if( nrblks > 0 && lclen > 0 )
			ret += nrblks * estimateSizeExactSparsity(brlen, lclen, sp);
		
		//estimate size of bottom right boundary block
		if( lrlen > 0 && lclen > 0  )
			ret += estimateSizeExactSparsity(lrlen, lclen, sp);
		
		return ret;
	}
	
	/**
	 * Similar to estimate() except that it provides worst-case estimates
	 * when the optimization type is ROBUST.
	 * 
	 * @param nrows number of rows
	 * @param ncols number of cols
	 * @return memory estimate
	 */
	public static long estimateSize(long nrows, long ncols) 
	{
		return estimateSizeExactSparsity(nrows, ncols, 1.0);
	}
	
	public static long estimateSizeEmptyBlock(long nrows, long ncols)
	{
		return estimateSizeExactSparsity(0, 0, 0.0d);
	}

	public static long estimateSizeTextOutput( long rows, long cols, long nnz, OutputInfo oinfo )
	{
		long bsize = MatrixBlock.estimateSizeOnDisk(rows, cols, nnz);
		if( oinfo == OutputInfo.TextCellOutputInfo || oinfo == OutputInfo.MatrixMarketOutputInfo )
			return bsize * 3;
		else if( oinfo == OutputInfo.CSVOutputInfo )
			return bsize * 2;
		
		//unknown output info
		return bsize;
	}
	
	/**
	 * Indicates if the given indexing range is block aligned, i.e., it does not require
	 * global aggregation of blocks.
	 * 
	 * @param ixrange indexing range
	 * @param mc matrix characteristics
	 * @return true if indexing range is block aligned
	 */
	public static boolean isIndexingRangeBlockAligned(IndexRange ixrange, MatrixCharacteristics mc) {
		long rl = ixrange.rowStart;
		long ru = ixrange.rowEnd;
		long cl = ixrange.colStart;
		long cu = ixrange.colEnd;
		long brlen = mc.getRowsPerBlock();
		long bclen = mc.getColsPerBlock();
		return isIndexingRangeBlockAligned(rl, ru, cl, cu, brlen, bclen);
	}
	
	/**
	 * Indicates if the given indexing range is block aligned, i.e., it does not require
	 * global aggregation of blocks.
	 * 
	 * @param rl rows lower
	 * @param ru rows upper
	 * @param cl cols lower
	 * @param cu cols upper
	 * @param brlen rows per block
	 * @param bclen cols per block
	 * @return true if indexing range is block aligned
	 */
	public static boolean isIndexingRangeBlockAligned(long rl, long ru, long cl, long cu, long brlen, long bclen) {
		return rl != -1 && ru != -1 && cl != -1 && cu != -1
				&&((rl-1)%brlen == 0 && (cl-1)%bclen == 0 
				|| (rl-1)/brlen == (ru-1)/brlen && (cl-1)%bclen == 0 
				|| (rl-1)%brlen == 0 && (cl-1)/bclen == (cu-1)/bclen);
	}
	/**
	 * Returns false if dimensions known to be invalid; other true
	 * 
	 * @param rows number of rows
	 * @param cols number of cols
	 * @return true if dimensions valid
	 */
	public static boolean isValidCPDimensions( long rows, long cols )
	{
		//the current CP runtime implementation requires that rows and cols
		//are integers since we use a single matrixblock to represent the
		//entire matrix
		return (rows <= Integer.MAX_VALUE && cols<=Integer.MAX_VALUE);
	}
	
	/**
	 * Returns false if schema and names are not properly specified; other true
	 * Length to be &gt; 0, and length of both to be equal.
	 * 
	 * @param schema the schema
	 * @param names the names
	 * @return false if schema and names are not properly specified
	 */
	public static boolean isValidCPDimensions( ValueType[] schema, String[] names )
	{
		// Length of schema and names to be same, and > 0.
		return (schema != null && names != null && schema.length > 0 && schema.length == names.length);
	}
	
	/**
	 * Determines if valid matrix size to be represented in CP data structures. Note that
	 * sparsity needs to be specified as rows*cols if unknown. 
	 * 
	 * @param rows number of rows
	 * @param cols number of cols
	 * @param sparsity the sparsity
	 * @return true if valid matrix size
	 */
	public static boolean isValidCPMatrixSize( long rows, long cols, double sparsity )
	{
		boolean ret = true;
		
		//the current CP runtime implementation has several limitations:
		//1) for dense: 16GB because we use a linearized array (bounded to int in java)
		//2) for sparse: 2G x 2G nnz because (1) nnz maintained as long, (2) potential changes 
		//   to dense, and (3) sparse row arrays also of max int size (worst case in case of skew)  
		long nnz = (long)(sparsity * rows * cols);
		boolean sparse = MatrixBlock.evalSparseFormatInMemory(rows, cols, nnz);
		
		if( sparse ) //SPARSE
		{
			//check max nnz (dependent on sparse block format)
			ret = (nnz <= MAX_NNZ_CP_SPARSE);
		}
		else //DENSE
		{
			//check number of matrix cell
			ret = ((rows * cols) <= MAX_NUMCELLS_CP_DENSE);
		}
			
		return ret;
	}
	
	/**
	 * Indicates if the given matrix characteristics exceed the threshold for 
	 * caching, i.e., the matrix should be cached.
	 * 
	 * @param dim2 dimension 2
	 * @param outMem ?
	 * @return true if the given matrix characteristics exceed threshold
	 */
	public static boolean exceedsCachingThreshold(long dim2, double outMem) {
		return !(dim2 > 1 && outMem < getLocalMemBudget()
			|| dim2 == 1 && outMem < getLocalMemBudget()/3);
	}
	
	/**
	 * Wrapper over internal filename construction for external usage. 
	 * 
	 * @return unique temp file name
	 */
	public static String getUniqueTempFileName() {
		return ConfigurationManager.getScratchSpace()
			+ Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID()
			+ Lop.FILE_SEPARATOR + ProgramConverter.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR 
			+ Dag.getNextUniqueFilenameSuffix();
	}

	public static boolean allowsToFilterEmptyBlockOutputs( Hop hop ) 
		throws HopsException
	{
		boolean ret = true;
		for( Hop p : hop.getParent() ) {
			p.optFindExecType(); //ensure exec type evaluated
			ret &=   (  p.getExecType()==ExecType.CP 
					 ||(p instanceof AggBinaryOp && allowsToFilterEmptyBlockOutputs(p) )
					 ||(p instanceof DataOp && ((DataOp)p).getDataOpType()==DataOpTypes.PERSISTENTWRITE && ((DataOp)p).getInputFormatType()==FileFormatTypes.TEXT))
				  && !(p instanceof FunctionOp || (p instanceof DataOp && ((DataOp)p).getInputFormatType()!=FileFormatTypes.TEXT) ); //no function call or transient write
		}
			
		return ret;	
	}

	public static int getConstrainedNumThreads(int maxNumThreads)
	{
		//by default max local parallelism (vcores) 
		int ret = InfrastructureAnalyzer.getLocalParallelism();
		
		//apply external max constraint (e.g., set by parfor or other rewrites)
		if( maxNumThreads > 0 ) {
			ret = Math.min(ret, maxNumThreads);
		}
		
		//apply global multi-threading constraint
		if( !ConfigurationManager.isParallelMatrixOperations() ) {
			ret = 1;
		}
			
		return ret;
	}
	
	////////////////////////
	// Sparsity Estimates //
	////////////////////////
	
	/**
	 * Estimates the result sparsity for Matrix Multiplication A %*% B. 
	 *  
	 * @param sp1 sparsity of A
	 * @param sp2 sparsity of B
	 * @param m nrow(A)
	 * @param k ncol(A), nrow(B)
	 * @param n ncol(B)
	 * @param worstcase true if worst case
	 * @return the sparsity
	 */
	public static double getMatMultSparsity(double sp1, double sp2, long m, long k, long n, boolean worstcase) 
	{
		if( worstcase ){
			double nnz1 = sp1 * m * k;
			double nnz2 = sp2 * k * n;
			return Math.min(1, nnz1/m) * Math.min(1, nnz2/n);
		}
		else
			return (1 - Math.pow(1-sp1*sp2, k) );
	}

	public static double getLeftIndexingSparsity( long rlen1, long clen1, long nnz1, long rlen2, long clen2, long nnz2 )
	{
		boolean scalarRhs = (rlen2==0 && clen2==0);
		
		//infer output worstcase output nnz
		long lnnz = -1;
		if( nnz1>=0 && scalarRhs )
			lnnz = nnz1+1;             // nnz(left) + scalar
		else if( nnz1>=0 && nnz2>=0 )
			lnnz = nnz1 + nnz2;        // nnz(left) + nnz(right)
		else if( nnz1>=0 && rlen2>0 && clen2>0 )
			lnnz = nnz1 + rlen2*clen2; // nnz(left) + nnz(right_dense)
		lnnz = Math.min(lnnz, rlen1*clen1);
		
		return getSparsity(rlen1, clen1, (lnnz>=0) ? lnnz : rlen1*clen1);
	}
	
	/**
	 * Determines if a given binary op is potentially conditional sparse safe. 
	 * 
	 * @param op the HOP OpOp2
	 * @return true if potentially conditional sparse safe
	 */
	public static boolean isBinaryOpConditionalSparseSafe( OpOp2 op ) 
	{
		return (   op==OpOp2.GREATER 
			    || op==OpOp2.LESS 
			    || op==OpOp2.NOTEQUAL 
			    || op==OpOp2.EQUAL 
			    || op==OpOp2.MINUS);
	}
	
	/**
	 * Determines if a given binary op with scalar literal guarantee an output
	 * sparsity which is exactly the same as its matrix input sparsity.
	 * 
	 * @param op the HOP OpOp2
	 * @param lit literal operator
	 * @return true if output sparsity same as matrix input sparsity
	 */
	public static boolean isBinaryOpConditionalSparseSafeExact( OpOp2 op, LiteralOp lit )
	{
		double val = HopRewriteUtils.getDoubleValueSafe(lit);
		
		return ( op==OpOp2.NOTEQUAL && val==0);
	}

	public static double getBinaryOpSparsityConditionalSparseSafe( double sp1, OpOp2 op, LiteralOp lit )
	{
		double val = HopRewriteUtils.getDoubleValueSafe(lit);
		
		return (  (op==OpOp2.GREATER  && val==0) 
				||(op==OpOp2.LESS     && val==0)
				||(op==OpOp2.NOTEQUAL && val==0)
				||(op==OpOp2.EQUAL    && val!=0)
				||(op==OpOp2.MINUS    && val==0)) ? sp1 : 1.0;
	}
	
	/**
	 * Estimates the result sparsity for matrix-matrix binary operations (A op B)
	 * 
	 * @param sp1 sparsity of A
	 * @param sp2 sparsity of B
	 * @param op binary operation
	 * @param worstcase true if worst case
	 * @return result sparsity for matrix-matrix binary operations
	 */
	public static double getBinaryOpSparsity(double sp1, double sp2, OpOp2 op, boolean worstcase) 
	{
		// default is worst-case estimate for robustness
		double ret = 1.0;
		
		if( worstcase )
		{
			//NOTE: for matrix-scalar operations this estimate is too conservative, because 
			//Math.min(1, sp1 + sp2) will always give a sparsity 1 if we pass sp2=1 for scalars.
			//In order to do better (with guarantees), we need to take the actual values into account  
			switch(op) 
			{
				case PLUS:
				case MINUS:
				case LESS: 
				case GREATER:
				case NOTEQUAL:
				case MIN:
				case MAX:
				case OR:
					ret = Math.min(1, sp1 + sp2); break;
				case MULT:
				case AND:
					ret = Math.min(sp1, sp2); break;
				case DIV:
				case MODULUS:
				case POW:
				case MINUS_NZ:
				case LOG_NZ:	
					ret = sp1; break; 
				//case EQUAL: //doesnt work on worstcase estimates, but on 
				//	ret = 1-Math.abs(sp1-sp2); break;	
	
				default:
					ret = 1.0;
			}
		}
		else
		{
			switch(op) {			
				case PLUS:
				case MINUS:
					// result[i,j] != 0 iff A[i,j] !=0 || B[i,j] != 0
					// worst case estimate = sp1+sp2
					ret = (1 - (1-sp1)*(1-sp2)); 
					break;
					
				case MULT:
					// result[i,j] != 0 iff A[i,j] !=0 && B[i,j] != 0
					// worst case estimate = min(sp1,sp2)
					ret = sp1 * sp2;  
					break;
					
				case DIV:
					ret = 1.0; // worst case estimate
					break;
					
				case LESS: 
				case LESSEQUAL:
				case GREATER:
				case GREATEREQUAL:
				case EQUAL: 
				case NOTEQUAL:
					ret = 1.0; // purely data-dependent operations, and hence worse-case estimate
					break;
					
				//MIN, MAX, AND, OR, LOG, POW
				default:
					ret = 1.0;
			}
		}
		
		return ret; 
	}
	
	public static double getSparsity( MatrixCharacteristics mc ) {
		return getSparsity(mc.getRows(), mc.getCols(), mc.getNonZeros());
	}
	
	public static double getSparsity( long dim1, long dim2, long nnz )
	{
		if( dim1<=0 || dim2<=0 || nnz<0 )
			return 1.0;
		else
			return Math.min(((double)nnz)/dim1/dim2, 1.0);
	}
	
	public static String toMB(double inB) {
		if ( inB < 0 )
			return "-";
		return String.format("%.0f", inB/(1024*1024) );
	}
	

	
	/**
	 * Function to evaluate simple size expressions over literals and now/ncol.
	 * 
	 * It returns the exact results of this expressions if known, otherwise
	 * Long.MAX_VALUE if unknown.
	 * 
	 * @param root the root high-level operator
	 * @param valMemo ?
	 * @return size expression
	 * @throws HopsException if HopsException occurs
	 */
	public static long rEvalSimpleLongExpression( Hop root, HashMap<Long, Long> valMemo ) 
		throws HopsException
	{
		long ret = Long.MAX_VALUE;
		
		//for simplicity and robustness call double and cast.
		HashMap<Long, Double> dvalMemo = new HashMap<Long, Double>();
		double tmp = rEvalSimpleDoubleExpression(root, dvalMemo);
		if( tmp!=Double.MAX_VALUE )
			ret = UtilFunctions.toLong( tmp );
		
		return ret;
	}
	
	public static long rEvalSimpleLongExpression( Hop root, HashMap<Long, Long> valMemo, LocalVariableMap vars ) 
		throws HopsException
	{
		long ret = Long.MAX_VALUE;
		
		//for simplicity and robustness call double and cast.
		HashMap<Long, Double> dvalMemo = new HashMap<Long, Double>();
		double tmp = rEvalSimpleDoubleExpression(root, dvalMemo, vars);
		if( tmp!=Double.MAX_VALUE )
			ret = UtilFunctions.toLong( tmp );
		
		return ret;
	}
	
	public static double rEvalSimpleDoubleExpression( Hop root, HashMap<Long, Double> valMemo ) 
		throws HopsException
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;
		
		//always use constants
		if( root instanceof LiteralOp )
			ret = HopRewriteUtils.getDoubleValue((LiteralOp)root);
		
		//advanced size expression evaluation
		if( OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION )
		{
			if( root instanceof UnaryOp )
				ret = rEvalSimpleUnaryDoubleExpression(root, valMemo);
			else if( root instanceof BinaryOp )
				ret = rEvalSimpleBinaryDoubleExpression(root, valMemo);
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}
	
	public static double rEvalSimpleDoubleExpression( Hop root, HashMap<Long, Double> valMemo, LocalVariableMap vars ) 
		throws HopsException
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;
		
		if( OptimizerUtils.ALLOW_SIZE_EXPRESSION_EVALUATION )
		{
			if( root instanceof LiteralOp )
				ret = HopRewriteUtils.getDoubleValue((LiteralOp)root);
			else if( root instanceof UnaryOp )
				ret = rEvalSimpleUnaryDoubleExpression(root, valMemo, vars);
			else if( root instanceof BinaryOp )
				ret = rEvalSimpleBinaryDoubleExpression(root, valMemo, vars);
			else if( root instanceof DataOp ) {
				String name = root.getName();
				Data dat = vars.get(name);
				if( dat!=null && dat instanceof ScalarObject )
					ret = ((ScalarObject)dat).getDoubleValue();
			}
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}

	protected static double rEvalSimpleUnaryDoubleExpression( Hop root, HashMap<Long, Double> valMemo ) 
		throws HopsException
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;
		
		UnaryOp uroot = (UnaryOp) root;
		Hop input = uroot.getInput().get(0);
		
		if(uroot.getOp() == Hop.OpOp1.NROW)
			ret = (input.getDim1()>0) ? input.getDim1() : Double.MAX_VALUE;
		else if( uroot.getOp() == Hop.OpOp1.NCOL )
			ret = (input.getDim2()>0) ? input.getDim2() : Double.MAX_VALUE;
		else
		{
			double lval = rEvalSimpleDoubleExpression(uroot.getInput().get(0), valMemo);
			if( lval != Double.MAX_VALUE )
			{
				switch( uroot.getOp() )
				{
					case SQRT:	ret = Math.sqrt(lval); break;
					case ROUND: ret = Math.round(lval); break;
					case CAST_AS_BOOLEAN: ret = (lval!=0)? 1 : 0; break;
					case CAST_AS_INT: ret = UtilFunctions.toLong(lval); break;
					case CAST_AS_DOUBLE: ret = lval; break;
					default: ret = Double.MAX_VALUE;
				}
			}
		}
			
		valMemo.put(root.getHopID(), ret);
		return ret;
	}

	protected static double rEvalSimpleUnaryDoubleExpression( Hop root, HashMap<Long, Double> valMemo, LocalVariableMap vars ) 
		throws HopsException
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;
		
		UnaryOp uroot = (UnaryOp) root;
		Hop input = uroot.getInput().get(0);
		
		if(uroot.getOp() == Hop.OpOp1.NROW)
			ret = (input.getDim1()>0) ? input.getDim1() : Double.MAX_VALUE;
		else if( uroot.getOp() == Hop.OpOp1.NCOL )
			ret = (input.getDim2()>0) ? input.getDim2() : Double.MAX_VALUE;
		else
		{
			double lval = rEvalSimpleDoubleExpression(uroot.getInput().get(0), valMemo, vars);
			if( lval != Double.MAX_VALUE )
			{
				switch( uroot.getOp() )
				{
					case SQRT:	ret = Math.sqrt(lval); break;
					case ROUND: ret = Math.round(lval); break;
					case CAST_AS_BOOLEAN: ret = (lval!=0)? 1 : 0; break;
					case CAST_AS_INT: ret = UtilFunctions.toLong(lval); break;
					case CAST_AS_DOUBLE: ret = lval; break;
					default: ret = Double.MAX_VALUE;
				}
			}
		}
			
		valMemo.put(root.getHopID(), ret);
		return ret;
	}

	protected static double rEvalSimpleBinaryDoubleExpression( Hop root, HashMap<Long, Double> valMemo ) 
		throws HopsException
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;

		BinaryOp broot = (BinaryOp) root;
		
		double lret = rEvalSimpleDoubleExpression(broot.getInput().get(0), valMemo);
		double rret = rEvalSimpleDoubleExpression(broot.getInput().get(1), valMemo);
		//note: positive and negative values might be valid subexpressions
		if( lret!=Double.MAX_VALUE && rret!=Double.MAX_VALUE ) //if known
		{
			switch( broot.getOp() )
			{
				case PLUS:	ret = lret + rret; break;
				case MINUS:	ret = lret - rret; break;
				case MULT:  ret = lret * rret; break;
				case DIV:   ret = lret / rret; break;
				case MIN:   ret = Math.min(lret, rret); break;
				case MAX:   ret = Math.max(lret, rret); break;
				case POW:   ret = Math.pow(lret, rret); break; 
				//special mod / inddiv for runtime consistency
				case MODULUS: ret = Modulus.getFnObject().execute(lret, rret); break;
				case INTDIV:  ret = IntegerDivide.getFnObject().execute(lret, rret); break; 
				default: ret = Double.MAX_VALUE;
			}
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}

	protected static double rEvalSimpleBinaryDoubleExpression( Hop root, HashMap<Long, Double> valMemo, LocalVariableMap vars ) 
		throws HopsException
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;

		BinaryOp broot = (BinaryOp) root;
		
		double lret = rEvalSimpleDoubleExpression(broot.getInput().get(0), valMemo, vars);
		double rret = rEvalSimpleDoubleExpression(broot.getInput().get(1), valMemo, vars);
		//note: positive and negative values might be valid subexpressions
		if( lret!=Double.MAX_VALUE && rret!=Double.MAX_VALUE ) //if known
		{
			switch( broot.getOp() )
			{
				case PLUS:	ret = lret + rret; break;
				case MINUS:	ret = lret - rret; break;
				case MULT:  ret = lret * rret; break;
				case DIV:   ret = lret / rret; break;
				case MIN:   ret = Math.min(lret, rret); break;
				case MAX:   ret = Math.max(lret, rret); break;
				case POW:   ret = Math.pow(lret, rret); break; 
				//special mod / inddiv for runtime consistency
				case MODULUS: ret = Modulus.getFnObject().execute(lret, rret); break;
				case INTDIV:  ret = IntegerDivide.getFnObject().execute(lret, rret); break; 
				default: ret = Double.MAX_VALUE;
			}
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}		
}

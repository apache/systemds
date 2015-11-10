/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.hops;

import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.hops.rewrite.HopRewriteUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.yarn.ropt.YarnClusterAnalyzer;

public class OptimizerUtils 
{
	private static final Log LOG = LogFactory.getLog(OptimizerUtils.class.getName());
	
	////////////////////////////////////////////////////////
	// Optimizer constants and flags (incl tuning knobs)  //
	////////////////////////////////////////////////////////
	/**
	 * Utilization factor used in deciding whether an operation to be scheduled on CP or MR. 
	 * NOTE: it is important that MEM_UTIL_FACTOR+CacheableData.CACHING_BUFFER_SIZE < 1.0
	 */
	public static double MEM_UTIL_FACTOR = 0.7d;
	
	/**
	 * Default memory size, which is used the actual estimate can not be computed 
	 * -- for example, when input/output dimensions are unknown. In case of ROBUST,
	 * the default is set to a large value so that operations are scheduled on MR.  
	 */
	public static double DEFAULT_SIZE;	
	
	public static final long DOUBLE_SIZE = 8;
	public static final long INT_SIZE = 4;
	public static final long CHAR_SIZE = 1;
	public static final long BOOLEAN_SIZE = 1;
	public static final double BIT_SIZE = (double)1/8;
	public static final double INVALID_SIZE = -1d; // memory estimate not computed

	public static final long MAX_NUMCELLS_CP_DENSE = Integer.MAX_VALUE;
	
	/**
	 * Enables/disables dynamic re-compilation of lops/instructions.
	 * If enabled, we recompile each program block that contains at least
	 * one hop that requires re-compilation (e.g., unknown statistics 
	 * during compilation, or program blocks in functions).  
	 */
	public static boolean ALLOW_DYN_RECOMPILATION = true;
	public static boolean ALLOW_PARALLEL_DYN_RECOMPILATION = ALLOW_DYN_RECOMPILATION && true;
	
	/**
	 * Enables/disables to put operations with data-dependent output
	 * size into individual statement blocks / program blocks.
	 * Since recompilation is done on the granularity of program blocks
	 * this enables recompilation of subsequent operations according
	 * to the actual output size. This rewrite might limit the opportunity
	 * for piggybacking and therefore should only be applied if 
	 * dyanmic recompilation is enabled as well.
	 */
	public static boolean ALLOW_INDIVIDUAL_SB_SPECIFIC_OPS = ALLOW_DYN_RECOMPILATION && true;

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
	
	/**
	 * 
	 */
	public static boolean ALLOW_ALGEBRAIC_SIMPLIFICATION = true; 
	
	/**
	 * Enables if-else branch removal for constant predicates (original literals or 
	 * results of constant folding). 
	 * 
	 */
	public static boolean ALLOW_BRANCH_REMOVAL = true;
	
	/**
	 * 
	 */
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

	/**
	 * 
	 */
	public static boolean ALLOW_RAND_JOB_RECOMPILE = true;
	
	/**
	 * Enables CP-side data transformation for small files.
	 */
	public static boolean ALLOW_TRANSFORM_RECOMPILE = true;

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
	 * Enables parallel read/write of all text formats (textcell, csv, mm)
	 * and binary formats (binary block). 
	 * 
	 */
	public static boolean PARALLEL_CP_READ_TEXTFORMATS = true;
	public static boolean PARALLEL_CP_WRITE_TEXTFORMATS = true;
	public static boolean PARALLEL_CP_READ_BINARYFORMATS = true;
	public static boolean PARALLEL_CP_WRITE_BINARYFORMATS = true;
	
	
	
	/**
	 * Specifies a multiplier computing the degree of parallelism of parallel
	 * text read/write out of the available degree of parallelism. Set it to 1.0
	 * to get a number of threads equal the number of virtual cores.
	 * 
	 */
	public static final double PARALLEL_CP_READ_PARALLELISM_MULTIPLIER = 1.0;
	public static final double PARALLEL_CP_WRITE_PARALLELISM_MULTIPLIER = 1.0;

	/**
	 * Enables multi-threaded matrix multiply for mm, mmchain, and tsmm.
	 * 
	 */
	public static boolean PARALLEL_CP_MATRIX_MULTIPLY = true;
	
	/**
	 * Enables the use of CombineSequenceFileInputFormat with splitsize = 2x hdfs blocksize, 
	 * if sort buffer size large enough and parallelism not hurt. This solves to issues: 
	 * (1) it combines small files (depending on producers), and (2) it reduces task
	 * latency of large jobs with many tasks by factor 2.
	 * 
	 */
	public static final boolean ALLOW_COMBINE_FILE_INPUT_FORMAT = true;
	
	
	//////////////////////
	// Optimizer levels //
	//////////////////////

	private static OptimizationLevel _optLevel = OptimizationLevel.O2_LOCAL_MEMORY_DEFAULT;
	
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
		return _optLevel;
	}
	
	public static boolean isMemoryBasedOptLevel() {
		return (_optLevel != OptimizationLevel.O0_LOCAL_STATIC);
	}
	
	public static boolean isOptLevel( OptimizationLevel level ){
		return (_optLevel == level);
	}
	
	/**
	 * 
	 * @param optlevel
	 * @throws DMLRuntimeException
	 */
	public static void setOptimizationLevel( int optlevel ) 
		throws DMLRuntimeException
	{
		if( optlevel < 0 || optlevel > 5 )
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
				_optLevel = OptimizationLevel.O0_LOCAL_STATIC;
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
				ALLOW_AUTO_VECTORIZATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_SUM_PRODUCT_REWRITES = false;
				break;
			// opt level 1: memory-based (no advanced rewrites)	
			case 1:
				_optLevel = OptimizationLevel.O1_LOCAL_MEMORY_MIN;
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
				ALLOW_AUTO_VECTORIZATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_SUM_PRODUCT_REWRITES = false;
				break;
			// opt level 2: memory-based (all advanced rewrites)
			case 2:
				_optLevel = OptimizationLevel.O2_LOCAL_MEMORY_DEFAULT;
				break;
			// opt level 3: resource optimization, time- and memory-based (2 w/ resource optimizat)
			case 3:
				_optLevel = OptimizationLevel.O3_LOCAL_RESOURCE_TIME_MEMORY;
			break;
							
			// opt level 3: global, time- and memory-based (all advanced rewrites)
			case 4:
				_optLevel = OptimizationLevel.O4_GLOBAL_TIME_MEMORY;
				break;
			// opt level 4: debug mode (no interfering rewrites)
			case 5:				
				_optLevel = OptimizationLevel.O5_DEBUG_MODE;
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_ALGEBRAIC_SIMPLIFICATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_DYN_RECOMPILATION = false;
				ALLOW_SIZE_EXPRESSION_EVALUATION = false;
				ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = false;
				ALLOW_RAND_JOB_RECOMPILE = false;
				ALLOW_SUM_PRODUCT_REWRITES = false;
				ALLOW_SPLIT_HOP_DAGS = false;
				break;
		}
		setDefaultSize();
		
		//handle parallel text io (incl awareness of thread contention in <jdk8)
		if (!ConfigurationManager.getConfig().getBooleanValue(DMLConfig.CP_PARALLEL_TEXTIO)) {
			PARALLEL_CP_READ_TEXTFORMATS = false;
			PARALLEL_CP_WRITE_TEXTFORMATS = false;
			PARALLEL_CP_READ_BINARYFORMATS = false;
			PARALLEL_CP_WRITE_BINARYFORMATS = false;
		}
		else if(   InfrastructureAnalyzer.isJavaVersionLessThanJDK8() 
			    && InfrastructureAnalyzer.getLocalParallelism() > 1   )
		{
			LOG.warn("Auto-disable multi-threaded text read for 'text' and 'csv' due to thread contention on JRE < 1.8"
					+ " (java.version="+ System.getProperty("java.version")+").");
			
			//disable parallel text read
			PARALLEL_CP_READ_TEXTFORMATS = false;
		}

		//handle parallel matrix mult / rand configuration
		if (!ConfigurationManager.getConfig().getBooleanValue(DMLConfig.CP_PARALLEL_MATRIXMULT)) {
			PARALLEL_CP_MATRIX_MULTIPLY = false;
		}	
	}
	
	/**
	 * 
	 */
	public static void setDefaultSize() 
	{
		//we need to set default_size larger than any execution context
		//memory budget, however, it should not produce overflows on sum
		DEFAULT_SIZE = Math.max( InfrastructureAnalyzer.getLocalMaxMemory(),
				                 Math.max(InfrastructureAnalyzer.getRemoteMaxMemoryMap(),
				                		  InfrastructureAnalyzer.getRemoteMaxMemoryReduce()));
	}
	
	/**
	 * Returns memory budget (according to util factor) in bytes
	 * 
	 * @param localOnly specifies if only budget of current JVM or also MR JVMs 
	 * @return
	 */
	public static double getLocalMemBudget()
	{
		double ret = InfrastructureAnalyzer.getLocalMaxMemory();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	/**
	 * 
	 * @return
	 */
	public static double getRemoteMemBudgetMap()
	{
		return getRemoteMemBudgetMap(false);
	}
	
	
	/**
	 * 
	 * @return
	 */
	public static double getRemoteMemBudgetMap(boolean substractSortBuffer)
	{
		double ret = InfrastructureAnalyzer.getRemoteMaxMemoryMap();
		if( substractSortBuffer )
			ret -= InfrastructureAnalyzer.getRemoteMaxMemorySortBuffer();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	/**
	 * 
	 * @return
	 */
	public static double getRemoteMemBudgetReduce()
	{
		double ret = InfrastructureAnalyzer.getRemoteMaxMemoryReduce();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}

	/**
	 * 
	 * @param size
	 * @return
	 */
	public static boolean checkSparkBroadcastMemoryBudget( double size )
	{
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();

		//basic requirement: the broadcast needs to to fit once in the remote broadcast memory 
		//and twice into the local memory budget because we have to create a partitioned broadcast
		//memory and hand it over to the spark context as in-memory object
		return ( size < memBudgetExec && 2*size < memBudgetLocal );
	}
	
	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param nnz
	 * @return
	 */
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
	
	/**
	 * 
	 * @param rlen
	 * @param clen
	 * @param brlen
	 * @param bclen
	 * @param nnz
	 * @return
	 */
	public static boolean checkSparkCollectMemoryBudget( long rlen, long clen, int brlen, int bclen, long nnz, long memPinned )
	{
		//compute size of output matrix and its blocked representation
		double sp = getSparsity(rlen, clen, nnz);
		double memMatrix = estimateSizeExactSparsity(rlen, clen, sp);
		double memPMatrix = estimatePartitionedSizeExactSparsity(rlen, clen, brlen, bclen, sp);
		
		//check if both output matrix and partitioned matrix fit into local mem budget
		return (memPinned + memMatrix + memPMatrix < getLocalMemBudget());
	}
	
	/**
	 * Returns the number of reducers that potentially run in parallel.
	 * This is either just the configured value (SystemML config) or
	 * the minimum of configured value and available reduce slots. 
	 * 
	 * @param configOnly
	 * @return
	 */
	public static int getNumReducers( boolean configOnly )
	{
		int ret = ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS);
		if( !configOnly ) {
			ret = Math.min(ret,InfrastructureAnalyzer.getRemoteParallelReduceTasks());
			
			//correction max number of reducers on yarn clusters
			if( InfrastructureAnalyzer.isYarnEnabled() )
				ret = (int)Math.max( ret, YarnClusterAnalyzer.getNumCores()/2 );
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public static int getNumMappers()
	{
		int ret = InfrastructureAnalyzer.getRemoteParallelMapTasks();
			
		//correction max number of reducers on yarn clusters
		if( InfrastructureAnalyzer.isYarnEnabled() )
			ret = (int)Math.max( ret, YarnClusterAnalyzer.getNumCores() );
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public static boolean isSparkExecutionMode() {
		return (   DMLScript.rtplatform == RUNTIME_PLATFORM.SPARK
				|| DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK);
	}
	
	/**
	 * 
	 * @return
	 */
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
	 * @return
	 */
	public static int getParallelTextReadParallelism()
	{
		if( !PARALLEL_CP_READ_TEXTFORMATS )
			return 1; // sequential execution
			
		//compute degree of parallelism for parallel text read
		double dop = InfrastructureAnalyzer.getLocalParallelism()
				     * PARALLEL_CP_READ_PARALLELISM_MULTIPLIER;
		return (int) Math.round(dop);
	}
	
	/**
	 * 
	 * @return
	 */
	public static int getParallelBinaryReadParallelism()
	{
		if( !PARALLEL_CP_READ_BINARYFORMATS )
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
	 * @return
	 */
	public static int getParallelTextWriteParallelism()
	{
		if( !PARALLEL_CP_WRITE_TEXTFORMATS )
			return 1; // sequential execution

		//compute degree of parallelism for parallel text read
		double dop = InfrastructureAnalyzer.getLocalParallelism()
				     * PARALLEL_CP_WRITE_PARALLELISM_MULTIPLIER;
		return (int) Math.round(dop);
	}

	/**
	 * 
	 * @return
	 */
	public static int getParallelBinaryWriteParallelism()
	{
		if( !PARALLEL_CP_WRITE_BINARYFORMATS )
			return 1; // sequential execution

		//compute degree of parallelism for parallel text read
		double dop = InfrastructureAnalyzer.getLocalParallelism()
				     * PARALLEL_CP_WRITE_PARALLELISM_MULTIPLIER;
		return (int) Math.round(dop);
	}
	
	////////////////////////
	// Memory Estimates   //
	////////////////////////
	
	/**
	 * 
	 * @param mc
	 * @return
	 */
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
	 * @param nrows
	 * @param ncols
	 * @param sp
	 * @return
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
	 * @param nrows
	 * @param ncols
	 * @param sp
	 * @return
	 */
	public static long estimateSizeExactSparsity(long nrows, long ncols, double sp) 
	{
		return MatrixBlock.estimateSizeInMemory(nrows,ncols,sp);
	}
	
	/**
	 * Estimates the footprint (in bytes) for a partitioned in-memory representation of a
	 * matrix with the given matrix characteristics
	 * 
	 * @param mc
	 * @return
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
	 * @param nrows
	 * @param ncols
	 * @param sp
	 * @return
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
	 * @param nrows
	 * @param ncols
	 * @param sp
	 * @return
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
	 * @param nrows
	 * @param ncols
	 * @param sp
	 * @return
	 */
	public static long estimateSize(long nrows, long ncols) 
	{
		return estimateSizeExactSparsity(nrows, ncols, 1.0);
	}
	
	/**
	 * 
	 * @param nrows
	 * @param ncols
	 * @return
	 */
	public static long estimateSizeEmptyBlock(long nrows, long ncols)
	{
		return estimateSizeExactSparsity(0, 0, 0.0d);
	}
	
	/**
	 * Estimates the memory footprint of a SparseRow with <code>clen</code>
	 * columns and <code>sp</code> sparsity. This method accounts for the
	 * overhead incurred by extra cells allocated (but not used) for SparseRow.
	 * It assumes that non-zeros are uniformly distributed in the matrix --
	 * i.e., #estimated nnz in a given SparseRow = clen*sp.
	 * 
	 * @param clen
	 * @param sp
	 * @return estimated size in bytes
	 */
	public static long estimateRowSize(long clen, double sp) 
	{	
		if ( sp == 0 )
			return 0;
		
		int basicSize = 28;
		int cellSize = 12; // every cell takes 12 (8+4) bytes
		if ( sp == 1 ) {
			return clen * cellSize; 
		}
		long  numCells = SparseRow.initialCapacity;
		if ( (long) (sp*clen) > numCells ) {
			numCells = (long) (sp*clen);
		}
		long allocatedCells = (long)Math.pow(2, Math.ceil(Math.log(numCells)/Math.log(2)) );
		long rowSize = basicSize +  allocatedCells * cellSize;
		return rowSize;
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
	 * Returns false if dimensions known to be invalid; other true
	 * 
	 * @param rows
	 * @param cols
	 * @return
	 */
	public static boolean isValidCPDimensions( long rows, long cols )
	{
		//the current CP runtime implementation requires that rows and cols
		//are integers since we use a single matrixblock to represent the
		//entire matrix
		return (rows <= Integer.MAX_VALUE && cols<=Integer.MAX_VALUE);
	}
	
	/**
	 * Determines if valid matrix size to be represented in CP data structures. Note that
	 * sparsity needs to be specified as rows*cols if unknown. 
	 * 
	 * @param rows
	 * @param cols
	 * @param sparsity
	 * @return
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
			//check max nnz
			ret = (nnz <= Long.MAX_VALUE);
		}
		else //DENSE
		{
			//check number of matrix cell
			ret = ((rows * cols) <= MAX_NUMCELLS_CP_DENSE);
		}
			
		return ret;
	}
	

	/**
	 * 
	 * @return
	 * @throws HopsException 
	 */
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
	
	/**
	 * 
	 * @return
	 */
	public static int getConstrainedNumThreads(int maxNumThreads)
	{
		//by default max local parallelism (vcores) 
		int ret = InfrastructureAnalyzer.getLocalParallelism();
		
		//apply external max constraint (e.g., set by parfor or other rewrites)
		if( maxNumThreads > 0 ) {
			ret = Math.min(ret, maxNumThreads);
		}
		
		//apply global multi-threading constraint
		if( !PARALLEL_CP_MATRIX_MULTIPLY ) {
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
	 * @param sp1 -- sparsity of A
	 * @param sp2 -- sparsity of B
	 * @param m -- nrow(A)
	 * @param k -- ncol(A), nrow(B)
	 * @param n -- ncol(B)
	 * @return
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
	
	/**
	 * 
	 * @param rlen1
	 * @param clen1
	 * @param nnz1
	 * @param rlen2
	 * @param clen2
	 * @param nnz2
	 * @return
	 */
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
	 * @param op
	 * @return
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
	 * @param op
	 * @param lit
	 * @return
	 */
	public static boolean isBinaryOpConditionalSparseSafeExact( OpOp2 op, LiteralOp lit )
	{
		double val = HopRewriteUtils.getDoubleValueSafe(lit);
		
		return ( op==OpOp2.NOTEQUAL && val==0);
	}
	
	/**
	 * 
	 * @param sp1
	 * @param op
	 * @param lit
	 * @return
	 */
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
	 * @param sp1 -- sparsity of A
	 * @param sp2 -- sparsity of B
	 * @param op -- binary operation
	 * @return
	 * 
	 * NOTE: append has specific computation
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
	 * @param root
	 * @return
	 * @throws HopsException 
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
	
	/**
	 * 
	 * @param root
	 * @param valMemo
	 * @param vars
	 * @return
	 * @throws HopsException 
	 */
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
	
	/**
	 * 
	 * @param root
	 * @param valMemo
	 * @return
	 * @throws HopsException
	 */
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
	
	/**
	 * 
	 * @param root
	 * @param valMemo
	 * @param vars
	 * @return
	 * @throws HopsException
	 */
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
	
	
	/**
	 * 
	 * @param root
	 * @param valMemo
	 * @return
	 * @throws HopsException
	 */
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
	
	/**
	 * 
	 * @param root
	 * @param valMemo
	 * @param vars
	 * @return
	 * @throws HopsException
	 */
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
	
	/**
	 * 
	 * @param root
	 * @param valMemo
	 * @return
	 * @throws HopsException
	 */
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
				default: ret = Double.MAX_VALUE;
			}
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}
	
	/**
	 * 
	 * @param root
	 * @param valMemo
	 * @param vars
	 * @return
	 * @throws HopsException
	 */
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
				default: ret = Double.MAX_VALUE;
			}
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}
	
		
}

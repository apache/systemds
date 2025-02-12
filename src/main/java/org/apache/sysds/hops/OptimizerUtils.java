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

package org.apache.sysds.hops;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.fedplanner.FTypes.FederatedPlanner;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.lops.Checkpoint;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.apache.sysds.runtime.controlprogram.caching.UnifiedMemoryManager;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlock.Type;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.fed.FEDInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;
import org.apache.sysds.utils.MemoryEstimates;

public class OptimizerUtils 
{
	////////////////////////////////////////////////////////
	// Optimizer constants and flags (incl tuning knobs)  //
	////////////////////////////////////////////////////////
	/**
	 * Utilization factor used in deciding whether an operation to be scheduled on CP or MR. 
	 * NOTE: it is important that MEM_UTIL_FACTOR+CacheableData.CACHING_BUFFER_SIZE &lt; 1.0
	 */
	public static double MEM_UTIL_FACTOR = 0.7d;
	/** Default buffer pool sizes for static (15%) and unified (85%) memory */
	public static double DEFAULT_MEM_UTIL_FACTOR = 0.15d;
	public static double DEFAULT_UMM_UTIL_FACTOR = 0.85d;

	/** Memory managers (static partitioned, unified) */
	public enum MemoryManager {
		STATIC_MEMORY_MANAGER,
		UNIFIED_MEMORY_MANAGER
	}

	/** Indicate the current memory manager in effect */
	public static MemoryManager MEMORY_MANAGER = null;
	/** Buffer pool size in bytes */
	public static long BUFFER_POOL_SIZE = 0;

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

	public static final long SAFE_REP_CHANGE_THRES = 8 * 1024 *1024; //8MB
	
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
	
	/**
	 * Enables the removal of (par)for-loops when from, to, and increment are constants
	 * (original literals or results of constant folding) and lead to an empty sequence,
	 * i.e., (par)for-loops without a single iteration.
	 */
	public static boolean ALLOW_FOR_LOOP_REMOVAL = true;

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
	 * Number of inter-procedural analysis (IPA) repetitions. If set to {@literal >=2}, we apply
	 * IPA multiple times in order to allow scalar propagation over complex function call
	 * graphs and various interactions between constant propagation, constant folding,
	 * and other rewrites such as branch removal and the merge of statement block sequences.
	 */
	public static int IPA_NUM_REPETITIONS = 5;

	/**
	 * Enables sum product rewrites such as mapmultchains. In the future, this will cover 
	 * all sum-product related rewrites.
	 */
	public static boolean ALLOW_SUM_PRODUCT_REWRITES = true;
	public static boolean ALLOW_SUM_PRODUCT_REWRITES2 = true;

	/**
	 * Enables additional mmchain optimizations. in the future, this might be merged with
	 * ALLOW_SUM_PRODUCT_REWRITES.
	 */
	public static boolean ALLOW_ADVANCED_MMCHAIN_REWRITES = false;
	
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
	 * Enables the update-in-place for all unary operators with a single
	 * consumer. In this case we do not allocate the output, but directly
	 * write the output values back to the input block.
	 */
	//TODO enabling it by default requires modifications in lineage-based reuse
	public static boolean ALLOW_UNARY_UPDATE_IN_PLACE = false;

	/**
	 * Enables update-in-place for binary operators if the first input
	 * has no consumers. In this case we directly write the output
	 * values back to the first input block.
	 */
	public static boolean ALLOW_BINARY_UPDATE_IN_PLACE = false;

	/**
	 * Replace eval second-order function calls with normal function call
	 * if the function name is a known string (after constant propagation).
	 */
	public static boolean ALLOW_EVAL_FCALL_REPLACEMENT = true;
	
	
	/**
	 * Enables a specific rewrite for code motion, i.e., hoisting loop invariant code
	 * out of while, for, and parfor loops.
	 */
	public static boolean ALLOW_CODE_MOTION = false;

	/**
	 * Compile federated instructions based on input federation state and privacy constraints.
	 */
	public static boolean FEDERATED_COMPILATION = false;
	public static Map<Integer, FEDInstruction.FederatedOutput> FEDERATED_SPECS = new HashMap<>();
	
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

	/**
	 * This variable allows for use of explicit local command, that forces a spark block to be executed and returned as a local block.
	 */
	public static boolean ALLOW_SCRIPT_LEVEL_LOCAL_COMMAND = false;

	/**
	 * This variable allows for insertion of Compress and decompress in the dml script from the user.
	 * This is added because we want to have a way to test, and verify the correct placement of compress and decompress commands.
	 */
	public static boolean ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND = true;

	/**
	 * This variable allows for insertion of Quantize and compress in the dml script from the user.
	 */
	public static boolean ALLOW_SCRIPT_LEVEL_QUANTIZE_COMPRESS_COMMAND = true;

	/**
	 * Boolean specifying if quantization-fused compression rewrite is allowed.
	 */
	public static boolean ALLOW_QUANTIZE_COMPRESS_REWRITE = true;

	/**
	 * Boolean specifying if compression rewrites is allowed. This is disabled at run time if the IPA for Workload aware compression
	 * is activated.
	 */
	public static boolean ALLOW_COMPRESSION_REWRITE = true;
	
	/**
	 * Enable transitive spark execution type selection. This refines the exec-type selection logic of unary aggregates 
	 * by pushing * the unary aggregates, whose inputs are created by spark instructions, to spark execution type as well.
	 */
	public static boolean ALLOW_TRANSITIVE_SPARK_EXEC_TYPE = true;

	/**
	 * Enable prefetch and broadcast. Prefetch asynchronously calls acquireReadAndRelease() to trigger remote
	 * operations, which would otherwise make the next instruction wait till completion. Broadcast allows
	 * asynchronously transferring the data to all the nodes.
	 */
	public static boolean ASYNC_PREFETCH = false; //both Spark and GPU
	public static boolean ASYNC_BROADCAST_SPARK = false;
	public static boolean ASYNC_CHECKPOINT_SPARK = false;

	/**
	 * Heuristic-based instruction ordering to maximize inter-operator PARALLELISM.
	 * Place the Spark operator chains first and trigger them to execute in parallel.
	 */
	public static boolean MAX_PARALLELIZE_ORDER = false;

	/**
	 * Cost-based instruction ordering to minimize total execution time under
	 * the constraint of available memory.
	 */
	public static boolean COST_BASED_ORDERING = false;

	/**
	 * Rule-based operator placement policy for GPU.
	 */
	public static boolean RULE_BASED_GPU_EXEC = false;

	/**
	 * Automatic placement of GPU lineage cache eviction
	 */

	public static boolean AUTO_GPU_CACHE_EVICTION = true;

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
	 *  level of SystemDS.
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
	}
		
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
	
	public static CompilerConfig constructCompilerConfig( DMLConfig dmlconf ) {
		return constructCompilerConfig(new CompilerConfig(), dmlconf);
	}
	
	public static CompilerConfig constructCompilerConfig( CompilerConfig cconf, DMLConfig dmlconf ) 
	{
		//each script sets its own block size, opt level etc
		cconf.set(ConfigType.BLOCK_SIZE, dmlconf.getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ));

		//handle optimization level
		int optlevel = dmlconf.getIntValue(DMLConfig.OPTIMIZATION_LEVEL);
		if( optlevel < 0 || optlevel > 7 )
			throw new DMLRuntimeException("Error: invalid optimization level '"+optlevel+"' (valid values: 0-5).");
	
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
				IPA_NUM_REPETITIONS = 1;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_FOR_LOOP_REMOVAL = false;
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
				IPA_NUM_REPETITIONS = 1;
				ALLOW_BRANCH_REMOVAL = false;
				ALLOW_FOR_LOOP_REMOVAL = false;
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
		if (!dmlconf.getBooleanValue(DMLConfig.CP_PARALLEL_IO)) {
			cconf.set(ConfigType.PARALLEL_CP_READ_TEXTFORMATS, false);
			cconf.set(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS, false);
			cconf.set(ConfigType.PARALLEL_CP_READ_BINARYFORMATS, false);
			cconf.set(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS, false);
		}

		//handle parallel matrix mult / rand configuration
		if (!dmlconf.getBooleanValue(DMLConfig.CP_PARALLEL_OPS)) {
			cconf.set(ConfigType.PARALLEL_CP_MATRIX_OPERATIONS, false);
		}
		
		//handle federated runtime conversion to avoid string comparisons
		String planner = dmlconf.getTextValue(DMLConfig.FEDERATED_PLANNER);
		if( FederatedPlanner.RUNTIME.name().equalsIgnoreCase(planner) ) {
			cconf.set(ConfigType.FEDERATED_RUNTIME, true);
		}
		
		return cconf;
	}
	
	public static void resetStaticCompilerFlags() {
		//TODO this is a workaround for MLContext to avoid a major refactoring before the release; this method 
		//should be removed as soon all modified static variables are properly handled in the compiler config
		ALLOW_ALGEBRAIC_SIMPLIFICATION = true;
		ALLOW_AUTO_VECTORIZATION = true;
		ALLOW_BRANCH_REMOVAL = true;
		ALLOW_FOR_LOOP_REMOVAL = true;
		ALLOW_CONSTANT_FOLDING = true;
		ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = true;
		ALLOW_INTER_PROCEDURAL_ANALYSIS = true;
		ALLOW_LOOP_UPDATE_IN_PLACE = true;
		ALLOW_OPERATOR_FUSION = true;
		ALLOW_RAND_JOB_RECOMPILE = true;
		ALLOW_SIZE_EXPRESSION_EVALUATION = true;
		ALLOW_SPLIT_HOP_DAGS = true;
		ALLOW_SUM_PRODUCT_REWRITES = true;
		ALLOW_WORSTCASE_SIZE_EXPRESSION_EVALUATION = true;
		IPA_NUM_REPETITIONS = 3;
	}

	public static long getDefaultSize() {
		//we need to set default_size larger than any execution context
		//memory budget, however, it should not produce overflows on sum
		return InfrastructureAnalyzer.getLocalMaxMemory();
	}
	
	public static void resetDefaultSize() {
		DEFAULT_SIZE = getDefaultSize();
	}
	
	
	public static int getDefaultFrameSize() {
		return DEFAULT_FRAME_BLOCKSIZE;
	}
	
	/**
	 * Returns memory budget (according to util factor) in bytes
	 * 
	 * @return local memory budget
	 */
	public static double getLocalMemBudget() {
		double ret = InfrastructureAnalyzer.getLocalMaxMemory();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}

	/**
	 * Returns buffer pool size as set in the config
	 *
	 * @return buffer pool size in bytes
	 */
	public static long getBufferPoolLimit() {
		if (BUFFER_POOL_SIZE != 0)
			return BUFFER_POOL_SIZE;
		DMLConfig conf = ConfigurationManager.getDMLConfig();
		double bufferPoolFactor = (double)(conf.getIntValue(DMLConfig.BUFFERPOOL_LIMIT))/100;
		bufferPoolFactor = Math.max(bufferPoolFactor, DEFAULT_MEM_UTIL_FACTOR);
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		return (long)(bufferPoolFactor * maxMem);
	}

	/**
	 * Check if unified memory manager is in effect
	 * @return boolean
	 */
	public static boolean isUMMEnabled() {
		if (MEMORY_MANAGER == null) {
			DMLConfig conf = ConfigurationManager.getDMLConfig();
			boolean isUMM = conf.getTextValue(DMLConfig.MEMORY_MANAGER).equalsIgnoreCase("unified");
			MEMORY_MANAGER = isUMM ? MemoryManager.UNIFIED_MEMORY_MANAGER : MemoryManager.STATIC_MEMORY_MANAGER;
		}
		return MEMORY_MANAGER == MemoryManager.UNIFIED_MEMORY_MANAGER;
	}

	/**
	 * Disable unified memory manager and fallback to static partitioning.
	 * Initialize LazyWriteBuffer with the default size (15%).
	 */
	public static void disableUMM() {
		MEMORY_MANAGER = MemoryManager.STATIC_MEMORY_MANAGER;
		LazyWriteBuffer.cleanup();
		LazyWriteBuffer.init();
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		BUFFER_POOL_SIZE = (long) (DEFAULT_MEM_UTIL_FACTOR * maxMem);
		LazyWriteBuffer.setWriteBufferLimit(BUFFER_POOL_SIZE);
	}

	/**
	 * Enable unified memory manager and initialize with the default size (85%).
	 */
	public static void enableUMM() {
		MEMORY_MANAGER = MemoryManager.UNIFIED_MEMORY_MANAGER;
		UnifiedMemoryManager.cleanup();
		UnifiedMemoryManager.init();
		long maxMem = InfrastructureAnalyzer.getLocalMaxMemory();
		BUFFER_POOL_SIZE = (long) (DEFAULT_UMM_UTIL_FACTOR * maxMem);
		UnifiedMemoryManager.setUMMLimit(BUFFER_POOL_SIZE);
	}
	
	public static boolean isMaxLocalParallelism(int k) {
		return InfrastructureAnalyzer.getLocalParallelism() == k;
	}

	public static boolean isTopLevelParFor() {
		//since every local parfor with degree of parallelism k>1 changes the
		//local memory budget, we can simply probe the current memory fraction
		return InfrastructureAnalyzer.getLocalMaxMemoryFraction() >= 0.99;
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

	public static boolean checkSparkBroadcastMemoryBudget( long rlen, long clen, long blen, long nnz ) {
		double memBudgetExec = SparkExecutionContext.getBroadcastMemoryBudget();
		double memBudgetLocal = OptimizerUtils.getLocalMemBudget();
		double sp = getSparsity(rlen, clen, nnz);
		double size = estimateSizeExactSparsity(rlen, clen, sp);
		double sizeP = estimatePartitionedSizeExactSparsity(rlen, clen, blen, sp);
		//basic requirement: the broadcast needs to to fit once in the remote broadcast memory 
		//and twice into the local memory budget because we have to create a partitioned broadcast
		//memory and hand it over to the spark context as in-memory object
		return (   OptimizerUtils.isValidCPDimensions(rlen, clen)
				&& sizeP < memBudgetExec && size+sizeP < memBudgetLocal );
	}

	public static boolean checkSparkCollectMemoryBudget(DataCharacteristics dc, long memPinned ) {
		if (dc instanceof MatrixCharacteristics) {
			return checkSparkCollectMemoryBudget(dc.getRows(), dc.getCols(),
				dc.getBlocksize(), dc.getNonZerosBound(), memPinned, false);
		} else {
			long[] dims = dc.getDims();
			return checkSparkCollectMemoryBudget(dims, dc.getNonZeros(), memPinned, false);
		}
	}
	
	public static boolean checkSparkCollectMemoryBudget(DataCharacteristics dc, long memPinned, boolean checkBP ) {
		if (dc instanceof MatrixCharacteristics) {
			return checkSparkCollectMemoryBudget(dc.getRows(), dc.getCols(),
				dc.getBlocksize(), dc.getNonZerosBound(), memPinned, checkBP);
		} else {
			long[] dims = dc.getDims();
			return checkSparkCollectMemoryBudget(dims, dc.getNonZeros(), memPinned, checkBP);
		}
	}
	
	private static boolean checkSparkCollectMemoryBudget( long rlen, long clen, int blen, long nnz, long memPinned, boolean checkBP ) {
		//compute size of output matrix and its blocked representation
		double sp = getSparsity(rlen, clen, nnz);
		double memMatrix = estimateSizeExactSparsity(rlen, clen, sp);
		double memPMatrix = estimatePartitionedSizeExactSparsity(rlen, clen, blen, sp);
		//check if both output matrix and partitioned matrix fit into local mem budget
		return (memPinned + memMatrix + memPMatrix < getLocalMemBudget())
		//check if the output matrix fits into the write buffer to avoid unnecessary evictions
			&& (!checkBP || memMatrix < LazyWriteBuffer.getWriteBufferLimit());
	}

	private static boolean checkSparkCollectMemoryBudget( long[] dims, long nnz, long memPinned, boolean checkBP ) {
		//compute size of output matrix and its blocked representation
		//double sp = getSparsity(dims, nnz);
		// TODO estimate exact size
		long doubleSize = UtilFunctions.prod(dims) * 8;
		double memTensor = doubleSize;
		double memPTensor = doubleSize;
		//check if both output matrix and partitioned matrix fit into local mem budget
		return (memPinned + memTensor + memPTensor < getLocalMemBudget())
				//check if the output matrix fits into the write buffer to avoid unnecessary evictions
				&& (!checkBP || memTensor < LazyWriteBuffer.getWriteBufferLimit());
	}

	public static boolean checkSparseBlockCSRConversion( DataCharacteristics dcIn ) {
		//we use the non-zero bound to make the important csr decision in 
		//an best effort manner (the precise non-zeros is irrelevant here)
		double sp = OptimizerUtils.getSparsity(
			dcIn.getRows(), dcIn.getCols(), dcIn.getNonZerosBound());
		return Checkpoint.CHECKPOINT_SPARSE_CSR 
			&& sp < MatrixBlock.SPARSITY_TURN_POINT;
	}
	
	/**
	 * Returns the number of tasks that potentially run in parallel.
	 * This is either just the configured value (SystemDS config) or
	 * the minimum of configured value and available task slots.
	 *
	 * @return number of tasks
	 */
	public static int getNumTasks() {
		if( isSparkExecutionMode() )
			return SparkExecutionContext.getDefaultParallelism(false);
		return InfrastructureAnalyzer.getLocalParallelism();
	}

	public static ExecMode getDefaultExecutionMode() {
		//default execution type is hybrid (cp+mr)
		ExecMode ret = ExecMode.HYBRID;
		
		//switch default to HYBRID (cp+spark) if in spark driver
		String sparkenv = System.getenv().get("SPARK_ENV_LOADED");
		if( sparkenv != null && sparkenv.equals("1") )
			ret = ExecMode.HYBRID;
		
		return ret;
	}

	public static boolean isSparkExecutionMode() {
		return DMLScript.getGlobalExecMode() == ExecMode.SPARK
			|| DMLScript.getGlobalExecMode() == ExecMode.HYBRID;
	}
	
	public static boolean isHybridExecutionMode() {
		return DMLScript.getGlobalExecMode() == ExecMode.HYBRID;
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
	
	public static long estimateSize(DataCharacteristics dc) {
		return estimateSizeExactSparsity(dc);
	}

	public static long estimateSizeExactSparsity(DataCharacteristics dc)
	{
		return estimateSizeExactSparsity(
				dc.getRows(),
				dc.getCols(),
				dc.getNonZeros());
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


	public static long estimateSizeExactFrame(long nRows, long nCols){
		// Currently we do not support frames larger than INT. 
		// Therefore, we estimate their size to be extremely large.
		// The large size force spark operations.
		if(nRows > Integer.MAX_VALUE)
			return Long.MAX_VALUE;
		
		// assuming String arrays and on average 8 characters per value.
		return (long)MemoryEstimates.stringArrayCost((int)nRows, 8) * nCols;
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
	 * @param dc matrix characteristics
	 * @return memory estimate
	 */
	public static long estimatePartitionedSizeExactSparsity(DataCharacteristics dc) {
		return estimatePartitionedSizeExactSparsity(dc, true);
	}
	
	public static long estimatePartitionedSizeExactSparsity(DataCharacteristics dc, boolean outputEmptyBlocks)
	{
		if (dc instanceof MatrixCharacteristics) {
			return estimatePartitionedSizeExactSparsity(
				dc.getRows(), dc.getCols(), dc.getBlocksize(),
				dc.getNonZerosBound(), outputEmptyBlocks);
		}
		else {
			// TODO estimate partitioned size exact for tensor
			long inaccurateSize = 8; // 8 for double
			for (int i = 0; i < dc.getNumDims(); i++) {
				inaccurateSize *= dc.getDim(i);
			}
			return inaccurateSize;
		}
	}
	
	/**
	 * Estimates the footprint (in bytes) for a partitioned in-memory representation of a
	 * matrix with dimensions=(nrows,ncols) and number of non-zeros nnz.
	 * 
	 * @param rlen number of rows
	 * @param clen number of cols
	 * @param blen rows/cols per block
	 * @param nnz number of non-zeros
	 * @return memory estimate
	 */
	public static long estimatePartitionedSizeExactSparsity(long rlen, long clen, long blen, long nnz)  {
		double sp = getSparsity(rlen, clen, nnz);
		return estimatePartitionedSizeExactSparsity(rlen, clen, blen, sp);
	}
	
	public static long estimatePartitionedSizeExactSparsity(long rlen, long clen, long blen, long nnz, boolean outputEmptyBlocks)  {
		double sp = getSparsity(rlen, clen, nnz);
		return estimatePartitionedSizeExactSparsity(rlen, clen, blen, sp, outputEmptyBlocks);
	}

	/**
	 * Estimates the footprint (in bytes) for a partitioned in-memory representation of a
	 * matrix with the hops dimensions and number of non-zeros nnz.
	 * 
	 * @param hop The hop to extract dimensions and nnz from
	 * @return the memory estimate
	 */
	public static long estimatePartitionedSizeExactSparsity(Hop hop){
		long rlen = hop.getDim1();
		long clen = hop.getDim2();
		int blen = hop.getBlocksize();
		long nnz = hop.getNnz();
		return  estimatePartitionedSizeExactSparsity(rlen, clen, blen, nnz);
	}
	
	/**
	 * Estimates the footprint (in bytes) for a partitioned in-memory representation of a
	 * matrix with dimensions=(nrows,ncols) and sparsity=sp.
	 * 
	 * @param rlen number of rows
	 * @param clen number of cols
	 * @param blen rows/cols per block
	 * @param sp sparsity
	 * @return memory estimate
	 */
	public static long estimatePartitionedSizeExactSparsity(long rlen, long clen, long blen, double sp) {
		return estimatePartitionedSizeExactSparsity(rlen, clen, blen, sp, true);
	}
	
	
	public static long estimatePartitionedSizeExactSparsity(long rlen, long clen, long blen, double sp, boolean outputEmptyBlocks)
	{
		long ret = 0;

		//check for guaranteed existence of empty blocks (less nnz than total number of blocks)
		long tnrblks = (long)Math.ceil((double)rlen/blen);
		long tncblks = (long)Math.ceil((double)clen/blen);
		long nnz = (long) Math.ceil(sp * rlen * clen);
		if( nnz <= tnrblks * tncblks ) {
			long lrlen = Math.min(rlen, blen);
			long lclen = Math.min(clen, blen);
			return nnz * MatrixBlock.estimateSizeSparseInMemory(lrlen, lclen, 1d/lrlen/lclen, Type.COO)
				 + (outputEmptyBlocks ? (tnrblks * tncblks - nnz) * estimateSizeEmptyBlock(lrlen, lclen) : 0);
		}
		
		//estimate size of full blen x blen blocks
		long nrblks = rlen / blen;
		long ncblks = clen / blen;
		if( nrblks * ncblks > 0 )
			ret += nrblks * ncblks * estimateSizeExactSparsity(blen, blen, sp);

		//estimate size of bottom boundary blocks 
		long lrlen = rlen % blen;
		if( ncblks > 0 && lrlen >= 0 )
			ret += ncblks * estimateSizeExactSparsity(lrlen, blen, sp);
		
		//estimate size of right boundary blocks
		long lclen = clen % blen;
		if( nrblks > 0 && lclen >= 0 )
			ret += nrblks * estimateSizeExactSparsity(blen, lclen, sp);
		
		//estimate size of bottom right boundary block
		if( lrlen >= 0 && lclen >= 0  )
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
	public static long estimateSize(long nrows, long ncols) {
		return estimateSizeExactSparsity(nrows, ncols, 1.0);
	}
	
	public static long estimateSizeEmptyBlock(long nrows, long ncols) {
		return estimateSizeExactSparsity(0, 0, 0.0d);
	}

	public static long estimateSizeTextOutput(long rows, long cols, long nnz, FileFormat fmt) {
		long bsize = MatrixBlock.estimateSizeOnDisk(rows, cols, nnz);
		if( fmt.isIJV() )
			return bsize * 3;
		else if( fmt == FileFormat.LIBSVM )
			return Math.round(bsize * 2.5);
		else if( fmt == FileFormat.CSV )
			return bsize * 2;
		return bsize;
	}
	
	public static long estimateSizeTextOutput(int[] dims, long nnz, FileFormat fmt) {
		// TODO accurate estimation
		if( fmt == FileFormat.TEXT )
			// nnz * (8 bytes for number + each dimension with an expected String length of 3 and one space)
			return nnz * (8 + dims.length * 4); // very simple estimation. example:100 100 1.345678
		throw new DMLRuntimeException("Tensor output format not implemented.");
	}
	
	public static double getTotalMemEstimate(Hop[] in, Hop out) {
		return getTotalMemEstimate(in, out, false);
	}
	
	public static double getTotalMemEstimate(Hop[] in, Hop out, boolean denseOut) {
		return Arrays.stream(in)
			.mapToDouble(h -> h.getOutputMemEstimate()).sum()
			+ (!denseOut ? out.getOutputMemEstimate() :
				OptimizerUtils.estimateSize(out.getDim1(), out.getDim2()));
	}
	
	/**
	 * Indicates if the given indexing range is block aligned, i.e., it does not require
	 * global aggregation of blocks.
	 * 
	 * @param ixrange indexing range
	 * @param mc matrix characteristics
	 * @return true if indexing range is block aligned
	 */
	public static boolean isIndexingRangeBlockAligned(IndexRange ixrange, DataCharacteristics mc) {
		long rl = ixrange.rowStart;
		long ru = ixrange.rowEnd;
		long cl = ixrange.colStart;
		long cu = ixrange.colEnd;
		long blen = mc.getBlocksize();
		return isIndexingRangeBlockAligned(rl, ru, cl, cu, blen);
	}
	
	/**
	 * Indicates if the given indexing range is block aligned, i.e., it does not require
	 * global aggregation of blocks.
	 * 
	 * @param rl rows lower
	 * @param ru rows upper
	 * @param cl cols lower
	 * @param cu cols upper
	 * @param blen rows/cols per block
	 * @return true if indexing range is block aligned
	 */
	public static boolean isIndexingRangeBlockAligned(long rl, long ru, long cl, long cu, long blen) {
		return rl != -1 && ru != -1 && cl != -1 && cu != -1
				&&((rl-1)%blen == 0 && (cl-1)%blen == 0 
				|| (rl-1)/blen == (ru-1)/blen && (cl-1)%blen == 0 
				|| (rl-1)%blen == 0 && (cl-1)/blen == (cu-1)/blen);
	}
	
	public static boolean isValidCPDimensions( DataCharacteristics mc ) {
		return isValidCPDimensions(mc.getRows(), mc.getCols());
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
		//NOTE: We heuristically cache matrices that are close to or larger
		//than the local memory budget. The different relative fractions 
		//according to number of columns is reflecting common operations
		//(e.g., two inputs/one output for binary vector operations)
		return !(dim2 > 1 && outMem < getLocalMemBudget()/2
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
			+ Lop.FILE_SEPARATOR + Lop.CP_ROOT_THREAD_ID + Lop.FILE_SEPARATOR 
			+ Dag.getNextUniqueFilenameSuffix();
	}

	public static boolean allowsToFilterEmptyBlockOutputs( Hop hop ) {
		boolean ret = true;
		for( Hop p : hop.getParent() ) {
			p.optFindExecType(); //ensure exec type evaluated
			ret &= ( p.getExecType()==ExecType.CP 
				||(p instanceof AggBinaryOp && allowsToFilterEmptyBlockOutputs(p) )
				||(HopRewriteUtils.isReorg(p, ReOrgOp.RESHAPE, ReOrgOp.TRANS) && allowsToFilterEmptyBlockOutputs(p) )
				||(HopRewriteUtils.isData(p, OpOpData.PERSISTENTWRITE) && ((DataOp)p).getFileFormat()==FileFormat.TEXT))
				&& !(p instanceof FunctionOp || (p instanceof DataOp && ((DataOp)p).getFileFormat()!=FileFormat.TEXT) ); //no function call or transient write
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

	public static int getTransformNumThreads()
	{
		//by default max local parallelism (vcores) 
		int ret = InfrastructureAnalyzer.getLocalParallelism();
		int maxNumThreads = ConfigurationManager.getNumThreads();
		
		//apply external max constraint (e.g., set by parfor or other rewrites)
		if( maxNumThreads > 0 ) {
			ret = Math.min(ret, maxNumThreads);
		}
		
		//check if enabled in config.xml
		if( !ConfigurationManager.isParallelTransform() ) {
			ret = 1;
		}
			
		return ret;
	}

	public static int getTokenizeNumThreads()
	{
		//by default max local parallelism (vcores)
		int ret = InfrastructureAnalyzer.getLocalParallelism();
		int maxNumThreads = ConfigurationManager.getNumThreads();

		//apply external max constraint (e.g., set by parfor or other rewrites)
		if( maxNumThreads > 0 ) {
			ret = Math.min(ret, maxNumThreads);
		}

		//check if enabled in config.xml
		if( !ConfigurationManager.isParallelTokenize() ) {
			ret = 1;
		}

		return ret;
	}

	public static Level getDefaultLogLevel() {
		Level log = Logger.getRootLogger().getLevel();
		return (log != null) ? log : Level.INFO;
	}
	
	////////////////////////
	// Sparsity Estimates //
	////////////////////////
	
	public static long getMatMultNnz(double sp1, double sp2, long m, long k, long n, boolean worstcase) {
		return getNnz( m, n, getMatMultSparsity(sp1, sp2, m, k, n, worstcase));
	}
	
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
	public static double getMatMultSparsity(double sp1, double sp2, long m, long k, long n, boolean worstcase) {
		if( worstcase ){
			double nnz1 = sp1 * m * k;
			double nnz2 = sp2 * k * n;
			return Math.min(1, nnz1/m) * Math.min(1, nnz2/n);
		}
		else
			return 1 - Math.pow(1-sp1*sp2, k);
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

	public static boolean isBinaryOpSparsityConditionalSparseSafe( OpOp2 op, LiteralOp lit ) {
		double val = HopRewriteUtils.getDoubleValueSafe(lit);
		return (  (op==OpOp2.GREATER  && val==0) 
				||(op==OpOp2.LESS     && val==0)
				||(op==OpOp2.NOTEQUAL && val==0)
				||(op==OpOp2.EQUAL    && val!=0)
				||(op==OpOp2.MINUS    && val==0)
				||(op==OpOp2.PLUS     && val==0)
				||(op==OpOp2.POW      && val!=0)
				||(op==OpOp2.MAX      && val<=0)
				||(op==OpOp2.MIN      && val>=0));
	}
	
	public static double getBinaryOpSparsityConditionalSparseSafe( double sp1, OpOp2 op, LiteralOp lit ) {
		return isBinaryOpSparsityConditionalSparseSafe(op, lit) ? sp1 : 1.0;
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
		
		if(op == null) // If Unknown op, assume the worst
			return ret;

		if( worstcase )
		{
			//NOTE: for matrix-scalar operations this estimate is too conservative, because 
			//Math.min(1, sp1 + sp2) will always give a sparsity 1 if we pass sp2=1 for scalars.
			//In order to do better (with guarantees), we need to take the actual values into account  
			switch(op) {
				case PLUS:
				case MINUS:
				case LESS:
				case GREATER:
				case NOTEQUAL:
				case MIN:
				case MAX:
				case OR:
					ret = worstcase ? Math.min(1, sp1 + sp2) :
						sp1 + sp2 - sp1 * sp2; break;
				case MULT:
				case AND:
					ret = worstcase ? Math.min(sp1, sp2) :
						sp1 * sp2; break;
				case DIV:
					ret = Math.min(1, sp1 + (1-sp2)); break;
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
	
	public static long getOuterNonZeros(long n1, long n2, long nnz1, long nnz2, OpOp2 op) {
		if( nnz1 < 0 || nnz2 < 0 || op == null )
			return n1 * n2;
		switch(op) {
			case PLUS:
			case MINUS:
			case LESS:
			case GREATER:
			case NOTEQUAL:
			case MIN:
			case MAX:
			case OR:
				return n1 * n2
					- (n1-nnz1) * (n2-nnz2);
			case MULT:
			case AND:
				return nnz1 * nnz2;
			default:
				return n1 * n2;
		}
	}
	
	public static long getNnz(long dim1, long dim2, double sp) {
		return Math.round(sp * dim1 * dim2);
	}
	
	public static double getSparsity( DataCharacteristics dc ) {
		return getSparsity(dc.getRows(), dc.getCols(), dc.getNonZeros());
	}
	
	public static double getSparsity( long dim1, long dim2, long nnz ) {
		return ( dim1<=0 || dim2<=0 || nnz<0 ) ? 1.0 :
			Math.min(((double)nnz)/dim1/dim2, 1.0);
	}

	public static double getSparsity(Hop hop){
		long dim1 = hop.getDim1();
		long dim2 = hop.getDim2();
		long nnz = hop.getNnz();
		return getSparsity(dim1, dim2, nnz);
	}

	public static double getSparsity(long[] dims, long nnz) {
		double sparsity = nnz;
		for (long dim : dims) {
			if (dim <= 0) {
				return 1.0;
			}
			sparsity /= dim;
		}
		return Math.min(sparsity, 1.0);
	}

	public static String toMB(double inB) {
		if ( inB < 0 )
			return "-";
		return String.format("%.0f", inB/(1024*1024) );
	}
	
	public static long getNumIterations(ForProgramBlock fpb, long defaultValue) {
		if( fpb.getStatementBlock()==null )
			return defaultValue;
		ForStatementBlock fsb = (ForStatementBlock) fpb.getStatementBlock();
		return getNumIterations(fsb, defaultValue);
	}

	public static long getNumIterations(ForStatementBlock fsb, long defaultValue){
		HashMap<Long,Long> memo = new HashMap<>();
		long from = rEvalSimpleLongExpression(fsb.getFromHops().getInput().get(0), memo);
		long to = rEvalSimpleLongExpression(fsb.getToHops().getInput().get(0), memo);
		long increment = (fsb.getIncrementHops()==null) ? (from < to) ? 1 : -1 :
			rEvalSimpleLongExpression(fsb.getIncrementHops().getInput().get(0), memo);
		if( from != Long.MAX_VALUE && to != Long.MAX_VALUE && increment != Long.MAX_VALUE )
			return (int)Math.ceil(((double)(to-from+1))/increment);
		else return defaultValue;
	}
	
	public static long getNumIterations(ForProgramBlock fpb, LocalVariableMap vars, long defaultValue) {
		if( fpb.getStatementBlock()==null )
			return defaultValue;
		ForStatementBlock fsb = (ForStatementBlock) fpb.getStatementBlock();
		try {
			HashMap<Long,Long> memo = new HashMap<>();
			long from = rEvalSimpleLongExpression(fsb.getFromHops().getInput().get(0), memo, vars);
			long to = rEvalSimpleLongExpression(fsb.getToHops().getInput().get(0), memo, vars);
			long increment = (fsb.getIncrementHops()==null) ? (from < to) ? 1 : -1 : 
				rEvalSimpleLongExpression(fsb.getIncrementHops().getInput().get(0), memo);
			if( from != Long.MAX_VALUE && to != Long.MAX_VALUE && increment != Long.MAX_VALUE )
				return (int)Math.ceil(((double)(to-from+1))/increment);
		}
		catch(Exception ex){}
		return defaultValue;
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
	 */
	public static long rEvalSimpleLongExpression( Hop root, Map<Long, Long> valMemo ) 
	{
		long ret = Long.MAX_VALUE;
		
		//for simplicity and robustness call double and cast.
		HashMap<Long, Double> dvalMemo = new HashMap<>();
		double tmp = rEvalSimpleDoubleExpression(root, dvalMemo);
		if( tmp!=Double.MAX_VALUE )
			ret = UtilFunctions.toLong( tmp );
		
		return ret;
	}
	
	public static long rEvalSimpleLongExpression( Hop root, Map<Long, Long> valMemo, LocalVariableMap vars ) 
	{
		long ret = Long.MAX_VALUE;
		
		//for simplicity and robustness call double and cast.
		HashMap<Long, Double> dvalMemo = new HashMap<>();
		double tmp = rEvalSimpleDoubleExpression(root, dvalMemo, vars);
		if( tmp!=Double.MAX_VALUE )
			ret = UtilFunctions.toLong( tmp );
		
		return ret;
	}
	
	public static double rEvalSimpleDoubleExpression( Hop root, Map<Long, Double> valMemo ) 
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
			else if( root instanceof TernaryOp )
				ret = rEvalSimpleTernaryDoubleExpression(root, valMemo);
		}
		
		valMemo.put(root.getHopID(), ret);
		return ret;
	}
	
	public static double rEvalSimpleDoubleExpression( Hop root, Map<Long, Double> valMemo, LocalVariableMap vars ) 
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

	protected static double rEvalSimpleUnaryDoubleExpression( Hop root, Map<Long, Double> valMemo ) 
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;
		
		UnaryOp uroot = (UnaryOp) root;
		Hop input = uroot.getInput().get(0);
		
		if(uroot.getOp() == OpOp1.NROW)
			ret = input.rowsKnown() ? input.getDim1() : Double.MAX_VALUE;
		else if( uroot.getOp() == OpOp1.NCOL )
			ret = input.colsKnown() ? input.getDim2() : Double.MAX_VALUE;
		else
		{
			double lval = rEvalSimpleDoubleExpression(uroot.getInput().get(0), valMemo);
			if( lval != Double.MAX_VALUE )
			{
				switch( uroot.getOp() )
				{
					case SQRT:  ret = Math.sqrt(lval); break;
					case ROUND: ret = Math.round(lval); break;
					case CEIL:  ret = Math.ceil(lval); break;
					case FLOOR: ret = Math.floor(lval); break;
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

	protected static double rEvalSimpleUnaryDoubleExpression( Hop root, Map<Long, Double> valMemo, LocalVariableMap vars ) 
	{
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;
		
		UnaryOp uroot = (UnaryOp) root;
		Hop input = uroot.getInput().get(0);
		
		if(uroot.getOp() == OpOp1.NROW)
			ret = input.rowsKnown() ? input.getDim1() : Double.MAX_VALUE;
		else if( uroot.getOp() == OpOp1.NCOL )
			ret = input.colsKnown() ? input.getDim2() : Double.MAX_VALUE;
		else
		{
			double lval = rEvalSimpleDoubleExpression(uroot.getInput().get(0), valMemo, vars);
			if( lval != Double.MAX_VALUE )
			{
				switch( uroot.getOp() )
				{
					case SQRT:  ret = Math.sqrt(lval); break;
					case ROUND: ret = Math.round(lval); break;
					case CEIL:  ret = Math.ceil(lval); break;
					case FLOOR: ret = Math.floor(lval); break;
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

	protected static double rEvalSimpleBinaryDoubleExpression( Hop root, Map<Long, Double> valMemo ) 
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

	protected static double rEvalSimpleTernaryDoubleExpression( Hop root, Map<Long, Double> valMemo ) {
		//memoization (prevent redundant computation of common subexpr)
		if( valMemo.containsKey(root.getHopID()) )
			return valMemo.get(root.getHopID());
		
		double ret = Double.MAX_VALUE;
		TernaryOp troot = (TernaryOp) root;
		if( troot.getOp()==OpOp3.IFELSE ) {
			if( HopRewriteUtils.isLiteralOfValue(troot.getInput(0), true) )
				ret = rEvalSimpleDoubleExpression(troot.getInput().get(1), valMemo);
			else if( HopRewriteUtils.isLiteralOfValue(troot.getInput(0), false) )
				ret = rEvalSimpleDoubleExpression(troot.getInput().get(2), valMemo);
		}
		valMemo.put(root.getHopID(), ret);
		return ret;
	}
	
	protected static double rEvalSimpleBinaryDoubleExpression( Hop root, Map<Long, Double> valMemo, LocalVariableMap vars ) 
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

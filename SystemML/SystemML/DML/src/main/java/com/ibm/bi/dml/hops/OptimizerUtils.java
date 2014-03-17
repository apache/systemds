/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.text.DecimalFormat;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.Hop.OpOp2;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.io.SparseRow;

public class OptimizerUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	////////////////////////////////////////////////////////
	// Optimizer constants and flags (incl tuning knobs)  //
	////////////////////////////////////////////////////////
	/**
	 * Utilization factor used in deciding whether an operation to be scheduled on CP or MR. 
	 * NOTE: it is important that MEM_UTIL_FACTOR+CacheableData.CACHING_BUFFER_SIZE < 1.0
	 */
	public static final double MEM_UTIL_FACTOR = 0.7d;
	
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
	public static final double INVALID_SIZE = -1d; // memory estimate not computed

	private static DecimalFormat df = new DecimalFormat("#.##");
	
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
	 * Enables if-else branch removal for constant predicates (original literals or 
	 * results of constant folding). 
	 * 
	 */
	public static boolean ALLOW_BRANCH_REMOVAL = true;
	
	/**
	 * Enables simplex expression evaluation for datagen parameters 'rows', 'cols'. Simple
	 * expressions are defined as binary operations on literals and nrow/ncol.
	 */
	public static boolean ALLOW_SIZE_EXPRESSION_EVALUATION = true;

	/**
	 * Enables interprocedural analysis between main script and functions as well as functions
	 * and other functions. This includes, for example, to propagate statistics into functions
	 * if save to do so (e.g., if called once).
	 */
	public static boolean ALLOW_INTER_PROCEDURAL_ANALYSIS = true;

	
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
	 */
	public enum OptimizationLevel { 
		O0_LOCAL_STATIC, 
		O1_LOCAL_MEMORY_MIN,
		O2_LOCAL_MEMORY_DEFAULT,
		O3_GLOBAL_TIME_MEMORY,
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
		if( optlevel < 0 || optlevel > 3 )
			throw new DMLRuntimeException("Error: invalid optimization level '"+optlevel+"' (valid values: 0-3).");
	
		switch( optlevel )
		{
			// opt level 0: static dimensionality
			case 0:
				_optLevel = OptimizationLevel.O0_LOCAL_STATIC;
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				break;
			// opt level 1: memory-based (no advanced rewrites)	
			case 1:
				_optLevel = OptimizationLevel.O1_LOCAL_MEMORY_MIN;
				ALLOW_CONSTANT_FOLDING = false;
				ALLOW_COMMON_SUBEXPRESSION_ELIMINATION = false;
				ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
				ALLOW_BRANCH_REMOVAL = false;
				break;
			// opt level 2: memory-based (all advanced rewrites)
			case 2:
				_optLevel = OptimizationLevel.O2_LOCAL_MEMORY_DEFAULT;
				break;
			// opt level 3: global, time- and memory-based (all advanced rewrites)
			case 3:
				_optLevel = OptimizationLevel.O3_GLOBAL_TIME_MEMORY;
				break;
		}
		setDefaultSize();
	}
	
	/**
	 * 
	 */
	private static void setDefaultSize() 
	{
		//we need to set default_size larger than any execution context
		//memory budget, however, it should not produce overflows on sum
		DEFAULT_SIZE = Math.max( InfrastructureAnalyzer.getLocalMaxMemory(),
				                 InfrastructureAnalyzer.getRemoteMaxMemory() );
	}
	
	/**
	 * Returns memory budget (according to util factor) in bytes
	 * 
	 * @param localOnly specifies if only budget of current JVM or also MR JVMs 
	 * @return
	 */
	public static double getMemBudget( boolean localOnly )
	{
		double ret = -1;		
		if( localOnly )
			ret = InfrastructureAnalyzer.getLocalMaxMemory();
		else
			ret = InfrastructureAnalyzer.getGlobalMaxMemory();
		
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	/**
	 * 
	 * @return
	 */
	public static double getRemoteMemBudget()
	{
		double ret = InfrastructureAnalyzer.getRemoteMaxMemory();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
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
		if( !configOnly )
			ret = Math.min(ret,InfrastructureAnalyzer.getRemoteParallelReduceTasks());
		
		return ret;
	}
	
	////////////////////////
	// Memory Estimates   //
	////////////////////////
	
	
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
		return MatrixBlockDSM.estimateSize(nrows,ncols,sp);
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
	public static long estimateSize(long nrows, long ncols, double expectedSparsity) 
	{
		double lexpectedSparsity = 1.0;		
		return estimateSizeExactSparsity(nrows, ncols, lexpectedSparsity);
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
				return (1 - (1-sp1)*(1-sp2)); 
			
			case MULT:
				// result[i,j] != 0 iff A[i,j] !=0 && B[i,j] != 0
				// worst case estimate = min(sp1,sp2)
				return sp1 * sp2;  
				
			case DIV:
				return 1.0; // worst case estimate
				
			case LESS: 
			case LESSEQUAL:
			case GREATER:
			case GREATEREQUAL:
			case EQUAL: 
			case NOTEQUAL:
				return 1.0; // purely data-dependent operations, and hence worse-case estimate
				
			//MIN, MAX, AND, OR, LOG, POW
			}
		}
		
		return ret; 
	}
	
	public static double getSparsity( long dim1, long dim2, long nnz )
	{
		if( dim1<=0 || dim2<=0 || nnz<0 )
			return 1.0;
		else
			return ((double)nnz)/dim1/dim2;
	}
	
	public static String toMB(double inB) {
		if ( inB < 0 )
			return "-";
		return String.valueOf( df.format((inB/(1024*1024))) ) + "";
	}
}

package com.ibm.bi.dml.hops;

import java.text.DecimalFormat;

import com.ibm.bi.dml.hops.Hops.OpOp2;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.runtime.matrix.io.SparseRow;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

public class OptimizerUtils {

	/**
	 * Optimization Types for Compilation
	 * 
	 *  STATIC - Decisions for scheduling operations on to CP/MR are based on
	 *  predefined set of rules, which check if the dimensions are below a 
	 *  fixed/static threshold (OLD Method of choosing between CP and MR).
	 *  
	 *  MEMORY_BASED - Every operation is scheduled either on CP or MR, solely
	 *  based on the amount of memory required to perform that operation. 
	 *  It does NOT take the execution time into account.
	 *
	 */
	public enum OptimizationType { STATIC, MEMORY_BASED };
	private static OptimizationType _optType = OptimizationType.MEMORY_BASED;
	public static OptimizationType getOptType() {
		return _optType;
	}
	
	/**
	 * Optimization Modes for Compilation
	 * 
	 * ROBUST - prepares for worst-case scenarios, and tries to avoid OutofMemoryExceptions
	 * 
	 * AGGRESSIVE - prepares for average-case scenarios, takes into account "expectations" 
	 * i.e., expected #nnz, expected dimensions. Robustness has to be incorporated by taking
	 * necessary precautions at runtime (e.g., make sure to avoid OutOfMemoryExceptions)
	 *              
	 */
	public enum OptimizationMode { NONE, ROBUST, AGGRESSIVE };
	private static OptimizationMode _optMode = OptimizationMode.ROBUST;
	
	public static OptimizationMode getOptMode() {
		return _optMode;
	}
	
	public static void setOptimizationLevel( int optlevel ) 
		throws DMLRuntimeException
	{
		if( optlevel < 1 || optlevel > 3 )
			throw new DMLRuntimeException("Error: invalid optimization level '"+optlevel+"' (valid values: 1-3).");
	
		switch( optlevel )
		{
			// opt level 1: static dimensionality
			case 1:
				_optType = OptimizationType.STATIC;
				_optMode = OptimizationMode.NONE;
				break;
			// opt level 2: memory-based (worst-case assumptions)	
			case 2:
				_optType = OptimizationType.MEMORY_BASED;
				_optMode = OptimizationMode.ROBUST;
				break;
			// opt level 3: memory-based (average-case assumptions)
			case 3:
				_optType = OptimizationType.MEMORY_BASED;
				_optMode = OptimizationMode.AGGRESSIVE;
				break;
		}
		setDefaultSize();
	}
	
	
	/**
	 * Optimization Constants
	 */
	
	// Utilization factor used in deciding whether an operation to be scheduled on CP or MR. 
	// NOTE: it is important that MEM_UTIL_FACTOR+CacheableData.CACHING_BUFFER_SIZE < 1.0
	public static final double MEM_UTIL_FACTOR = 0.7d;
	
	/*
	 * Default memory size, which is used the actual estimate can not be computed 
	 * -- for example, when input/output dimensions are unknown. In case of ROBUST,
	 * the default is set to a large value so that operations are scheduled on MR.  
	 */
	public static double DEFAULT_SIZE;
	public static final double DEF_MEM_FACTOR = 0.25d; 
	private static void setDefaultSize() {
		if ( _optMode == OptimizationMode.ROBUST ) 
			DEFAULT_SIZE = InfrastructureAnalyzer.getLocalMaxMemory();
		else 
			DEFAULT_SIZE = DEF_MEM_FACTOR * InfrastructureAnalyzer.getLocalMaxMemory();
	}

	// used when result sparsity can not be estimated, as per uniform distribution of non-zeros
	public static final double DEF_SPARSITY = 0.1d;
	
	// size of native scalars
	public static final long DOUBLE_SIZE = 8;
	public static final long INT_SIZE = 4;
	public static final long CHAR_SIZE = 1;
	public static final long BOOLEAN_SIZE = 1;
	public static final double INVALID_SIZE = -1d; // used to indicate the memory estimate is not computed
	
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
		if (_optMode == OptimizationMode.ROBUST  ) {
			// In case of ROBUST, ignore the value given for <code>sp</code>, 
			// and provide a worst-case estimate i.e., a dense representation
			expectedSparsity = 1.0;
		}
		
		return estimateSizeExactSparsity(nrows, ncols, expectedSparsity);
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
	public static long estimateRowSize(long clen, double sp) {
		
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
	public static double matMultSparsity(double sp1, double sp2, long m, long k, long n) {
		return (1 - Math.pow(1-sp1*sp2, k) );
	}
	
	/**
	 * Estimates the result sparsity for matrix-matrix binary operations (A op B)
	 * 
	 * @param sp1 -- sparsity of A
	 * @param sp2 -- sparsity of B
	 * @param op -- binary operation
	 * @return
	 */
	public static double binaryOpSparsity(double sp1, double sp2, OpOp2 op) {
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
		
		return 1.0; // default is worst-case estimate for robustness
	}
	
	static DecimalFormat df = new DecimalFormat("#.##");
	public static String toMB(double inB) {
		if ( inB < 0 )
			return "-";
		return String.valueOf( df.format((inB/(1024*1024))) ) + "";
	}
}

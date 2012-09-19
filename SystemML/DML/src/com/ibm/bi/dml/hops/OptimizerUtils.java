package com.ibm.bi.dml.hops;

import com.ibm.bi.dml.hops.Hops.OpOp2;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;

public class OptimizerUtils {

	/**
	 * Optimization Types for Compilation
	 * 
	 *  MEMORY_BASED - Every operation is scheduled either on CP or MR, solely
	 *  based on the amount of memory required to perform that operation. 
	 *  It does NOT take the execution time into account.
	 *
	 */
	public enum OptimizationType { MEMORY_BASED };
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
	public enum OptimizationMode { ROBUST, AGGRESSIVE };
	private static OptimizationMode _optMode = OptimizationMode.AGGRESSIVE;
	
	public static OptimizationMode getOptMode() {
		return _optMode;
	}
	
	/**
	 * Constants used in estimating the memory footprint for hops
	 */
	
	// Utilization factor used in deciding whether an operation to be scheduled on CP or MR. 
	public static final double MEM_UTIL_FACTOR = 0.7d;
	
	// Used in determining the size of default memory estimates -- applicable only for AGGRESSIVE optimization. 
	public static final double DEF_MEM_FACTOR = 0.25d; 

	// size of native scalars
	public static final long DOUBLE_SIZE = 8;
	public static final long INT_SIZE = 4;
	public static final long CHAR_SIZE = 1;
	public static final long BOOLEAN_SIZE = 1;

	// used when result sparsity can not be estimated, as per uniform distribution of non-zeros
	public static final double DEFAULT_SPARSITY = 0.1d;
	
	/*
	 * Default memory size, which is used the actual estimate can not be computed 
	 * -- for example, when input/output dimensions are unknown. In case of ROBUST,
	 * the default is set to a large value so that operations are scheduled on MR.  
	 */
	public static double DEFAULT_SIZE = -1d;
	static {
		if ( _optMode == OptimizationMode.ROBUST ) 
			DEFAULT_SIZE = InfrastructureAnalyzer.getLocalMaxMemory();
		else 
			DEFAULT_SIZE = DEF_MEM_FACTOR * InfrastructureAnalyzer.getLocalMaxMemory();
	}

	// used to indicate the memory estimate is not computed
	public static final long INVALID_SIZE = -2;

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
	public static long estimate(long nrows, long ncols, double sp) {
		
		boolean sparse_rep = true; // type of representation
		if (ncols == 1) {
			sparse_rep = false;
		}
		else {
			sparse_rep = sp < MatrixBlockDSM.SPARCITY_TURN_POINT;
		}
		
		long size = 44;// the basic variables and references sizes
		if (sparse_rep) {
			size += (Math.floor(sp * ncols) * 12 + 28) * nrows;
		} else
			size += nrows * ncols * 8;

		return size;
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
	public static long estimateSize(long nrows, long ncols, double sp) {
		if (_optMode == OptimizationMode.ROBUST  ) {
			// In case of ROBUST, ignore the value given for <code>sp</code>, 
			// and provide a worst-case estimate i.e., a dense representation
			sp = 1.0;
		}
		return estimate(nrows, ncols, sp);
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
}

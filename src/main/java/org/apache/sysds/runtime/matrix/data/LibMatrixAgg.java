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

package org.apache.sysds.runtime.matrix.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;
import org.apache.sysds.runtime.codegen.SpoofOperator.SideInputSparseCell;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFP64DEDUP;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceDiag;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator.AggregateOperationTypes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;


/**
 * MB:
 * Library for matrix aggregations including ak+, uak+ for all
 * combinations of dense and sparse representations, and corrections.
 * Those are performance-critical operations because they are used
 * on combiners/reducers of important operations like tsmm, mvmult,
 * indexing, but also basic sum/min/max/mean, row*, col*, etc. Specific
 * handling is especially required for all non sparse-safe operations
 * in order to prevent unnecessary worse asymptotic behavior.
 *
 * This library currently covers the following opcodes:
 * ak+, uak+, uark+, uack+, uasqk+, uarsqk+, uacsqk+,
 * uamin, uarmin, uacmin, uamax, uarmax, uacmax,
 * ua*, uamean, uarmean, uacmean, uavar, uarvar, uacvar,
 * uarimax, uaktrace, cumk+, cummin, cummax, cum*, tak+,
 * cm, cov
 * 
 * TODO next opcode extensions: a+, colindexmax
 */
public class LibMatrixAgg {
	protected static final Log LOG = LogFactory.getLog(LibMatrixAgg.class.getName());

	//internal configuration parameters
	private static final boolean NAN_AWARENESS = false;
	public static final long PAR_NUMCELL_THRESHOLD1 = 1024*256; //Min 256K elements
	private static final long PAR_NUMCELL_THRESHOLD2 = 1024*4;   //Min 4K elements
	private static final long PAR_INTERMEDIATE_SIZE_THRESHOLD = 2*1024*1024; //Max 2MB
	
	////////////////////////////////
	// public matrix agg interface
	////////////////////////////////
	
	private enum AggType {
		KAHAN_SUM,
		KAHAN_SUM_SQ,
		SUM, 
		SUM_SQ,
		CUM_KAHAN_SUM,
		ROW_CUM_SUM,
		CUM_MIN,
		CUM_MAX,
		CUM_PROD,
		CUM_SUM_PROD,
		MIN,
		MAX,
		MEAN,
		VAR,
		MAX_INDEX,
		MIN_INDEX,
		PROD,
		INVALID,
	}

	private LibMatrixAgg() {
		//prevent instantiation via private constructor
	}
	
	/**
	 * Core incremental matrix aggregate (ak+) as used in mapmult, tsmm, 
	 * cpmm, etc. Note that we try to keep the current 
	 * aggVal and aggCorr in dense format in order to allow efficient
	 * access according to the dense/sparse input. 
	 * 
	 * @param in input matrix
	 * @param aggVal current aggregate values (in/out)
	 * @param aggCorr current aggregate correction (in/out)
	 * @param deep deep copy flag
	 */
	public static void aggregateBinaryMatrix(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr, boolean deep) {
		//Timing time = new Timing(true);
		//boolean saggVal = aggVal.sparse, saggCorr = aggCorr.sparse;
		//long naggVal = aggVal.nonZeros, naggCorr = aggCorr.nonZeros;
		
		//common empty block handling
		if( in.isEmptyBlock(false) ) {
			return;
		}
		if( !deep && aggVal.isEmptyBlock(false) ) {
			//shallow copy without correction allocation
			aggVal.copyShallow(in);
			return;
		}
		
		//ensure MCSR instead of CSR for update in-place
		if( aggVal.sparse && aggVal.isAllocated() && aggVal.getSparseBlock() instanceof SparseBlockCSR )
			aggVal.sparseBlock = SparseBlockFactory.copySparseBlock(SparseBlock.Type.MCSR, aggVal.getSparseBlock(), true);
		if( aggCorr.sparse && aggCorr.isAllocated() && aggCorr.getSparseBlock() instanceof SparseBlockCSR )
			aggCorr.sparseBlock = SparseBlockFactory.copySparseBlock(SparseBlock.Type.MCSR, aggCorr.getSparseBlock(), true);
		
		//core aggregation
		if(!in.sparse && !aggVal.sparse && !aggCorr.sparse)
			aggregateBinaryMatrixAllDense(in, aggVal, aggCorr);
		else if(in.sparse && !aggVal.sparse && !aggCorr.sparse)
			aggregateBinaryMatrixSparseDense(in, aggVal, aggCorr);
		else if(in.sparse ) //any aggVal, aggCorr
			aggregateBinaryMatrixSparseGeneric(in, aggVal, aggCorr);
		else //if( !in.sparse ) //any aggVal, aggCorr
			aggregateBinaryMatrixDenseGeneric(in, aggVal, aggCorr);
		
		//System.out.println("agg ("+in.rlen+","+in.clen+","+in.nonZeros+","+in.sparse+"), ("+naggVal+","+saggVal+"), ("+naggCorr+","+saggCorr+") -> " +
		//	"("+aggVal.nonZeros+","+aggVal.sparse+"), ("+aggCorr.nonZeros+","+aggCorr.sparse+") in "+time.stop()+"ms.");
	}
	
	/**
	 * Core incremental matrix aggregate (ak+) as used for uack+ and acrk+.
	 * Embedded correction values.
	 * 
	 * DOES NOT EVALUATE SPARSITY SINCE IT IS USED IN INCREMENTAL AGGREGATION
	 * 
	 * @param in matrix block
	 * @param aggVal aggregate operator
	 * @param aop aggregate operator
	 */
	public static void aggregateBinaryMatrix(MatrixBlock in, MatrixBlock aggVal, AggregateOperator aop) {
		//sanity check matching dimensions 
		if( in.getNumRows()!=aggVal.getNumRows() || in.getNumColumns()!=aggVal.getNumColumns() )
			throw new DMLRuntimeException("Dimension mismatch on aggregate: "+in.getNumRows()+"x"+in.getNumColumns()+
					" vs "+aggVal.getNumRows()+"x"+aggVal.getNumColumns());


		//core aggregation
		boolean lastRowCorr = (aop.correction == CorrectionLocationType.LASTROW);
		boolean lastColCorr = (aop.correction == CorrectionLocationType.LASTCOLUMN);
		if( !in.sparse && lastRowCorr )
			aggregateBinaryMatrixLastRowDenseGeneric(in, aggVal);
		else if( in.sparse && lastRowCorr )
			aggregateBinaryMatrixLastRowSparseGeneric(in, aggVal);
		else if( !in.sparse && lastColCorr )
			aggregateBinaryMatrixLastColDenseGeneric(in, aggVal);
		else //if( in.sparse && lastColCorr )
			aggregateBinaryMatrixLastColSparseGeneric(in, aggVal);

	}

	public static MatrixBlock aggregateUnaryMatrix(AggregateUnaryOperator op, MatrixBlock in, MatrixValue result,
	int blen, MatrixIndexes indexesIn, boolean inCP){

		MatrixBlock ret = LibMatrixAgg.prepareAggregateUnaryOutput(in, op, result, blen);
		
		if( LibMatrixAgg.isSupportedUnaryAggregateOperator(op) ) {
			LibMatrixAgg.aggregateUnaryMatrix(in, ret, op, op.getNumThreads());
			LibMatrixAgg.recomputeIndexes(ret, op, blen, indexesIn);
		}
		else
			LibMatrixAggUnarySpecialization.aggregateUnary(in, op, ret, blen, indexesIn);
		
		if(op.aggOp.existsCorrection() && inCP)
			ret.dropLastRowsOrColumns(op.aggOp.correction);
		
		return ret;
	}

	public static void aggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, AggregateUnaryOperator uaop) {
		aggregateUnaryMatrix(in, out, uaop, true);
	}


	public static void aggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, AggregateUnaryOperator uaop,
		boolean allowReformatToSparse) {

		AggType aggtype = getAggType(uaop);
		final int m = in.rlen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		
		//filter empty input blocks (incl special handling for sparse-unsafe operations)
		if( in.isEmptyBlock(false) ){
			aggregateUnaryMatrixEmpty(in, out, aggtype, uaop.indexFn);
			return;
		}
		
		//Timing time = new Timing(true);
		
		//allocate output arrays (if required)
		out.reset(m2, n2, false); //always dense
		out.allocateDenseBlock();
		
		if( !in.sparse )
			aggregateUnaryMatrixDense(in, out, aggtype, uaop.aggOp.increOp.fn, uaop.indexFn, 0, m);
		else
			aggregateUnaryMatrixSparse(in, out, aggtype, uaop.aggOp.increOp.fn, uaop.indexFn, 0, m);
		
		//cleanup output and change representation (if necessary)
		out.recomputeNonZeros(uaop.getNumThreads());
		if(allowReformatToSparse)
			out.examSparsity();
	}

	public static void aggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, AggregateUnaryOperator uaop, int k) {
		//fall back to sequential version if necessary
		if( !satisfiesMultiThreadingConstraints(in, out, uaop, k) ) {
			if(uaop.aggOp.increOp.fn instanceof Builtin && (((((Builtin) uaop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MININDEX)
				|| (((Builtin) uaop.aggOp.increOp.fn).getBuiltinCode() == BuiltinCode.MAXINDEX)) && uaop.aggOp.correction.getNumRemovedRowsColumns()==0))
					out.clen = 2;
			aggregateUnaryMatrix(in, out, uaop);
			return;
		}
		
		//prepare meta data
		AggType aggtype = getAggType(uaop);
		final int m = in.rlen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		
		//filter empty input blocks (incl special handling for sparse-unsafe operations)
		if( in.isEmptyBlock(false) ){
			aggregateUnaryMatrixEmpty(in, out, aggtype, uaop.indexFn);
			return;
		}
		
		//Timing time = new Timing(true);
		
		//allocate output arrays (if required)
		if( uaop.indexFn instanceof ReduceCol ) {
			out.reset(m2, n2, false); //always dense
			out.allocateDenseBlock();
		}

		//core multi-threaded unary aggregate computation
		//(currently: always parallelization over number of rows)
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<AggTask> tasks = new ArrayList<>();
			ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(m, k,
				(uaop.indexFn instanceof ReduceRow)); //use static partitioning for col*()
			for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ ) {
				tasks.add( (uaop.indexFn instanceof ReduceCol) ? 
					new RowAggTask(in, out, aggtype, uaop, lb, lb+blklens.get(i)) :
					new PartialAggTask(in, out, aggtype, uaop, lb, lb+blklens.get(i)) );
			}
			List<Future<Object>> rtasks = pool.invokeAll(tasks);

			//aggregate partial results
			if( uaop.indexFn instanceof ReduceCol ) {
				//error handling and nnz aggregation
				out.setNonZeros(rtasks.stream()
					.mapToLong(t -> (long)UtilFunctions.getSafe(t)).sum());
			}
			else { //colAgg()/agg()
				out.copy(((PartialAggTask)tasks.get(0)).getResult(), false); //for init
				for( int i=1; i<tasks.size(); i++ )
					aggregateFinalResult(uaop.aggOp, out, ((PartialAggTask)tasks.get(i)).getResult());
				out.recomputeNonZeros();
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		//cleanup output and change representation (if necessary)
		out.examSparsity();
		
		//System.out.println("uagg k="+k+" ("+in.rlen+","+in.clen+","+in.sparse+") in "+time.stop()+"ms.");
	}

	public static MatrixBlock cumaggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, UnaryOperator uop) {
		return cumaggregateUnaryMatrix(in, out, uop, null);
	}
	
	public static MatrixBlock cumaggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, UnaryOperator uop, double[] agg) {
		//Check this implementation, standard case for cumagg (single threaded)

		//prepare meta data
		AggType aggtype = getAggType(uop);
		final int m = in.rlen;
		final int m2 = out.rlen;
		final int n2 = out.clen;

		//filter empty input blocks (incl special handling for sparse-unsafe operations)
		if( in.isEmpty() && (agg == null || aggtype == AggType.CUM_SUM_PROD) ) {
			return aggregateUnaryMatrixEmpty(in, out, aggtype, null);
		}
		
		//allocate output arrays (if required)
		if( !uop.isInplace() || in.isInSparseFormat() || in.isEmpty() ) {
			out.reset(m2, n2, false); //always dense
			out.allocateDenseBlock();
			if( in.isEmpty() )
				in.allocateBlock();
		}
		else {
			out = in;
		}
		
		//Timing time = new Timing(true);

		if( !in.sparse )
			cumaggregateUnaryMatrixDense(in, out, aggtype, uop.fn, agg, 0, m);
		else
			cumaggregateUnaryMatrixSparse(in, out, aggtype, uop.fn, agg, 0, m);
		
		//cleanup output and change representation (if necessary)
		out.recomputeNonZeros();
		out.examSparsity();
		
		//System.out.println("uop ("+in.rlen+","+in.clen+","+in.sparse+") in "+time.stop()+"ms.");
		
		return out;
	}

	public static MatrixBlock cumaggregateUnaryMatrix(MatrixBlock in, MatrixBlock out, UnaryOperator uop, int k) {
		AggregateUnaryOperator uaop = InstructionUtils.parseBasicCumulativeAggregateUnaryOperator(uop);
		
		//fall back to sequential if necessary or agg not supported
		if( k <= 1 || (long)in.rlen*in.clen < PAR_NUMCELL_THRESHOLD1 || in.rlen <= k
			|| out.clen*8*k > PAR_INTERMEDIATE_SIZE_THRESHOLD || uaop == null || !out.isThreadSafe()) {
			return cumaggregateUnaryMatrix(in, out, uop);
		}
		
		//prepare meta data 
		AggType aggtype = getAggType(uop);
		final int m = in.rlen;
		final int m2 = out.rlen;
		final int n2 = out.clen;
		final int mk = aggtype==AggType.CUM_KAHAN_SUM?2:1;
		
		//filter empty input blocks (incl special handling for sparse-unsafe operations)
		if( in.isEmpty() ){
			return aggregateUnaryMatrixEmpty(in, out, aggtype, null);
		}

		//Timing time = new Timing(true);
		
		//allocate output arrays (if required)
		if( !uop.isInplace() || in.isInSparseFormat() || in.isEmpty() ) {
			out.reset(m2, n2, false); //always dense
			out.allocateDenseBlock();
		}
		else {
			out = in;
		}
		
		//core multi-threaded unary aggregate computation
		//(currently: always parallelization over number of rows)
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			int blklen = (int)(Math.ceil((double)m/k));
			
			//step 1: compute aggregates per row partition
			AggType uaoptype = getAggType(uaop);
			ArrayList<PartialAggTask> tasks = new ArrayList<>();
			for( int i=0; i<k & i*blklen<m; i++ )
				tasks.add( new PartialAggTask(in, new MatrixBlock(mk,n2,false), uaoptype, uaop, i*blklen, Math.min((i+1)*blklen, m)) );
			List<Future<Object>> taskret = pool.invokeAll(tasks);
			for( Future<Object> task : taskret )
				task.get(); //check for errors
			
			//step 2: cumulative aggregate of partition aggregates
			MatrixBlock tmp = new MatrixBlock(tasks.size(), n2, false);
			for( int i=0; i<tasks.size(); i++ ) {
				MatrixBlock row = tasks.get(i).getResult();
				if( uaop.aggOp.existsCorrection() )
					row.dropLastRowsOrColumns(uaop.aggOp.correction);
				tmp.leftIndexingOperations(row, i, i, 0, n2-1, tmp, UpdateType.INPLACE_PINNED);
			}
			MatrixBlock tmp2 = cumaggregateUnaryMatrix(tmp, new MatrixBlock(tasks.size(), n2, false), uop);
			
			//step 3: compute final cumulative aggregate
			ArrayList<CumAggTask> tasks2 = new ArrayList<>();
			for( int i=0; i<k & i*blklen<m; i++ ) {
				double[] agg = (i==0)? null : 
					DataConverter.convertToDoubleVector(tmp2.slice(i-1, i-1, 0, n2-1, new MatrixBlock()), false);
				tasks2.add( new CumAggTask(in, agg, out, aggtype, uop, i*blklen, Math.min((i+1)*blklen, m)) );
			}
			List<Future<Long>> taskret2 = pool.invokeAll(tasks2);		
			//step 4: aggregate nnz
			out.nonZeros = 0; 
			for( Future<Long> task : taskret2 )
				out.nonZeros += task.get();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		//cleanup output and change representation (if necessary)
		out.examSparsity();
		
		//System.out.println("uop k="+k+" ("+in.rlen+","+in.clen+","+in.sparse+") in "+time.stop()+"ms.");
		
		return out;
	}

	/**
	 * Single threaded Covariance and Central Moment operations
	 * 
	 * CM = Central Moment
	 * 
	 * COV = Covariance 
	 * 
	 * @param in1 Main input matrix
	 * @param in2 Second input matrix
	 * @param in3 Third input matrix (not output since output is returned)
	 * @param fn Value function to apply
	 * @return Central Moment or Covariance object
	 */
	public static CmCovObject aggregateCmCov(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, ValueFunction fn) {
		CmCovObject cmobj = new CmCovObject();
		
		// empty block handling (important for result correctness, otherwise
		// we get a NaN due to 0/0 on reading out the required result)
		if( in1.isEmptyBlock(false) && fn instanceof CM ) {
			fn.execute(cmobj, 0.0, in1.getNumRows());
			return cmobj;
		}
		
		return aggregateCmCov(in1, in2, in3, fn, 0, in1.getNumRows());
	}
	
	/**
	 * Multi threaded Covariance and Central Moment operations
	 * 
	 * CM = Central Moment
	 * 
	 * COV = Covariance 
	 * 
	 * @param in1 Main input matrix
	 * @param in2 Second input matrix
	 * @param in3 Third input matrix (not output since output is returned)
	 * @param fn Value function to apply
	 * @param k Parallelization degree
	 * @return Central Moment or Covariance object
	 */
	public static CmCovObject aggregateCmCov(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, ValueFunction fn, int k) {
		if( in1.isEmptyBlock(false) || !satisfiesMultiThreadingConstraints(in1, k) )
			return aggregateCmCov(in1, in2, in3, fn);
		
		CmCovObject ret = null;
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<AggCmCovTask> tasks = new ArrayList<>();
			ArrayList<Integer> blklens = UtilFunctions.getBalancedBlockSizesDefault(in1.rlen, k, false);
			for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
				tasks.add(new AggCmCovTask(in1, in2, in3, fn, lb, lb+blklens.get(i)));
			List<Future<CmCovObject>> rtasks = pool.invokeAll(tasks);
			
			//aggregate partial results and error handling
			ret = rtasks.get(0).get();
			for( int i=1; i<rtasks.size(); i++ )
				fn.execute(ret, rtasks.get(i).get());
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		return ret;
	}
	
	public static MatrixBlock aggregateTernary(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, MatrixBlock ret, AggregateTernaryOperator op) {
		//early abort if any block is empty
		if( in1.isEmptyBlock(false) || in2.isEmptyBlock(false) || in3!=null&&in3.isEmptyBlock(false) ) {
			return ret;
		}
		
		//Timing time = new Timing(true);
		
		//allocate output arrays (if required)
		ret.reset(ret.rlen, ret.clen, false); //always dense
		ret.allocateDenseBlock();
		
		IndexFunction ixFn = op.indexFn;
		if( !in1.sparse && !in2.sparse && (in3==null||!in3.sparse) ) //DENSE
			aggregateTernaryDense(in1, in2, in3, ret, ixFn, 0, in1.rlen);
		else //GENERAL CASE
			aggregateTernaryGeneric(in1, in2, in3, ret, ixFn, 0, in1.rlen);
		
		//cleanup output and change representation (if necessary)
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("tak+ ("+in1.rlen+","+in1.sparse+","+in2.sparse+","+in3.sparse+") in "+time.stop()+"ms.");
	
		return ret;
	}

	public static MatrixBlock aggregateTernary(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, MatrixBlock ret, AggregateTernaryOperator op, int k) {
		//fall back to sequential version if necessary
		if( k <= 1 
			|| in1.nonZeros+in2.nonZeros < PAR_NUMCELL_THRESHOLD1 
			|| in1.rlen <= k/2 
			// || (!(op.indexFn instanceof ReduceCol) &&  ret.clen*8*k > PAR_INTERMEDIATE_SIZE_THRESHOLD)
			 ) {
			return aggregateTernary(in1, in2, in3, ret, op);
		}
		
		//early abort if any block is empty
		if( in1.isEmptyBlock(false) || in2.isEmptyBlock(false) || in3!=null&&in3.isEmptyBlock(false) ) {
			return ret;
		}
			
		//Timing time = new Timing(true);
		
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<AggTernaryTask> tasks = new ArrayList<>();
			int blklen = (int)(Math.ceil((double)in1.rlen/k));
			IndexFunction ixFn = op.indexFn;
			for( int i=0; i<k & i*blklen<in1.rlen; i++ )
				tasks.add( new AggTernaryTask(in1, in2, in3, ret, ixFn, i*blklen, Math.min((i+1)*blklen, in1.rlen)));
			List<Future<MatrixBlock>> rtasks = pool.invokeAll(tasks);	

			//aggregate partial results and error handling
			ret.copy(rtasks.get(0).get(), false); //for init
			for( int i=1; i<rtasks.size(); i++ )
				aggregateFinalResult(op.aggOp, ret, rtasks.get(i).get());
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		//cleanup output and change representation (if necessary)
		ret.recomputeNonZeros();
		ret.examSparsity();
		
		//System.out.println("tak+ k="+k+" ("+in1.rlen+","+in1.sparse+","+in2.sparse+","+in3.sparse+") in "+time.stop()+"ms.");	
	
		return ret;
	}

	public static void groupedAggregate(MatrixBlock groups, MatrixBlock target, MatrixBlock weights, MatrixBlock result, int numGroups, Operator op) {
		if( !(op instanceof CMOperator || op instanceof AggregateOperator) ) {
			throw new DMLRuntimeException("Invalid operator (" + op + ") encountered while processing groupedAggregate.");
		}
		
		//CM operator for count, mean, variance
		//note: current support only for column vectors
		if(op instanceof CMOperator) {
			CMOperator cmOp = (CMOperator) op;
			if( cmOp.getAggOpType()==AggregateOperationTypes.COUNT && weights==null && target.clen==1 ) {
				//special case for vector counts
				groupedAggregateVecCount(groups, result, numGroups);
			}
			else { //general case
				groupedAggregateCM(groups, target, weights, result, numGroups, cmOp, 0, target.clen);
			}
		}
		//Aggregate operator for sum (via kahan sum)
		//note: support for row/column vectors and dense/sparse
		else if( op instanceof AggregateOperator ) {
			AggregateOperator aggop = (AggregateOperator) op;
			groupedAggregateKahanPlus(groups, target, weights, result, numGroups, aggop, 0, target.clen);
		}
		
		//postprocessing sparse/dense formats
		//(nnz already maintained via append)
		result.examSparsity();
	}

	public static void groupedAggregate(MatrixBlock groups, MatrixBlock target, MatrixBlock weights, MatrixBlock result, int numGroups, Operator op, int k) 
	{
		//fall back to sequential version if necessary
		boolean rowVector = (target.getNumRows()==1 && target.getNumColumns()>1);
		if( k <= 1 || (long)target.rlen*target.clen < PAR_NUMCELL_THRESHOLD1 || rowVector || target.clen==1) {
			groupedAggregate(groups, target, weights, result, numGroups, op);
			return;
		}
		
		if( !(op instanceof CMOperator || op instanceof AggregateOperator) ) {
			throw new DMLRuntimeException("Invalid operator (" + op + ") encountered while processing groupedAggregate.");
		}
	
		//preprocessing (no need to check isThreadSafe)
		result.sparse = false;
		result.allocateDenseBlock();
		
		//core multi-threaded grouped aggregate computation
		//(currently: parallelization over columns to avoid additional memory requirements)
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			ArrayList<GrpAggTask> tasks = new ArrayList<>();
			int blklen = (int)(Math.ceil((double)target.clen/k));
			for( int i=0; i<k & i*blklen<target.clen; i++ )
				tasks.add( new GrpAggTask(groups, target, weights, result, numGroups, op, i*blklen, Math.min((i+1)*blklen, target.clen)) );
			List<Future<Object>> taskret = pool.invokeAll(tasks);

			for(Future<Object> task : taskret)
				task.get(); //error handling
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		//postprocessing
		result.recomputeNonZeros();
		result.examSparsity();
	}

	public static boolean isSupportedUnaryAggregateOperator( AggregateUnaryOperator op ) {
		AggType type = getAggType( op );
		return (type != AggType.INVALID);
	}
	
	public static boolean isSupportedUnaryOperator( UnaryOperator op ) {
		AggType type = getAggType( op );
		return (type != AggType.INVALID);
	}
	
	public static boolean satisfiesMultiThreadingConstraints(MatrixBlock in, MatrixBlock out, AggregateUnaryOperator uaop, int k) {
		boolean sharedTP = (InfrastructureAnalyzer.getLocalParallelism() == k);
		return k > 1 && out.isThreadSafe() && in.rlen > (sharedTP ? k/8 : k/2)
			&& (uaop.indexFn instanceof ReduceCol || out.clen*8*k < PAR_INTERMEDIATE_SIZE_THRESHOLD) //size
			&& in.nonZeros > (sharedTP ? PAR_NUMCELL_THRESHOLD2 : PAR_NUMCELL_THRESHOLD1);
	}
	
	public static boolean satisfiesMultiThreadingConstraints(MatrixBlock in, int k) {
		boolean sharedTP = (InfrastructureAnalyzer.getLocalParallelism() == k);
		return k > 1 && in.rlen > (sharedTP ? k/8 : k/2)
			&& in.nonZeros > (sharedTP ? PAR_NUMCELL_THRESHOLD2 : PAR_NUMCELL_THRESHOLD1);
	}
	
	/**
	 * Recompute outputs (e.g., maxindex or minindex) according to block indexes from MR.
	 * TODO: this should not be part of block operations but of the MR instruction.
	 * 
	 * @param out matrix block
	 * @param op aggregate unary operator
	 * @param blen number of rows/cols in a block
	 * @param ix matrix indexes
	 */
	public static void recomputeIndexes( MatrixBlock out, AggregateUnaryOperator op, int blen, MatrixIndexes ix )
	{
		AggType type = getAggType(op);
		if( (type == AggType.MAX_INDEX || type == AggType.MIN_INDEX) && ix != null && ix.getColumnIndex()!=1 ) //MAXINDEX or MININDEX
		{
			int m = out.rlen;
			double[] c = out.getDenseBlockValues();
			for( int i=0, cix=0; i<m; i++, cix+=2 )
				c[cix] = UtilFunctions.computeCellIndex(ix.getColumnIndex(), blen, (int)c[cix]-1);
		}
	}	

	private static AggType getAggType( AggregateUnaryOperator op )
	{
		ValueFunction vfn = op.aggOp.increOp.fn;
		IndexFunction ifn = op.indexFn;
		
		//(kahan) sum / sum squared / trace (for ReduceDiag)
		if( vfn instanceof KahanFunction
			&& (op.aggOp.correction == CorrectionLocationType.LASTCOLUMN || op.aggOp.correction == CorrectionLocationType.LASTROW)
			&& (ifn instanceof ReduceAll || ifn instanceof ReduceCol || ifn instanceof ReduceRow || ifn instanceof ReduceDiag) )
		{
			if (vfn instanceof KahanPlus)
				return AggType.KAHAN_SUM;
			else if (vfn instanceof KahanPlusSq)
				return AggType.KAHAN_SUM_SQ;
		}

		final boolean rAll_rCol_rRow = ifn instanceof ReduceAll || ifn instanceof ReduceCol || ifn instanceof ReduceRow;

		//mean
		if( vfn instanceof Mean 
			&& (op.aggOp.correction == CorrectionLocationType.LASTTWOCOLUMNS || op.aggOp.correction == CorrectionLocationType.LASTTWOROWS)
			&& rAll_rCol_rRow )
		{
			return AggType.MEAN;
		}

		//variance
		if( vfn instanceof CM
				&& ((CM) vfn).getAggOpType() == AggregateOperationTypes.VARIANCE
				&& (op.aggOp.correction == CorrectionLocationType.LASTFOURCOLUMNS ||
					op.aggOp.correction == CorrectionLocationType.LASTFOURROWS)
				&& rAll_rCol_rRow )
		{
			return AggType.VAR;
		}

		//prod
		if(vfn instanceof Multiply && rAll_rCol_rRow)
			return AggType.PROD;

		if(vfn instanceof Plus && rAll_rCol_rRow)
			return AggType.SUM;

		// min / max
		if(vfn instanceof Builtin && rAll_rCol_rRow) {
			BuiltinCode bfcode = ((Builtin)vfn).bFunc;
			switch( bfcode ){
				case MAX: return AggType.MAX;
				case MIN: return AggType.MIN;
				case MAXINDEX: return AggType.MAX_INDEX;
				case MININDEX: return AggType.MIN_INDEX;
				default: //do nothing
			}
		}
		
		return AggType.INVALID;
	}

	private static AggType getAggType( UnaryOperator op ) {
		ValueFunction vfn = op.fn;
		if( vfn instanceof Builtin ) {
			BuiltinCode bfunc = ((Builtin) vfn).bFunc;
			switch( bfunc ) {
				case CUMSUM:     return AggType.CUM_KAHAN_SUM;
				case ROWCUMSUM:	 return AggType.ROW_CUM_SUM;
				case CUMPROD:    return AggType.CUM_PROD;
				case CUMMIN:     return AggType.CUM_MIN;
				case CUMMAX:     return AggType.CUM_MAX;
				case CUMSUMPROD: return AggType.CUM_SUM_PROD;
				default:         return AggType.INVALID;
			}
		}
		return AggType.INVALID;
	}

	private static void aggregateFinalResult( AggregateOperator aop, MatrixBlock out, MatrixBlock partout ) {
		AggregateOperator laop = aop;
		
		//special handling for mean where the final aggregate operator (kahan plus)
		//is not equals to the partial aggregate operator
		if( aop.increOp.fn instanceof Mean ) {
			laop = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), aop.correction);
		}

		//incremental aggregation of final results
		if( laop.existsCorrection() )
			out.incrementalAggregate(laop, partout);
		else
			out.binaryOperationsInPlace(laop.increOp, partout);
	}
	
	private static CmCovObject aggregateCmCov(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, ValueFunction fn, int rl, int ru) {
		CmCovObject ret = new CmCovObject();
		
		if( in2 == null && in3 == null ) { //CM
			int nzcount = 0;
			if(in1.sparse && in1.sparseBlock!=null) { //SPARSE
				int ru2 = Math.min(ru, in1.sparseBlock.numRows());
				for(int r = rl; r < ru2; r++) {
					SparseBlock a = in1.sparseBlock;
					if(a.isEmpty(r)) 
						continue;
					int apos = a.pos(r);
					int alen = a.size(r);
					double[] avals = a.values(r);
					for(int i=apos; i<apos+alen; i++) {
						fn.execute(ret, avals[i]);
						nzcount++;
					}
				}
				// account for zeros in the vector
				fn.execute(ret, 0.0, ru2-rl-nzcount);
			}
			else if(in1.denseBlock!=null) { //DENSE
				//always vector (see check above)
				double[] a = in1.getDenseBlockValues();
				for(int i=rl; i<ru; i++)
					fn.execute(ret, a[i]);
			}
		}
		else if( in3 == null ) { //CM w/ weights, COV
			if (in1.sparse && in1.sparseBlock!=null) { //SPARSE
				for(int i = rl; i < ru; i++) { 
					fn.execute(ret,
						in1.get(i,0),
						in2.get(i,0));
				}
			}
			else if(in1.denseBlock!=null) //DENSE
			{
				//always vectors (see check above)
				double[] a = in1.getDenseBlockValues();
				if( !in2.sparse ) {
					if(in2.denseBlock!=null) {
						double[] w = in2.getDenseBlockValues();
						for( int i = rl; i < ru; i++ )
							fn.execute(ret, a[i], w[i]);
					}
				}
				else {
					for(int i = rl; i < ru; i++) 
						fn.execute(ret, a[i], in2.get(i,0) );
				}
			}
		}
		else { // COV w/ weights
			if(in1.sparse && in1.sparseBlock!=null) { //SPARSE
				for(int i = rl; i < ru; i++ ) {
					fn.execute(ret,
						in1.get(i,0),
						in2.get(i,0),
						in3.get(i,0));
				}
			}
			else if(in1.denseBlock!=null) { //DENSE
				//always vectors (see check above)
				double[] a = in1.getDenseBlockValues();
				
				if( !in2.sparse && !in3.sparse ) {
					double[] w = in3.getDenseBlockValues();
					if(in2.denseBlock!=null) {
						double[] b = in2.getDenseBlockValues();
						for( int i=rl; i<ru; i++ )
							fn.execute(ret, a[i], b[i], w[i]);
					}
				}
				else {
					for(int i = rl; i < ru; i++) {
						fn.execute(ret, a[i],
							in2.get(i,0),
							in3.get(i,0));
					}
				}
			}
		}

		return ret;
	}

	private static void aggregateTernaryDense(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, MatrixBlock ret, IndexFunction ixFn, int rl, int ru)
	{
		//compute block operations
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
		
		double[] a = in1.getDenseBlockValues();
		double[] b1 = in2.getDenseBlockValues();
		double[] b2 = (in3!=null) ? in3.getDenseBlockValues() : null; //if null, literal 1
		final int n = in1.clen;
		
		if( ixFn instanceof ReduceAll ) //tak+*
		{
			for( int i=rl, ix=rl*n; i<ru; i++ ) 
				for( int j=0; j<n; j++, ix++ ) {
					double b2val = (b2 != null) ? b2[ix] : 1;
					double val = a[ix] * b1[ix] * b2val;
					kplus.execute2( kbuff, val );
				}
			ret.set(0, 0, kbuff._sum);
			ret.set(0, 1, kbuff._correction);
		}
		else //tack+*
		{
			double[] c = ret.getDenseBlockValues();
			for( int i=rl, ix=rl*n; i<ru; i++ )
				for( int j=0; j<n; j++, ix++ ) {
					double b2val = (b2 != null) ? b2[ix] : 1;
					double val = a[ix] * b1[ix] * b2val;
					kbuff._sum = c[j];
					kbuff._correction = c[j+n];
					kplus.execute2(kbuff, val);
					c[j] = kbuff._sum;
					c[j+n] = kbuff._correction;
				}
		}
	}

	private static void aggregateTernaryGeneric(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, MatrixBlock ret, IndexFunction ixFn, int rl, int ru)
	{		
		//compute block operations
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
		
		//guaranteed to have at least one sparse input, sort by nnz, assume num cells if 
		//(potentially incorrect) in dense representation, keep null at end via stable sort
		MatrixBlock[] blocks = new MatrixBlock[]{in1, in2, in3};
		Arrays.sort(blocks, new Comparator<MatrixBlock>() {
			@Override
			public int compare(MatrixBlock o1, MatrixBlock o2) {
				long nnz1 = (o1!=null && o1.sparse) ? o1.nonZeros : Long.MAX_VALUE;
				long nnz2 = (o2!=null && o2.sparse) ? o2.nonZeros : Long.MAX_VALUE;
				return Long.compare(nnz1, nnz2);
			}
		});
		MatrixBlock lin1 = blocks[0];
		MatrixBlock lin2 = blocks[1];
		MatrixBlock lin3 = blocks[2];
		
		SparseBlock a = lin1.sparseBlock;
		final int n = in1.clen;
		
		if( ixFn instanceof ReduceAll ) //tak+*
		{
			for( int i=rl; i<ru; i++ )
				if( !a.isEmpty(i) ) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						double val1 = avals[j];
						double val2 = lin2.get(i, aix[j]);
						double val = val1 * val2;
						if( val != 0 && lin3 != null )
							val *= lin3.get(i, aix[j]);
						kplus.execute2( kbuff, val );
					}
				}	
			ret.set(0, 0, kbuff._sum);
			ret.set(0, 1, kbuff._correction);
		}
		else //tack+*
		{
			double[] c = ret.getDenseBlockValues();
			for( int i=rl; i<ru; i++ )
				if( !a.isEmpty(i) ) {
					int apos = a.pos(i);
					int alen = a.size(i);
					int[] aix = a.indexes(i);
					double[] avals = a.values(i);
					for( int j=apos; j<apos+alen; j++ ) {
						int colIx = aix[j];
						double val1 = avals[j];
						double val2 = lin2.get(i, colIx);
						double val = val1 * val2;
						if( val != 0 && lin3 != null )
							val *= lin3.get(i, colIx);
						kbuff._sum = c[colIx];
						kbuff._correction = c[colIx+n];
						kplus.execute2( kbuff, val );	
						c[colIx] = kbuff._sum;
						c[colIx+n] = kbuff._correction;	
					}
				}	
		}
	}
	

	/**
	 * This is a specific implementation for aggregate(fn="sum"), where we use KahanPlus for numerical
	 * stability. In contrast to other functions of aggregate, this implementation supports row and column
	 * vectors for target and exploits sparse representations since KahanPlus is sparse-safe.
	 * 
	 * @param groups matrix block groups
	 * @param target matrix block target
	 * @param weights matrix block weights
	 * @param result matrix block result
	 * @param numGroups number of groups
	 * @param aggop aggregate operator
	 * @param cl column lower index
	 * @param cu column upper index
	 */
	private static void groupedAggregateKahanPlus( MatrixBlock groups, MatrixBlock target, MatrixBlock weights, MatrixBlock result, int numGroups, AggregateOperator aggop, int cl, int cu ) {
		boolean rowVector = (target.getNumRows()==1 && target.getNumColumns()>1);
		double w = 1; //default weight
		
		//skip empty blocks (sparse-safe operation)
		if( target.isEmptyBlock(false) ) 
			return;
		
		//init group buffers
		int numCols2 = cu-cl;
		KahanObject[][] buffer = new KahanObject[numGroups][numCols2];
		for( int i=0; i<numGroups; i++ )
			for( int j=0; j<numCols2; j++ )
				buffer[i][j] = new KahanObject(aggop.initialValue, 0);
			
		if( rowVector ) //target is rowvector
		{	
			//note: always sequential, no need to respect cl/cu 
			
			if( target.sparse ) //SPARSE target
			{
				if( !target.sparseBlock.isEmpty(0) )
				{
					int pos = target.sparseBlock.pos(0);
					int len = target.sparseBlock.size(0);
					int[] aix = target.sparseBlock.indexes(0);
					double[] avals = target.sparseBlock.values(0);
					for( int j=pos; j<pos+len; j++ ) //for each nnz
					{
						int g = (int) groups.get(aix[j], 0);
						if ( g > numGroups )
							continue;
						if ( weights != null )
							w = weights.get(aix[j],0);
						aggop.increOp.fn.execute(buffer[g-1][0], avals[j]*w);
					}
				}
			}
			else //DENSE target
			{
				double[] a = target.getDenseBlockValues();
				for ( int i=0; i < target.getNumColumns(); i++ ) {
					double d = a[ i ];
					if( d != 0 ) //sparse-safe
					{
						int g = (int) groups.get(i, 0);
						if ( g > numGroups )
							continue;
						if ( weights != null )
							w = weights.get(i,0);
						// buffer is 0-indexed, whereas range of values for g = [1,numGroups]
						aggop.increOp.fn.execute(buffer[g-1][0], d*w);
					}
				}
			}
		}
		else //column vector or matrix 
		{
			if( target.sparse ) //SPARSE target
			{
				SparseBlock a = target.sparseBlock;
				
				for( int i=0; i < groups.getNumRows(); i++ ) 
				{
					int g = (int) groups.get(i, 0);
					if ( g > numGroups )
						continue;
					
					if( !a.isEmpty(i) )
					{
						int pos = a.pos(i);
						int len = a.size(i);
						int[] aix = a.indexes(i);
						double[] avals = a.values(i);
						int j = (cl==0) ? 0 : a.posFIndexGTE(i,cl);
						j = (j >= 0) ? pos+j : pos+len;
						
						for( ; j<pos+len && aix[j]<cu; j++ ) //for each nnz
						{
							if ( weights != null )
								w = weights.get(aix[j],0);
							aggop.increOp.fn.execute(buffer[g-1][aix[j]-cl], avals[j]*w);
						}
					}
				}
			}
			else //DENSE target
			{
				DenseBlock a = target.getDenseBlock();
				for( int i=0; i < groups.getNumRows(); i++ ) {
					int g = (int) groups.get(i, 0);
					if ( g > numGroups )
						continue;
					double[] avals = a.values(i);
					int aix = a.pos(i);
					for( int j=cl; j < cu; j++ ) {
						double d = avals[ aix+j ];
						if( d != 0 ) { //sparse-safe
							if ( weights != null )
								w = weights.get(i,0);
							// buffer is 0-indexed, whereas range of values for g = [1,numGroups]
							aggop.increOp.fn.execute(buffer[g-1][j-cl], d*w);
						}
					}
				}
			}
		}
		
		// extract the results from group buffers
		for( int i=0; i < numGroups; i++ )
			for( int j=0; j < numCols2; j++ )
				result.appendValue(i, j+cl, buffer[i][j]._sum);
	}

	private static void groupedAggregateCM( MatrixBlock groups, MatrixBlock target, MatrixBlock weights, MatrixBlock result, int numGroups, CMOperator cmOp, int cl, int cu ) {
		CM cmFn = CM.getCMFnObject(cmOp.getAggOpType());
		double w = 1; //default weight
		
		//init group buffers
		int numCols2 = cu-cl;
		CmCovObject[][] cmValues = new CmCovObject[numGroups][numCols2];
		for ( int i=0; i < numGroups; i++ )
			for( int j=0; j < numCols2; j++  )
				cmValues[i][j] = new CmCovObject();
		
		//column vector or matrix
		if( target.sparse ) { //SPARSE target
			//note: we create a sparse side input for a linear scan (w/o binary search)
			//over the sparse representation despite the sparse-unsafe operations 
			SparseBlock a = target.sparseBlock;
			SideInputSparseCell sa = new SideInputSparseCell(
				new SideInput(null, target, target.clen));
			
			for( int i=0; i < groups.getNumRows(); i++ ) {
				int g = (int) groups.get(i, 0);
				if( g > numGroups ) continue;
				
				//sparse unsafe correction empty row
				if( a.isEmpty(i) ){
					w = (weights != null) ? weights.get(i,0) : w;
					for( int j=cl; j<cu; j++ )
						cmFn.execute(cmValues[g-1][j-cl], 0, w);
					continue;
				}
				
				//process non-empty row
				for( int j=cl; j<cu; j++ ) {
					double d = sa.getValue(i, j);
					if ( weights != null )
						w = weights.get(i,0);
					cmFn.execute(cmValues[g-1][j-cl], d, w);
				}
			}
		}
		else { //DENSE target
			DenseBlock a = target.getDenseBlock();
			for( int i=0; i < groups.getNumRows(); i++ ) {
				int g = (int) groups.get(i, 0);
				if ( g > numGroups )
					continue;
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=cl; j<cu; j++ ) {
					double d = avals[ aix+j ]; //sparse unsafe
					if ( weights != null )
						w = weights.get(i,0);
					// buffer is 0-indexed, whereas range of values for g = [1,numGroups]
					cmFn.execute(cmValues[g-1][j-cl], d, w);
				}
			}
		}
		
		// extract the required value from each CM_COV_Object
		for( int i=0; i < numGroups; i++ ) 
			for( int j=0; j < numCols2; j++ ) {
				// result is 0-indexed, so is cmValues
				result.appendValue(i, j, cmValues[i][j+cl].getRequiredResult(cmOp));
			}
	}

	private static void groupedAggregateVecCount( MatrixBlock groups, MatrixBlock result, int numGroups ) {
		//note: groups are always dense because 0 invalid
		if( groups.isInSparseFormat() || groups.isEmptyBlock(false) )
			throw new DMLRuntimeException("Unsupported sparse input for aggregate-count on group vector.");
		
		double[] a = groups.getDenseBlockValues();
		int[] tmp = new int[numGroups];
		int m = groups.rlen;
		
		//compute counts
		for( int i = 0; i < m; i++ ) {
			int g = (int) a[i];
			if ( g > numGroups )
				continue;
			tmp[g-1]++;
		}
		
		//copy counts into result
		for( int i=0; i<numGroups; i++ ) {
			result.appendValue(i, 0, tmp[i]);
		}
	}

	private static void aggregateBinaryMatrixAllDense(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) {
		//allocate output arrays (if required)
		aggVal.allocateDenseBlock(); //should always stay in dense
		aggCorr.allocateDenseBlock(); //should always stay in dense
		
		double[] a = in.getDenseBlockValues();
		double[] c = aggVal.getDenseBlockValues();
		double[] cc = aggCorr.getDenseBlockValues();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		final int len = Math.min(a.length, in.rlen*in.clen);
		
		int nnzC = 0;
		int nnzCC = 0;
		
		for( int i=0; i<len; i++ )
		{
			buffer1._sum        = c[i];
			buffer1._correction = cc[i];
			akplus.execute2(buffer1, a[i]);
			c[i]  = buffer1._sum;
			cc[i] = buffer1._correction;
			nnzC += (buffer1._sum!=0)?1:0;
			nnzCC += (buffer1._correction!=0)?1:0;
		}
		
		aggVal.nonZeros = nnzC;
		aggCorr.nonZeros = nnzCC;
	}

	private static void aggregateBinaryMatrixSparseDense(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) {
		//allocate output arrays (if required)
		aggVal.allocateDenseBlock(); //should always stay in dense
		aggCorr.allocateDenseBlock(); //should always stay in dense
		
		SparseBlock a = in.getSparseBlock();
		double[] c = aggVal.getDenseBlockValues();
		double[] cc = aggCorr.getDenseBlockValues();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int n = in.clen;
		final int rlen = Math.min(a.numRows(), m);
		
		for( int i=0, cix=0; i<rlen; i++, cix+=n )
		{
			if( !a.isEmpty(i) )
			{
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				
				for( int j=apos; j<apos+alen; j++ )
				{
					int ix = cix+aix[j];
					buffer1._sum        = c[ix];
					buffer1._correction = cc[ix];
					akplus.execute2(buffer1, avals[j]);
					c[ix]  = buffer1._sum;
					cc[ix] = buffer1._correction;
				}
			}
		}
		
		aggVal.recomputeNonZeros();
		aggCorr.recomputeNonZeros(); 
	}

	private static void aggregateBinaryMatrixSparseGeneric(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) {
		SparseBlock a = in.getSparseBlock();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int rlen = Math.min(a.numRows(), m);
		
		for( int i=0; i<rlen; i++ )
		{
			if( !a.isEmpty(i) )
			{
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				
				for( int j=apos; j<apos+alen; j++ )
				{
					int jix = aix[j];
					buffer1._sum        = aggVal.get(i, jix);
					buffer1._correction = aggCorr.get(i, jix);
					akplus.execute2(buffer1, avals[j]);
					aggVal.set(i, jix, buffer1._sum);
					aggCorr.set(i, jix, buffer1._correction);
				}
			}
		}
		
		//note: nnz of aggVal/aggCorr maintained internally
		if( aggVal.sparse )
			aggVal.examSparsity(false);
		if( aggCorr.sparse )
			aggCorr.examSparsity(false);
	}

	private static void aggregateBinaryMatrixDenseGeneric(MatrixBlock in, MatrixBlock aggVal, MatrixBlock aggCorr) {
		final int m = in.rlen;
		final int n = in.clen;
		
		double[] a = in.getDenseBlockValues();
		
		KahanObject buffer = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		//incl implicit nnz maintenance
		for(int i=0, ix=0; i<m; i++)
			for(int j=0; j<n; j++, ix++)
			{
				buffer._sum = aggVal.get(i, j);
				buffer._correction = aggCorr.get(i, j);
				akplus.execute(buffer, a[ix]);
				aggVal.set(i, j, buffer._sum);
				aggCorr.set(i, j, buffer._correction);
			}
		
		//note: nnz of aggVal/aggCorr maintained internally 
		if( aggVal.sparse )
			aggVal.examSparsity(false);
		if( aggCorr.sparse )
			aggCorr.examSparsity(false);
	}

	private static void aggregateBinaryMatrixLastRowDenseGeneric(MatrixBlock in, MatrixBlock aggVal) {
		if( in.denseBlock==null || in.isEmptyBlock(false))
			return;
		
		final int m = in.rlen;
		if(m != 2)
			throw new DMLRuntimeException("Invalid input for Aggregate Binary Matrix with correction in last row");
		final int n = in.clen;
		
		double[] a = in.getDenseBlockValues();

		if(aggVal.isEmpty()) {
			aggVal.allocateDenseBlock();
		}
		else if(aggVal.isInSparseFormat()){
			// If for some reason the agg Val is sparse then force it to dence,
			// since the values that are going to be added
			// will make it dense anyway.
			aggVal.sparseToDense();
			if(aggVal.denseBlock == null)
				aggVal.allocateDenseBlock();
		}
		
		double[] t = aggVal.getDenseBlockValues();
		KahanObject buffer = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		// j is the pointer to column.
		// c is the pointer to correction. 
		for(int j=0, c = n; j<n; j++, c++){
			buffer._sum = t[j];
			buffer._correction = t[c];
			akplus.execute(buffer, a[j], a[c]);
			t[j] =  buffer._sum;
			t[c] = buffer._correction;
		}
		
		aggVal.recomputeNonZeros();
	}

	private static void aggregateBinaryMatrixLastRowSparseGeneric(MatrixBlock in, MatrixBlock aggVal) {
		//sparse-safe operation
		if( in.isEmptyBlock(false) )
			return;
		
		SparseBlock a = in.getSparseBlock();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int rlen = Math.min(a.numRows(), m);
		
		if(aggVal.isEmpty())
			aggVal.allocateSparseRowsBlock();
		
		// add to aggVal with implicit nnz maintenance
		for( int i=0; i<rlen-1; i++ ) {
			if( a.isEmpty(i) )
				continue;
			int apos = a.pos(i);
			int alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i);
			
			for( int j=apos; j<apos+alen; j++ ) {
				int jix = aix[j];
				double corr = in.get(m-1, jix);
				buffer1._sum        = aggVal.get(i, jix);
				buffer1._correction = aggVal.get(m-1, jix);
				akplus.execute(buffer1, avals[j], corr);
				aggVal.set(i, jix, buffer1._sum);
				aggVal.set(m-1, jix, buffer1._correction);
			}
		}
	}

	private static void aggregateBinaryMatrixLastColDenseGeneric(MatrixBlock in, MatrixBlock aggVal) {
		if( in.denseBlock==null || in.isEmptyBlock(false) )
			return;
		
		final int m = in.rlen;
		final int n = in.clen;
		
		double[] a = in.getDenseBlockValues();
		
		KahanObject buffer = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		//incl implicit nnz maintenance
		for(int i=0, ix=0; i<m; i++, ix+=n)
			for(int j=0; j<n-1; j++)
			{
				buffer._sum = aggVal.get(i, j);
				buffer._correction = aggVal.get(i, n-1);
				akplus.execute(buffer, a[ix+j], a[ix+j+1]);
				aggVal.set(i, j, buffer._sum);
				aggVal.set(i, n-1, buffer._correction);
			}

	}

	private static void aggregateBinaryMatrixLastColSparseGeneric(MatrixBlock in, MatrixBlock aggVal) {
		//sparse-safe operation
		if( in.isEmptyBlock(false) )
			return;
		
		SparseBlock a = in.getSparseBlock();
		
		KahanObject buffer1 = new KahanObject(0, 0);
		KahanPlus akplus = KahanPlus.getKahanPlusFnObject();
		
		final int m = in.rlen;
		final int n = in.clen;
		final int rlen = Math.min(a.numRows(), m);
		
		for( int i=0; i<rlen; i++ )
		{
			if( !a.isEmpty(i) )
			{
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				
				for( int j=apos; j<apos+alen && aix[j]<n-1; j++ )
				{
					int jix = aix[j];
					double corr = in.get(i, n-1);
					buffer1._sum        = aggVal.get(i, jix);
					buffer1._correction = aggVal.get(i, n-1);
					akplus.execute(buffer1, avals[j], corr);
					aggVal.set(i, jix, buffer1._sum);
					aggVal.set(i, n-1, buffer1._correction);
				}
			}
		}
	}

	private static void aggregateUnaryMatrixDense(MatrixBlock in, MatrixBlock out, AggType optype, ValueFunction vFn, IndexFunction ixFn, int rl, int ru) {
		final int n = in.clen;
		
		//note: due to corrections, even the output might be a large dense block
		DenseBlock a = in.getDenseBlock();
		DenseBlock c = out.getDenseBlock();
		
		switch( optype )
		{
			case KAHAN_SUM: { //SUM/TRACE via k+, 
				KahanObject kbuff = new KahanObject(0, 0);
				if( ixFn instanceof ReduceAll ) // SUM
					d_uakp(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWSUM
					d_uarkp(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLSUM
					d_uackp(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				else if( ixFn instanceof ReduceDiag ) //TRACE
					d_uakptrace(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				break;
			}
			case SUM:{
				if(a instanceof DenseBlockFP64DEDUP)
					throw new NotImplementedException();
				else if(ixFn instanceof ReduceAll) // SUM
					d_uap(a, c, n, rl, ru);
				else if(ixFn instanceof ReduceCol) // ROWSUM
					d_uarp(a, c, n, rl, ru);
				else if(ixFn instanceof ReduceRow) // COLSUM
					d_uacp(a, c, n, rl, ru);
				else if(ixFn instanceof ReduceDiag) // TRACE
					throw new NotImplementedException();
				break;
			}
			case KAHAN_SUM_SQ: { //SUM_SQ via k+,
				KahanObject kbuff = new KahanObject(0, 0);
				if( ixFn instanceof ReduceAll ) //SUM_SQ
					d_uasqkp(a, c, n, kbuff, (KahanPlusSq)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWSUM_SQ
					d_uarsqkp(a, c, n, kbuff, (KahanPlusSq)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLSUM_SQ
					d_uacsqkp(a, c, n, kbuff, (KahanPlusSq)vFn, rl, ru);
				break;
			}
			case CUM_KAHAN_SUM: { //CUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				d_ucumkp(in.getDenseBlock(), null, out.getDenseBlock(), n, kbuff, kplus, rl, ru);
				break;
			}
			case ROW_CUM_SUM: { //ROWCUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				d_urowcumkp(in.getDenseBlock(), null, out.getDenseBlock(), n, kbuff, kplus, rl, ru);
				break;
			}
			case CUM_PROD: { //CUMPROD
				d_ucumm(in.getDenseBlockValues(), null, out.getDenseBlockValues(), n, rl, ru);
				break;
			}
			case CUM_MIN:
			case CUM_MAX: {
				double init = (optype==AggType.CUM_MAX) ? Double.NEGATIVE_INFINITY:Double.POSITIVE_INFINITY;
				d_ucummxx(in.getDenseBlockValues(), null, out.getDenseBlockValues(), n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MIN: 
			case MAX: { //MAX/MIN
				double init = (optype==AggType.MAX) ? Double.NEGATIVE_INFINITY:Double.POSITIVE_INFINITY;
				if( ixFn instanceof ReduceAll ) // MIN/MAX
					d_uamxx(a, c, n, init, (Builtin)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWMIN/ROWMAX
					d_uarmxx(a, c, n, init, (Builtin)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLMIN/COLMAX
					d_uacmxx(a, c, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MAX_INDEX: {
				double init = Double.NEGATIVE_INFINITY;
				if( ixFn instanceof ReduceCol ) //ROWINDEXMAX
					d_uarimax(a, c, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MIN_INDEX: {
				double init = Double.POSITIVE_INFINITY;
				if( ixFn instanceof ReduceCol ) //ROWINDEXMIN
					d_uarimin(a, c, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MEAN: { //MEAN
				KahanObject kbuff = new KahanObject(0, 0);
				if( ixFn instanceof ReduceAll ) // MEAN
					d_uamean(a, c, n, kbuff, (Mean)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWMEAN
					d_uarmean(a, c, n, kbuff, (Mean)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLMEAN
					d_uacmean(a, c, n, kbuff, (Mean)vFn, rl, ru);
				break;
			}
			case VAR: { //VAR
				CmCovObject cbuff = new CmCovObject();
				if( ixFn instanceof ReduceAll ) //VAR
					d_uavar(a, c, n, cbuff, (CM)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWVAR
					d_uarvar(a, c, n, cbuff, (CM)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLVAR
					d_uacvar(a, c, n, cbuff, (CM)vFn, rl, ru);
				break;
			}
			case PROD: { //PROD
				long nnz = in.getNonZeros();
				if( ixFn instanceof ReduceAll ) // PROD
					d_uam(a, c, n, rl, ru , nnz);
				else if( ixFn instanceof ReduceCol ) //ROWPROD
					d_uarm(a, c, n, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLPROD
					d_uacm(a, c, n, rl, ru, nnz);
				break;
			}
			
			default:
				throw new DMLRuntimeException("Unsupported aggregation type: "+optype);
		}
	}

	private static void aggregateUnaryMatrixSparse(MatrixBlock in, MatrixBlock out, AggType optype, ValueFunction vFn, IndexFunction ixFn, int rl, int ru) {
		final int m = in.rlen;
		final int n = in.clen;
		
		//note: due to corrections, even the output might be a large dense block
		SparseBlock a = in.getSparseBlock();
		DenseBlock c = out.getDenseBlock();
		
		switch( optype )
		{
			case KAHAN_SUM: { //SUM via k+
				KahanObject kbuff = new KahanObject(0, 0);
				if( ixFn instanceof ReduceAll ) // SUM
					s_uakp(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWSUM
					s_uarkp(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLSUM
					s_uackp(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				else if( ixFn instanceof ReduceDiag ) //TRACE
					s_uakptrace(a, c, n, kbuff, (KahanPlus)vFn, rl, ru);
				break;
			}
			case SUM:{
				if( ixFn instanceof ReduceAll ) // SUM
					s_uap(a, c, n, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWSUM
					s_uarp(a, c, n, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLSUM
					s_uacp(a, c, n, rl, ru);
				else if( ixFn instanceof ReduceDiag ) //TRACE
					throw new NotImplementedException();
				break;
			}
			case KAHAN_SUM_SQ: { //SUM_SQ via k+
				KahanObject kbuff = new KahanObject(0, 0);
				if( ixFn instanceof ReduceAll ) //SUM_SQ
					s_uasqkp(a, c, n, kbuff, (KahanPlusSq)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWSUM_SQ
					s_uarsqkp(a, c, n, kbuff, (KahanPlusSq)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLSUM_SQ
					s_uacsqkp(a, c, n, kbuff, (KahanPlusSq)vFn, rl, ru);
				break;
			}
			case CUM_KAHAN_SUM: { //CUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				s_ucumkp(a, null, out.getDenseBlock(), m, n, kbuff, kplus, rl, ru);
				break;
			}
			case ROW_CUM_SUM: { //ROWCUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				s_urowcumkp(a, null, out.getDenseBlock(), m, n, kbuff, kplus, rl, ru);
				break;
			}
			case CUM_PROD: { //CUMPROD
				s_ucumm(a, null, out.getDenseBlockValues(), n, rl, ru);
				break;
			}
			case CUM_MIN:
			case CUM_MAX: {
				double init = (optype==AggType.CUM_MAX) ? Double.NEGATIVE_INFINITY:Double.POSITIVE_INFINITY;
				s_ucummxx(a, null, out.getDenseBlockValues(), n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MIN:
			case MAX: { //MAX/MIN
				double init = (optype==AggType.MAX) ? Double.NEGATIVE_INFINITY:Double.POSITIVE_INFINITY;
				if( ixFn instanceof ReduceAll ) // MIN/MAX
					s_uamxx(a, c, n, init, (Builtin)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWMIN/ROWMAX
					s_uarmxx(a, c, n, init, (Builtin)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLMIN/COLMAX
					s_uacmxx(a, c, m, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MAX_INDEX: {
				double init = Double.NEGATIVE_INFINITY;
				if( ixFn instanceof ReduceCol ) //ROWINDEXMAX
					s_uarimax(a, c, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MIN_INDEX: {
				double init = Double.POSITIVE_INFINITY;
				if( ixFn instanceof ReduceCol ) //ROWINDEXMAX
					s_uarimin(a, c, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			case MEAN: {
				KahanObject kbuff = new KahanObject(0, 0);
				if( ixFn instanceof ReduceAll ) // MEAN
					s_uamean(a, c, n, kbuff, (Mean)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWMEAN
					s_uarmean(a, c, n, kbuff, (Mean)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLMEAN
					s_uacmean(a, c, n, kbuff, (Mean)vFn, rl, ru);
				break;
			}
			case VAR: { //VAR
				CmCovObject cbuff = new CmCovObject();
				if( ixFn instanceof ReduceAll ) //VAR
					s_uavar(a, c, n, cbuff, (CM)vFn, rl, ru);
				else if( ixFn instanceof ReduceCol ) //ROWVAR
					s_uarvar(a, c, n, cbuff, (CM)vFn, rl, ru);
				else if( ixFn instanceof ReduceRow ) //COLVAR
					s_uacvar(a, c, n, cbuff, (CM)vFn, rl, ru);
				break;
			}
			case PROD: { //PROD
				if( ixFn instanceof ReduceAll ) // PROD
					s_uam(a, c, n, rl, ru );
				else if( ixFn instanceof ReduceCol ) // ROWPROD
					s_uarm(a, c, n, rl, ru );
				else if( ixFn instanceof ReduceRow ) // COLPROD
					s_uacm(a, c, n, rl, ru );
				break;
			}

			default:
				throw new DMLRuntimeException("Unsupported aggregation type: "+optype);
		}
	}

	private static void cumaggregateUnaryMatrixDense(MatrixBlock in, MatrixBlock out, AggType optype, ValueFunction vFn, double[] agg, int rl, int ru) {
		final int n = in.clen;
		
		DenseBlock da = in.getDenseBlock();
		DenseBlock dc = out.getDenseBlock();
		
		switch( optype ) {
			case CUM_KAHAN_SUM: { //CUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				d_ucumkp(da, agg, dc, n, kbuff, kplus, rl, ru);
				break;
			}
			case ROW_CUM_SUM: { //ROWCUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				d_urowcumkp(da, agg, dc, n, kbuff, kplus, rl, ru);
				break;
			}
			case CUM_SUM_PROD: { //CUMSUMPROD
				if( n != 2 )
					throw new DMLRuntimeException("Cumsumprod expects two-column input (n="+n+").");
				d_ucumkpp(da, agg, dc, rl, ru);
				break;
			}
			case CUM_PROD: { //CUMPROD
				if(!da.isContiguous())
					throw new NotImplementedException("Not implemented large block Cum Prod : " + optype);
				double[] a = in.getDenseBlockValues();
				double[] c = out.getDenseBlockValues();
				d_ucumm(a, agg, c, n, rl, ru);
				break;
			}
			case CUM_MIN:
			case CUM_MAX: {
				if(!da.isContiguous())
					throw new NotImplementedException("Not implemented large block Cum min or max " + optype);
				double[] a = in.getDenseBlockValues();
				double[] c = out.getDenseBlockValues();
				double init = (optype==AggType.CUM_MAX)? Double.NEGATIVE_INFINITY:Double.POSITIVE_INFINITY;
				d_ucummxx(a, agg, c, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			default:
				throw new DMLRuntimeException("Unsupported cumulative aggregation type: "+optype);
		}
	}

	private static void cumaggregateUnaryMatrixSparse(MatrixBlock in, MatrixBlock out, AggType optype, ValueFunction vFn, double[] agg, int rl, int ru) {
		final int m = in.rlen;
		final int n = in.clen;
		
		SparseBlock a = in.getSparseBlock();
		DenseBlock dc = out.getDenseBlock();
		double[] c = out.getDenseBlockValues();
		
		switch( optype ) {
			case CUM_KAHAN_SUM: { //CUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				s_ucumkp(a, agg, dc, m, n, kbuff, kplus, rl, ru);
				break;
			}
			case ROW_CUM_SUM: { //ROWCUMSUM
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				s_urowcumkp(a, agg, dc, m, n, kbuff, kplus, rl, ru);
				break;
			}
			case CUM_SUM_PROD: { //CUMSUMPROD
				if( n != 2 )
					throw new DMLRuntimeException("Cumsumprod expects two-column input (n="+n+").");
				s_ucumkpp(a, agg, dc, rl, ru);
				break;
			}
			case CUM_PROD: { //CUMPROD
				s_ucumm(a, agg, c, n, rl, ru);
				break;
			}
			case CUM_MIN:
			case CUM_MAX: {
				double init = (optype==AggType.CUM_MAX) ? Double.NEGATIVE_INFINITY:Double.POSITIVE_INFINITY;
				s_ucummxx(a, agg, c, n, init, (Builtin)vFn, rl, ru);
				break;
			}
			default:
				throw new DMLRuntimeException("Unsupported cumulative aggregation type: "+optype);
		}
	}

	private static MatrixBlock aggregateUnaryMatrixEmpty(MatrixBlock in, MatrixBlock out, AggType optype, IndexFunction ixFn) {
		//handle all full aggregates over matrices with zero rows or columns
		if( ixFn instanceof ReduceAll && (in.getNumRows() == 0 || in.getNumColumns() == 0) ) {
			double val = Double.NaN;
			switch( optype ) {
				case PROD:         val = 1; break;
				case SUM: 
				case SUM_SQ:
				case KAHAN_SUM:
				case ROW_CUM_SUM:
				case KAHAN_SUM_SQ: val = 0; break;
				case MIN:          val = Double.POSITIVE_INFINITY; break;
				case MAX:          val = Double.NEGATIVE_INFINITY; break;
				case MEAN:
				case VAR:
				case MIN_INDEX:
				case MAX_INDEX:
				default:           val = Double.NaN; break;
			}
			out.set(0, 0, val);
			return out;
		}
		
		//handle pseudo sparse-safe operations over empty inputs 
		if(optype == AggType.KAHAN_SUM || optype == AggType.KAHAN_SUM_SQ
				|| optype == AggType.SUM || optype == AggType.SUM_SQ 
				|| optype == AggType.MIN || optype == AggType.MAX || optype == AggType.PROD
				|| optype == AggType.CUM_KAHAN_SUM || optype == AggType.ROW_CUM_SUM || optype == AggType.CUM_PROD
				|| optype == AggType.CUM_MIN || optype == AggType.CUM_MAX)
		{
			return out;
		}
		
		//compute result based on meta data only
		switch( optype )
		{
			case MAX_INDEX: {
				if( ixFn instanceof ReduceCol ) { //ROWINDEXMAX
					for(int i=0; i<out.rlen; i++) {
						out.set(i, 0, in.clen); //maxindex
					}
				}
				break;
			}
			case MIN_INDEX: {
				if( ixFn instanceof ReduceCol ) //ROWINDEXMIN
					for(int i=0; i<out.rlen; i++) {
						out.set(i, 0, in.clen); //minindex
					}
				break;
			}
			case MEAN: {
				if( ixFn instanceof ReduceAll ) // MEAN
					out.set(0, 1, in.rlen*in.clen); //count
				else if( ixFn instanceof ReduceCol ) //ROWMEAN
					for( int i=0; i<in.rlen; i++ ) //0-sum and 0-correction 
						out.set(i, 1, in.clen); //count
				else if( ixFn instanceof ReduceRow ) //COLMEAN
					for( int j=0; j<in.clen; j++ ) //0-sum and 0-correction 
						out.set(1, j, in.rlen); //count
				break;
			}
			case VAR: {
				// results: { var | mean, count, m2 correction, mean correction }
				if( ixFn instanceof ReduceAll ) //VAR
					out.set(0, 2, in.rlen*in.clen); //count
				else if( ixFn instanceof ReduceCol ) //ROWVAR
					for( int i=0; i<in.rlen; i++ )
						out.set(i, 2, in.clen); //count
				else if( ixFn instanceof ReduceRow ) //COLVAR
					for( int j=0; j<in.clen; j++ )
						out.set(2, j, in.rlen); //count
				break;
			}
			case CUM_SUM_PROD:{
				break;
			}
			default:
				throw new DMLRuntimeException("Unsupported aggregation type: "+optype);
		}
		
		return out;
	}
	
	
	////////////////////////////////////////////
	// core aggregation functions             //
	////////////////////////////////////////////

	/**
	 * SUM, opcode: uak+, dense input. 
	 * 
	 * @param a Input block
	 * @param c Output block
	 * @param n Input block number of columns
	 * @param kbuff Kahn addition buffer
	 * @param kplus Kahn plus operator
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uakp( DenseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		if(a instanceof DenseBlockFP64DEDUP)
			uakpDedup((DenseBlockFP64DEDUP) a, c, n, kbuff, kplus, rl, ru);
		else {
			final int bil = a.index(rl);
			final int biu = a.index(ru - 1);
			for (int bi = bil; bi <= biu; bi++) {
				int lpos = (bi == bil) ? a.pos(rl) : 0;
				int len = (bi == biu) ? a.pos(ru - 1) - lpos + n : a.blockSize(bi) * n;
				sum(a.valuesAt(bi), lpos, len, kbuff, kplus);
			}
		}
		c.set(kbuff);
	}

	private static void d_uap(DenseBlock a, DenseBlock c, int n, int rl, int ru) {
		final int bil = a.index(rl);
		final int biu = a.index(ru - 1);
		double runningSum = 0.0;
		for(int bi = bil; bi <= biu; bi++) { // for each block
			final int lpos = (bi == bil) ? a.pos(rl) : 0;
			final int len = (bi == biu) ? a.pos(ru - 1) - lpos + n : a.blockSize(bi) * n;
			final double[] aVals = a.valuesAt(bi); // get all the values
			for(int i = lpos; i < lpos + len; i++) // all values in the block
				runningSum += aVals[i];
		}
		c.set(runningSum);
	}
	
	/**
	 * ROWSUM, opcode: uark+, dense input.
	 * 
	 * @param a Input matrix to rowSum
	 * @param c Output matrix to set the row sums into
	 * @param n The number of columns in the output
	 * @param kbuff kahn buffer
	 * @param kplus Kahn plus operator
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uarkp( DenseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) 
	{
		if(a instanceof DenseBlockFP64DEDUP)
			uarkpDedup((DenseBlockFP64DEDUP) a, c, n, kbuff, kplus, rl, ru);
		else {
			for (int i = rl; i < ru; i++) {
				kbuff.set(0, 0); //reset buffer
				sum(a.values(i), a.pos(i), n, kbuff, kplus);
				c.set(i, kbuff);
			}
		}
	}

	private static void d_uarp(DenseBlock a, DenseBlock c, int n, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			final int off = a.pos(i);
			final double[] aVals = a.values(i);
			double tmp = 0.0;
			for(int col = off; col< off + n; col++)
				tmp += aVals[col];
			c.set(i, 0, tmp);
		}
	}

	/**
	 * COLSUM, opcode: uack+, dense input.
	 * 
	 * @param a     Input block
	 * @param c     Output block
	 * @param n     number of column in the input
	 * @param kbuff Kahn buffer
	 * @param kplus Kahn plus operator
	 * @param rl    row lower index
	 * @param ru    row upper index
	 */
	private static void d_uackp( DenseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		if(a instanceof DenseBlockFP64DEDUP)
			uackpDedup((DenseBlockFP64DEDUP) a, c, n, kbuff, kplus, rl, ru);
		else {
			for( int i=rl; i<ru; i++ )
				sumAgg( a.values(i), c, a.pos(i), n, kbuff, kplus );
		}
	}

	private static void d_uacp( DenseBlock a, DenseBlock c, int n, int rl, int ru ) {
		// Output always a vector.
		double[] cVals = c.values(0);
		for(int r = rl; r < ru; r++){
			int apos = a.pos(r);
			double[] avals = a.values(r);
			for(int i = 0; i < n; i++){
				cVals[i] += avals[apos + i];
			}
		}
	}

	/**
	 * SUM_SQ, opcode: uasqk+, dense input.
	 *
	 * @param a Array of values to square & sum.
	 * @param c Output array to store sum and correction factor.
	 * @param n Number of values per row.
	 * @param kbuff A KahanObject to hold the current sum and
	 *              correction factor for the Kahan summation
	 *              algorithm.
	 * @param kplusSq A KahanPlusSq object to perform summation of
	 *                squared values.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void d_uasqkp(DenseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlusSq kplusSq, int rl, int ru)
	{
		final int bil = a.index(rl);
		final int biu = a.index(ru-1);
		for(int bi=bil; bi<=biu; bi++) {
			int lpos = (bi==bil) ? a.pos(rl) : 0;
			int len = (bi==biu) ? a.pos(ru-1)-lpos+n : a.blockSize(bi)*n;
			sum(a.valuesAt(bi), lpos, len, kbuff, kplusSq);
		}
		c.set(kbuff);
	}

	/**
	 * ROWSUM_SQ, opcode: uarsqk+, dense input.
	 *
	 * @param a Array of values to square & sum row-wise.
	 * @param c Output array to store sum and correction factor
	 *          for each row.
	 * @param n Number of values per row.
	 * @param kbuff A KahanObject to hold the current sum and
	 *              correction factor for the Kahan summation
	 *              algorithm.
	 * @param kplusSq A KahanPlusSq object to perform summation of
	 *                squared values.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void d_uarsqkp(DenseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlusSq kplusSq, int rl, int ru) {
		for (int i=rl; i<ru; i++) {
			kbuff.set(0, 0); //reset buffer
			sum(a.values(i), a.pos(i), n, kbuff, kplusSq);
			c.set(i, kbuff);
		}
	}

	/**
	 * COLSUM_SQ, opcode: uacsqk+, dense input.
	 *
	 * @param a Array of values to square & sum column-wise.
	 * @param c Output array to store sum and correction factor
	 *          for each column.
	 * @param n Number of values per row.
	 * @param kbuff A KahanObject to hold the current sum and
	 *              correction factor for the Kahan summation
	 *              algorithm.
	 * @param kplusSq A KahanPlusSq object to perform summation of
	 *                squared values.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void d_uacsqkp(DenseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlusSq kplusSq, int rl, int ru) {
		for( int i=rl; i<ru; i++ )
			sumAgg(a.values(i), c, a.pos(i), n, kbuff, kplusSq);
	}

	/**
	 * CUMSUM, opcode: ucumk+, dense input.
	 * 
	 * @param a ?
	 * @param agg ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kplus ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_ucumkp( DenseBlock a, double[] agg, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		//init current row sum/correction arrays w/ neutral 0
		DenseBlock csums = DenseBlockFactory.createDenseBlock(2, n);
		if( agg != null )
			csums.set(0, agg);
		//scan once and compute prefix sums
		for( int i=rl; i<ru; i++ ) {
			sumAgg( a.values(i), csums, a.pos(i), n, kbuff, kplus );
			c.set(i, csums.values(0));
		}
	}

	/**
	 * ROWCUMSUM, opcode: urowcumk+, dense input.
	 *
	 * @param a input matrix
	 * @param agg initial array
	 * @param c output matrix
	 * @param n number of rows
	 * @param kbuff collects sum
	 * @param kplus sums up
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_urowcumkp( DenseBlock a, double[] agg, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		//row-wise cumulative sum w/ optional row offsets
		for (int i = rl; i < ru; i++) {
			double start = 0.0;
			int localRow = i - rl;
			if (agg != null) {
				if (localRow >= 0 && localRow < agg.length) {
					start = agg[localRow];
				}
			}
			kbuff.set(start, 0);
			//compute cumulative sum over row
			for (int j = 0; j < n; j++) {
				double val = a.get(i, j);
				kplus.execute2(kbuff, val);
				c.set(i, j, kbuff._sum);
			}

		}
	}
	
	/**
	 * CUMSUMPROD, opcode: ucumk+*, dense input.
	 * 
	 * @param a ?
	 * @param agg ?
	 * @param c ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_ucumkpp( DenseBlock a, double[] agg, DenseBlock c, int rl, int ru ) {
		//init current row sum/correction arrays w/ neutral 0
		double sum = (agg != null) ? agg[0] : 0;
		//scan once and compute prefix sums
		double[] avals = a.valuesAt(0);
		double[] cvals = c.values(0);
		for( int i=rl, ix=rl*2; i<ru; i++, ix+=2 ) {
			sum = cvals[i] = avals[ix] + avals[ix+1] * sum;
		}
	}
	
	/**
	 * CUMPROD, opcode: ucum*, dense input.
	 * 
	 * @param a ?
	 * @param agg ?
	 * @param c ?
	 * @param n ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_ucumm( double[] a, double[] agg, double[] c, int n, int rl, int ru ) 
	{
		//init current row product array w/ neutral 1
		double[] cprods = (agg!=null) ? agg : new double[ n ]; 
		if( agg == null )
			Arrays.fill(cprods, 1);
		
		//scan once and compute prefix products
		for( int i=rl, aix=rl*n; i<ru; i++, aix+=n ) {
			productAgg( a, cprods, aix, 0, n );
			System.arraycopy(cprods, 0, c, aix, n);
		}
	}
	
	/**
	 * CUMMIN/CUMMAX, opcode: ucummin/ucummax, dense input.
	 * 
	 * @param a ?
	 * @param agg ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_ucummxx( double[] a, double[] agg, double[] c, int n, double init, Builtin builtin, int rl, int ru )
	{
		//init current row min/max array w/ extreme value 
		double[] cmxx = (agg!=null) ? agg : new double[ n ]; 
		if( agg == null )
			Arrays.fill(cmxx, init);
				
		//scan once and compute prefix min/max
		for( int i=rl, aix=rl*n; i<ru; i++, aix+=n ) {
			builtinAgg( a, cmxx, aix, n, builtin );
			System.arraycopy(cmxx, 0, c, aix, n);
		}
	}
	/**
	 * TRACE, opcode: uaktrace 
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kplus ?
	 * @param rl ?
	 * @param ru ?
	 */
	private static void d_uakptrace( DenseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) 
	{
		//aggregate diag (via ix=n+1)
		for( int i=rl; i<ru; i++ )
			kplus.execute2(kbuff, a.get(i, i));
		c.set(kbuff);
	}
	
	/**
	 * MIN/MAX, opcode: uamin/uamax, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uamxx( DenseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		double tmp = init;
		final int bil = a.index(rl);
		final int biu = a.index(ru-1);
		for(int bi=bil; bi<=biu; bi++) {
			int lpos = (bi==bil) ? a.pos(rl) : 0;
			int len = (bi==biu) ? a.pos(ru-1)-lpos+n : a.blockSize(bi)*n;
			tmp = builtin(a.valuesAt(bi), lpos, tmp, len, builtin);
		}
		c.set(0, 0, tmp);
	}
	
	/**
	 * ROWMIN/ROWMAX, opcode: uarmin/uarmax, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uarmxx( DenseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		for( int i=rl; i<ru; i++ )
			c.set(i, 0, builtin(a.values(i), a.pos(i), init, n, builtin));
	}
	
	/**
	 * COLMIN/COLMAX, opcode: uacmin/uacmax, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uacmxx( DenseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		//init output (base for incremental agg)
		c.set(init);
		//execute builtin aggregate
		double[] lc = c.values(0); //guaranteed single row
		for( int i=rl; i<ru; i++ )
			builtinAgg( a.values(i), lc, a.pos(i), n, builtin );
	}

	/**
	 * ROWINDEXMAX, opcode: uarimax, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uarimax( DenseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		if( n <= 0 )
			throw new DMLRuntimeException("rowIndexMax undefined for ncol="+n);
		for( int i=rl; i<ru; i++ ) {
			int maxindex = indexmax(a.values(i), a.pos(i), init, n, builtin);
			c.set(i, 0, (double)maxindex + 1);
			c.set(i, 1, a.get(i, maxindex)); //max value
		}
	}
	
	/**
	 * ROWINDEXMIN, opcode: uarimin, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uarimin( DenseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		if( n <= 0 )
			throw new DMLRuntimeException("rowIndexMin undefined for ncol="+n);
		for( int i=rl; i<ru; i++ ) {
			int minindex = indexmin(a.values(i), a.pos(i), init, n, builtin);
			c.set(i, 0, (double)minindex + 1);
			c.set(i, 1, a.get(i, minindex)); //min value
		}
	}
	
	/**
	 * MEAN, opcode: uamean, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kmean ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uamean( DenseBlock a, DenseBlock c, int n, KahanObject kbuff, Mean kmean, int rl, int ru )
	{
		final int bil = a.index(rl);
		final int biu = a.index(ru-1);
		int tlen = 0;
		for(int bi=bil; bi<=biu; bi++) {
			int lpos = (bi==bil) ? a.pos(rl) : 0;
			int len = (bi==biu) ? a.pos(ru-1)-lpos+n : a.blockSize(bi)*n;
			mean(a.valuesAt(bi), lpos, len, 0, kbuff, kmean);
			tlen += len;
		}
		c.set(0, 0, kbuff._sum);
		c.set(0, 1, tlen);
		c.set(0, 2, kbuff._correction);
	}
	
	/**
	 * ROWMEAN, opcode: uarmean, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kmean ?
	 * @param rl ?
	 * @param ru ?
	 */
	private static void d_uarmean( DenseBlock a, DenseBlock c, int n, KahanObject kbuff, Mean kmean, int rl, int ru )
	{
		for( int i=rl; i<ru; i++ ) {
			kbuff.set(0, 0); //reset buffer
			mean(a.values(i), a.pos(i), n, 0, kbuff, kmean);
			c.set(i, 0, kbuff._sum);
			c.set(i, 1, n);
			c.set(i, 2, kbuff._correction);
		}
	}
	
	/**
	 * COLMEAN, opcode: uacmean, dense input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kmean ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uacmean( DenseBlock a, DenseBlock c, int n, KahanObject kbuff, Mean kmean, int rl, int ru ) {
		//execute builtin aggregate
		for( int i=rl; i<ru; i++ )
			meanAgg( a.values(i), c, a.pos(i), n, kbuff, kmean );
	}

	/**
	 * VAR, opcode: uavar, dense input.
	 *
	 * @param a Array of values.
	 * @param c Output array to store variance, mean, count,
	 *          m2 correction factor, and mean correction factor.
	 * @param n Number of values per row.
	 * @param cbuff A CM_COV_Object to hold various intermediate
	 *              values for the variance calculation.
	 * @param cm A CM object of type Variance to perform the variance
	 *           calculation.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void d_uavar(DenseBlock a, DenseBlock c, int n, CmCovObject cbuff, CM cm, int rl, int ru) {
		final int bil = a.index(rl);
		final int biu = a.index(ru-1);
		for(int bi=bil; bi<=biu; bi++) {
			int lpos = (bi==bil) ? a.pos(rl) : 0;
			int len = (bi==biu) ? a.pos(ru-1)-lpos+n : a.blockSize(bi)*n;
			var(a.valuesAt(bi), lpos, len, cbuff, cm);
		}
		// store results: { var | mean, count, m2 correction, mean correction }
		c.set(0, 0, cbuff.getRequiredResult(AggregateOperationTypes.VARIANCE));
		c.set(0, 1, cbuff.mean._sum);
		c.set(0, 2, cbuff.w);
		c.set(0, 3, cbuff.m2._correction);
		c.set(0, 4, cbuff.mean._correction);
	}

	/**
	 * ROWVAR, opcode: uarvar, dense input.
	 *
	 * @param a Array of values.
	 * @param c Output array to store variance, mean, count,
	 *          m2 correction factor, and mean correction factor
	 *          for each row.
	 * @param n Number of values per row.
	 * @param cbuff A CM_COV_Object to hold various intermediate
	 *              values for the variance calculation.
	 * @param cm A CM object of type Variance to perform the variance
	 *           calculation.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void d_uarvar(DenseBlock a, DenseBlock c, int n, CmCovObject cbuff, CM cm, int rl, int ru) {
		// calculate variance for each row
		for (int i=rl; i<ru; i++) {
			cbuff.reset(); // reset buffer for each row
			var(a.values(i), a.pos(i), n, cbuff, cm);
			// store row results: { var | mean, count, m2 correction, mean correction }
			c.set(i, 0, cbuff.getRequiredResult(AggregateOperationTypes.VARIANCE));
			c.set(i, 1, cbuff.mean._sum);
			c.set(i, 2, cbuff.w);
			c.set(i, 3, cbuff.m2._correction);
			c.set(i, 4, cbuff.mean._correction);
		}
	}

	/**
	 * COLVAR, opcode: uacvar, dense input.
	 *
	 * @param a Array of values.
	 * @param c Output array to store variance, mean, count,
	 *          m2 correction factor, and mean correction factor
	 *          for each column.
	 * @param n Number of values per row.
	 * @param cbuff A CM_COV_Object to hold various intermediate
	 *              values for the variance calculation.
	 * @param cm A CM object of type Variance to perform the variance
	 *           calculation.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void d_uacvar(DenseBlock a, DenseBlock c, int n, CmCovObject cbuff, CM cm, int rl, int ru) {
		// calculate variance for each column incrementally
		for (int i=rl; i<ru; i++)
			varAgg(a.values(i), c, a.pos(i), n, cbuff, cm);
	}

	/**
	 * PROD, opcode: ua*, dense input.
	 * 
	 * @param a dense input block
	 * @param c dense output block
	 * @param n number columns in input
	 * @param rl row lower index
	 * @param ru row upper index
	 * @param nnz number non zero values in matrix block
	 */
	private static void d_uam( DenseBlock a, DenseBlock c, int n, int rl, int ru, long nnz ) {
		if(nnz < (long)a.getDim(0) * n)
			return; // if the input contains a zero the return is zero.
		final int bil = a.index(rl);
		final int biu = a.index(ru-1);
		double tmp = 1;
		for(int bi=bil; bi<=biu; bi++) {
			int lpos = (bi==bil) ? a.pos(rl) : 0;
			int len = (bi==biu) ? a.pos(ru-1)-lpos+n : a.blockSize(bi)*n;
			tmp *= product( a.valuesAt(bi), lpos, len );
		}
		c.set(0, 0, tmp);
	}

	/**
	 * ROWPROD, opcode: uar*, dense input.
	 * 
	 * @param a dense input block
	 * @param c dense output block
	 * @param n number columns in input
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void d_uarm( DenseBlock a, DenseBlock c, int n, int rl, int ru ) {
		double[] lc = c.values(0);
		for( int i=rl; i<ru; i++ )
			lc[i] = product(a.values(i), a.pos(i), n);
	}
	
	/**
	 * COLPROD, opcode: uac*, dense input.
	 * 
	 * @param a dense input block
	 * @param c dense output block
	 * @param n number columns in input
	 * @param rl row lower index
	 * @param ru row upper index
	 * @param nnz number non zeros in matrix
	 */
	private static void d_uacm( DenseBlock a, DenseBlock c, int n, int rl, int ru, long nnz ) {
		double[] lc = c.set(1).values(0); // guaranteed single row
		if(nnz < (double) a.getDim(0) * n * 0.90) { // 90% sparsity.
			// process with early termination if a zero is hit in all columns.
			final int blockz = 32;
			for(int block = rl; block < ru; block += blockz) {
				// process blocks of rows before checking for zero
				final int be = Math.min(block + blockz, ru);
				for(int i = block; i < be; i++)
					for(int ci = 0, ai = i * n; ci < n; ci++, ai++)
						lc[ci] *= a.values(i)[ai];

				// Zeros check
				int nz = 0;
				for(double d : lc)
					nz += d != 0 ? 1 : 0;
				if(nz == 0)
					return;
			}
			if(!NAN_AWARENESS)
				correctNan(lc);
		}
		else {// full processing
			for(int i = rl; i < ru; i++)
				for(int ci = 0, ai = i * n; ci < n; ci++, ai++)
					lc[ci] *= a.values(i)[ai];
			if(!NAN_AWARENESS)
				correctNan(lc);
		}
	}

	protected static void correctNan(double[] res) {
		// since there is no nan values every in a dictionary, we exploit that
		// nan oly occur if we multiplied infinity with 0.
		for(int j = 0; j < res.length; j++)
			if(Double.isNaN(res[j]))
				res[j] = 0;
	}
	
	/**
	 * SUM, opcode: uak+, sparse input. 
	 * 
	 * @param a Sparse input block
	 * @param c dense output block
	 * @param n number columns in input
	 * @param kbuff Kahn buffer
	 * @param kplus Kahn operator
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uakp( SparseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru )
	{
		if( a.isContiguous() ) {
			sum(a.values(rl), a.pos(rl), (int)a.size(rl, ru), kbuff, kplus);
		}
		else {
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) )
					sum(a.values(i), a.pos(i), a.size(i), kbuff, kplus);
			}
		}
		c.set(kbuff);
	}
	

	private static void s_uap(SparseBlock a, DenseBlock c, int n, int rl, int ru) {
		double tmp = 0.0;
		if(a.isContiguous()) {
			final double[] aVal = a.values(rl);
			final int s = a.pos(rl);
			final long e = a.size(rl, ru);
			for(int i = s; i < e; i++)
				tmp += aVal[i];
		}
		else {
			for(int i = rl; i < ru; i++)
				tmp  += s_sumRow(a, i);
		}
		c.set(tmp);
	}

	/**
	 * ROWSUM, opcode: uark+, sparse input.
	 * 
	 * @param a     Spasrse block to row sum on
	 * @param c     Dense output block
	 * @param n     Number of column in the input block
	 * @param kbuff Kahan buffer
	 * @param kplus Kahan plus operator
	 * @param rl    Row lower index
	 * @param ru    Row upper index
	 */
	private static void s_uarkp( SparseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		//compute row aggregates
		for( int i=rl; i<ru; i++ ) {
			if( a.isEmpty(i) ) continue;
			kbuff.set(0, 0); //reset buffer
			sum( a.values(i), a.pos(i), a.size(i), kbuff, kplus );
			c.set(i, kbuff);
		}
	}

	private static void s_uarp(SparseBlock a, DenseBlock c, int n, int rl, int ru) {
		// compute row aggregates
		for(int i = rl; i < ru; i++)
			c.set(i, 0, s_sumRow(a, i));
	}

	private static double s_sumRow(SparseBlock a, int r) {
			if(a.isEmpty(r))
				return 0.0;
		double tmp = 0.0;
		final double[] aVal = a.values(r);
		final int aPos = a.pos(r);
		final int aEnd = aPos + a.size(r);
		for(int j = aPos; j < aEnd; j++)
			tmp += aVal[j];
		return tmp;
	}

	
	/**
	 * COLSUM, opcode: uack+, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kplus ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uackp( SparseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) 
	{
		//compute column aggregates
		if( a.isContiguous() ) {
			sumAgg( a.values(rl), c, a.indexes(rl), a.pos(rl), (int)a.size(rl, ru), n, kbuff, kplus );
		}
		else {
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) )
					sumAgg( a.values(i), c, a.indexes(i), a.pos(i), a.size(i), n, kbuff, kplus );
			}
		}
	}

	private static void s_uacp(SparseBlock a, DenseBlock c, int n, int rl, int ru) {
		final double[] cVal = c.values(0);
		if(a.isContiguous())
			sumAgg(a.values(rl), cVal, a.indexes(rl), a.pos(rl), (int) a.size(rl, ru), n);
		else
			for(int i = rl; i < ru; i++)
				if(!a.isEmpty(i))
					sumAgg(a.values(i), cVal, a.indexes(i), a.pos(i), a.size(i), n);
	}

	/**
	 * SUM_SQ, opcode: uasqk+, sparse input.
	 *
	 * @param a Sparse array of values to square & sum.
	 * @param c Output array to store sum and correction factor.
	 * @param n Number of values per row.
	 * @param kbuff A KahanObject to hold the current sum and
	 *              correction factor for the Kahan summation
	 *              algorithm.
	 * @param kplusSq A KahanPlusSq object to perform summation of
	 *                squared values.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void s_uasqkp(SparseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlusSq kplusSq, int rl, int ru )
	{
		if( a.isContiguous() ) {
			sum(a.values(rl), a.pos(rl), (int)a.size(rl, ru), kbuff, kplusSq);
		}
		else {
			for (int i=rl; i<ru; i++) {
				if (!a.isEmpty(i))
					sum(a.values(i), a.pos(i), a.size(i), kbuff, kplusSq);
			}
		}
		c.set(kbuff);
	}

	/**
	 * ROWSUM_SQ, opcode: uarsqk+, sparse input.
	 *
	 * @param a Sparse array of values to square & sum row-wise.
	 * @param c Output array to store sum and correction factor
	 *          for each row.
	 * @param n Number of values per row.
	 * @param kbuff A KahanObject to hold the current sum and
	 *              correction factor for the Kahan summation
	 *              algorithm.
	 * @param kplusSq A KahanPlusSq object to perform summation of
	 *                squared values.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void s_uarsqkp(SparseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlusSq kplusSq, int rl, int ru )
	{
		//compute row aggregates
		for (int i=rl; i<ru; i++) {
			if( a.isEmpty(i) ) continue;
			kbuff.set(0, 0); //reset buffer
			sum(a.values(i), a.pos(i), a.size(i), kbuff, kplusSq);
			c.set(i, kbuff);
		}
	}

	/**
	 * COLSUM_SQ, opcode: uacsqk+, sparse input.
	 *
	 * @param a Sparse array of values to square & sum column-wise.
	 * @param c Output array to store sum and correction factor
	 *          for each column.
	 * @param n Number of values per row.
	 * @param kbuff A KahanObject to hold the current sum and
	 *              correction factor for the Kahan summation
	 *              algorithm.
	 * @param kplusSq A KahanPlusSq object to perform summation of
	 *                squared values.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void s_uacsqkp(SparseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlusSq kplusSq, int rl, int ru )
	{
		//compute column aggregates
		if( a.isContiguous() ) {
			sumAgg(a.values(rl), c, a.indexes(rl), a.pos(rl), (int)a.size(rl, ru), n, kbuff, kplusSq);
		}
		else {
			for (int i=rl; i<ru; i++) {
				if (!a.isEmpty(i))
					sumAgg(a.values(i), c, a.indexes(i), a.pos(i), a.size(i), n, kbuff, kplusSq);
			}
		}
	}

	/**
	 * CUMSUM, opcode: ucumk+, sparse input.
	 * 
	 * @param a ?
	 * @param agg ?
	 * @param c ?
	 * @param m ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kplus ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_ucumkp( SparseBlock a, double[] agg, DenseBlock c, int m, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru )
	{
		//init current row sum/correction arrays w/ neutral 0
		DenseBlock csums = DenseBlockFactory.createDenseBlock(2, n);
		if( agg != null )
			csums.set(0, agg);
		//scan once and compute prefix sums
		for( int i=rl; i<ru; i++ ) {
			if( !a.isEmpty(i) )
				sumAgg( a.values(i), csums, a.indexes(i), a.pos(i), a.size(i), n, kbuff, kplus );
			//always copy current sum (not sparse-safe)
			c.set(i, csums.values(0));
		}
	}

	/**
	 * ROWCUMSUM, opcode: urowcumk+, sparse input.
	 *
	 * @param a input matrix
	 * @param agg intial array
	 * @param c output matrix
	 * @param m number of columns
	 * @param n number of rows
	 * @param kbuff collects sum
	 * @param kplus sums up
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_urowcumkp(SparseBlock a, double[] agg, DenseBlock c, int m, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru) {
		//scan rows and compute row-wise prefix sums
		for (int i = rl; i < ru; i++) {
			double start = 0.0;
			int localRow = i - rl;
			if (agg != null && localRow >= 0 && localRow < agg.length)
				start = agg[localRow];
			if (!a.isEmpty(i)) {
				double[] ain = a.values(i);
				int[] aix = a.indexes(i);
				int apos = a.pos(i);
				int alen = a.size(i);
				kbuff.set(start, 0);
				int sparseIdx = 0;
				//prefix sum over sparse row
				for (int j = 0; j < n; j++) {
					if (sparseIdx < alen && aix[apos + sparseIdx] == j) {
						kplus.execute2(kbuff, ain[apos + sparseIdx]);
						start = kbuff._sum;
						sparseIdx++;
					}
					c.set(i, j, start);
				}
			}
			else {
				//fill empty row with start value
				for (int j = 0; j < n; j++)
					c.set(i, j, start);
			}
		}
	}
	
	
	/**
	 * CUMSUMPROD, opcode: ucumk+*, dense input.
	 * 
	 * @param a sparse block
	 * @param agg ?
	 * @param c ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_ucumkpp( SparseBlock a, double[] agg, DenseBlock c, int rl, int ru ) {
		//init current row sum/correction arrays w/ neutral 0
		double sum = (agg != null) ? agg[0] : 0;
		//scan once and compute prefix sums
		double[] cvals = c.values(0);
		for( int i=rl; i<ru; i++ ) {
			if( a.isEmpty(i) )
				sum = cvals[i] = 0;
			else if( a.size(i) == 2 ) {
				double[] avals = a.values(i); int apos = a.pos(i);
				sum = cvals[i] = avals[apos] + avals[apos+1] * sum;
			}
			else //fallback
				sum = cvals[i] = a.get(i,0) + a.get(i,1) * sum;
		}
	}
	
	/**
	 * CUMPROD, opcode: ucum*, sparse input.
	 * 
	 * @param a ?
	 * @param agg ?
	 * @param c ?
	 * @param n ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_ucumm( SparseBlock a, double[] agg, double[] c, int n, int rl, int ru )
	{
		//init current row prod arrays w/ neutral 1
		double[] cprod = (agg!=null) ? agg : new double[ n ]; 
		if( agg == null )
			Arrays.fill(cprod, 1);
		
		//init count arrays (helper, see correction)
		int[] cnt = new int[ n ]; 

		//scan once and compute prefix products
		for( int i=rl, ix=rl*n; i<ru; i++, ix+=n )
		{
			//multiply row of non-zero elements
			if( !a.isEmpty(i) ) {
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				productAgg( avals, cprod, aix, apos, 0, alen );
				countAgg( avals, cnt, aix, apos, alen );
			}

			//correction (not sparse-safe and cumulative)
			//note: we need to determine if there are only nnz in a column
			for( int j=0; j<n; j++ )
				if( cnt[j] < i+1 ) //no dense column
					cprod[j] *= 0;
			
			//always copy current sum (not sparse-safe)
			System.arraycopy(cprod, 0, c, ix, n);
		}
	}
	
	/**
	 * CUMMIN/CUMMAX, opcode: ucummin/ucummax, sparse input.
	 * 
	 * @param a ?
	 * @param agg ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_ucummxx( SparseBlock a, double[] agg, double[] c, int n, double init, Builtin builtin, int rl, int ru ) 
	{
		//init current row min/max array w/ extreme value 
		double[] cmxx = (agg!=null) ? agg : new double[ n ]; 
		if( agg == null )
			Arrays.fill(cmxx, init);
				
		//init count arrays (helper, see correction)
		int[] cnt = new int[ n ]; 

		//compute column aggregates min/max
		for( int i=rl, ix=rl*n; i<ru; i++, ix+=n )
		{
			if( !a.isEmpty(i) ) {
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				builtinAgg( avals, cmxx, aix, apos, alen, builtin );
				countAgg( avals, cnt, aix, apos, alen );
			}
			
			//correction (not sparse-safe and cumulative)
			//note: we need to determine if there are only nnz in a column
			for( int j=0; j<n; j++ )
				if( cnt[j] < i+1 ) //no dense column
					cmxx[j] = builtin.execute(cmxx[j], 0);
			
			//always copy current sum (not sparse-safe)
			System.arraycopy(cmxx, 0, c, ix, n);
		}
	}
	
	/**
	 * TRACE, opcode: uaktrace, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kplus ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uakptrace( SparseBlock a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		for( int i=rl; i<ru; i++ )
			if( !a.isEmpty(i) ) 
				kplus.execute2(kbuff, a.get(i,i));
		c.set(kbuff);
	}
	
	/**
	 * MIN/MAX, opcode: uamin/uamax, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uamxx( SparseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru )
	{
		double ret = init; //keep init val
		
		if( a.isContiguous() ) {
			int alen = (int) a.size(rl, ru);
			double val = builtin(a.values(rl), a.pos(rl), init, alen, builtin);
			ret = builtin.execute(ret, val);
			//correction (not sparse-safe)
			ret = (alen<(ru-rl)*n) ? builtin.execute(ret, 0) : ret;
		}
		else {
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) ) {
					double lval = builtin(a.values(i), a.pos(i), init, a.size(i), builtin);
					ret = builtin.execute(ret, lval);
				}		
				//correction (not sparse-safe)
				if( a.size(i) < n )
					ret = builtin.execute(ret, 0); 
			}
		}
		c.set(0, 0, ret);
	}
	
	/**
	 * ROWMIN/ROWMAX, opcode: uarmin/uarmax, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uarmxx( SparseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		//init result (for empty rows)
		c.set(rl, ru, 0, 1, init); //not sparse-safe
		
		for( int i=rl; i<ru; i++ ) {
			if( !a.isEmpty(i) )
				c.set(i, 0, builtin(a.values(i), a.pos(i), init, a.size(i), builtin));
			//correction (not sparse-safe)
			if( a.size(i) < n )
				c.set(i, 0, builtin.execute(c.get(i, 0), 0));
		}
	}
	
	/**
	 * COLMIN/COLMAX, opcode: uacmin/uacmax, sparse input.
	 * 
	 * @param a ?
	 * @param dc ?
	 * @param m ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uacmxx( SparseBlock a, DenseBlock dc, int m, int n, double init, Builtin builtin, int rl, int ru ) 
	{
		//init output (base for incremental agg)
		dc.set(init);
		
		//init count arrays (helper, see correction)
		//due to missing correction guaranteed to be single block
		double[] c = dc.valuesAt(0);
		int[] cnt = new int[ n ]; 

		//compute column aggregates min/max
		if( a.isContiguous() ) {
			int alen = (int) a.size(rl, ru);
			builtinAgg( a.values(rl), c, a.indexes(rl), a.pos(rl), alen, builtin );
			countAgg( a.values(rl), cnt, a.indexes(rl), a.pos(rl), alen );
		}
		else {
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) ) {
					int apos = a.pos(i);
					int alen = a.size(i);
					double[] avals = a.values(i);
					int[] aix = a.indexes(i);
					builtinAgg( avals, c, aix, apos, alen, builtin );
					countAgg( avals, cnt, aix, apos, alen );
				}
			}
		}
		
		//correction (not sparse-safe)
		//note: we need to determine if there are only nnz in a column
		// in order to know if for example a colMax of -8 is true or need
		// to be replaced with a 0 because there was a missing nonzero. 
		for( int i=0; i<n; i++ )
			if( cnt[i] < ru-rl ) //no dense column
				c[i] = builtin.execute(c[i], 0);
	}

	/**
	 * ROWINDEXMAX, opcode: uarimax, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uarimax( SparseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		if( n <= 0 )
			throw new DMLRuntimeException("rowIndexMax is undefined for ncol="+n);
		for( int i=rl; i<ru; i++ ) {
			if( !a.isEmpty(i) ) {
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				int maxindex = indexmax(a.values(i), apos, init, alen, builtin);
				double maxvalue = avals[apos+maxindex];
				c.set(i, 0, (double)aix[apos+maxindex] + 1);
				c.set(i, 1, maxvalue);
				//correction (not sparse-safe)
				if( alen < n && builtin.execute(0, maxvalue) == 1 ) {
					int ix = n-1; //find last 0 value
					for( int j=apos+alen-1; j>=apos; j--, ix-- )
						if( aix[j]!=ix )
							break;
					c.set(i, 0, ix + 1); //max index (last)
					c.set(i, 1, 0); //max value
				}
			}
			else { //if( arow==null )
				//correction (not sparse-safe)
				c.set(i, 0, n); //max index (last)
				c.set(i, 1, 0); //max value
			}
		}
	}
	
	/**
	 * ROWINDEXMIN, opcode: uarimin, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param init ?
	 * @param builtin ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uarimin( SparseBlock a, DenseBlock c, int n, double init, Builtin builtin, int rl, int ru ) {
		if( n <= 0 )
			throw new DMLRuntimeException("rowIndexMin is undefined for ncol="+n);
		for( int i=rl; i<ru; i++ ) {
			if( !a.isEmpty(i) ) {
				int apos = a.pos(i);
				int alen = a.size(i);
				int[] aix = a.indexes(i);
				double[] avals = a.values(i);
				int minindex = indexmin(avals, apos, init, alen, builtin);
				double minvalue = avals[apos+minindex];
				c.set(i, 0, (double)aix[apos+minindex] + 1);
				c.set(i, 1, minvalue); //min value among non-zeros
				//correction (not sparse-safe)
				if(alen < n && builtin.execute(0, minvalue) == 1) {
					int ix = n-1; //find last 0 value
					for( int j=alen-1; j>=0; j--, ix-- )
						if( aix[apos+j]!=ix )
							break;
					c.set(i, 0, ix + 1); //min index (last)
					c.set(i, 1, 0); //min value
				}
			}
			else { //if( arow==null )
				//correction (not sparse-safe)
				c.set(i, 0, n); //min index (last)
				c.set(i, 1, 0); //min value
			}
		}
	}
	
	/**
	 * MEAN, opcode: uamean, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kmean ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uamean( SparseBlock a, DenseBlock c, int n, KahanObject kbuff, Mean kmean, int rl, int ru )
	{
		int len = (ru-rl) * n;
		int count = 0;
		
		//correction remaining tuples (not sparse-safe)
		//note: before aggregate computation in order to
		//exploit 0 sum (noop) and better numerical stability
		count += (ru-rl)*n - a.size(rl, ru);
		
		//compute aggregate mean
		if( a.isContiguous() ) {
			int alen = (int) a.size(rl, ru);
			mean(a.values(rl), a.pos(rl), alen, count, kbuff, kmean);
			count += alen;
		}
		else {
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) ) {
					int alen = a.size(i);
					mean(a.values(i), a.pos(i), alen, count, kbuff, kmean);
					count += alen;
				}
			}
		}
		c.set(0, 0 , kbuff._sum);
		c.set(0, 1, len);
		c.set(0, 2, kbuff._correction);
	}

	/**
	 * ROWMEAN, opcode: uarmean, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kmean ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uarmean( SparseBlock a, DenseBlock c, int n, KahanObject kbuff, Mean kmean, int rl, int ru ) {
		for( int i=rl; i<ru; i++ ) {
			//correction remaining tuples (not sparse-safe)
			//note: before aggregate computation in order to
			//exploit 0 sum (noop) and better numerical stability
			int count = (a.isEmpty(i)) ? n : n-a.size(i);
			kbuff.set(0, 0); //reset buffer
			if( !a.isEmpty(i) )
				mean(a.values(i), a.pos(i), a.size(i), count, kbuff, kmean);
			c.set(i, 0, kbuff._sum);
			c.set(i, 1, n);
			c.set(i, 2, kbuff._correction);
		}
	}
	
	/**
	 * COLMEAN, opcode: uacmean, sparse input.
	 * 
	 * @param a ?
	 * @param c ?
	 * @param n ?
	 * @param kbuff ?
	 * @param kmean ?
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uacmean( SparseBlock a, DenseBlock c, int n, KahanObject kbuff, Mean kmean, int rl, int ru ) 
	{
		//correction remaining tuples (not sparse-safe)
		//note: before aggregate computation in order to
		//exploit 0 sum (noop) and better numerical stability
		c.set(1, 2, 0, n, ru-rl);
		double[] lc = c.values(1); //counts single row
		int cpos = c.pos(1);
		if( a.isContiguous() ) {
			countDisAgg( a.values(rl), lc, a.indexes(rl), a.pos(rl), cpos, (int)a.size(rl, ru) );
		}
		else {
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) )
					countDisAgg( a.values(i), lc, a.indexes(i), a.pos(i), cpos, a.size(i) );
			}
		}
		
		//compute column aggregate means
		if( a.isContiguous() ) {
			meanAgg( a.values(rl), c, a.indexes(rl), a.pos(rl), (int)a.size(rl, ru), n, kbuff, kmean );
		}
		else {
			for( int i=rl; i<ru; i++ ) {
				if( !a.isEmpty(i) )
					meanAgg( a.values(i), c, a.indexes(i), a.pos(i), a.size(i), n, kbuff, kmean );
			}
		}
	}

	/**
	 * VAR, opcode: uavar, sparse input.
	 *
	 * @param a Sparse array of values.
	 * @param c Output array to store variance, mean, count,
	 *          m2 correction factor, and mean correction factor.
	 * @param n Number of values per row.
	 * @param cbuff A CM_COV_Object to hold various intermediate
	 *              values for the variance calculation.
	 * @param cm A CM object of type Variance to perform the variance
	 *           calculation.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void s_uavar(SparseBlock a, DenseBlock c, int n, CmCovObject cbuff, CM cm, int rl, int ru) {
		// compute and store count of empty cells before aggregation
		int count = (ru-rl)*n - (int)a.size(rl, ru);
		cbuff.w = count;
		// calculate aggregated variance (only using non-empty cells)
		if( a.isContiguous() ) {
			var(a.values(rl), a.pos(rl), (int)a.size(rl, ru), cbuff, cm);
		}
		else {
			for (int i=rl; i<ru; i++) {
				if (!a.isEmpty(i))
					var(a.values(i), a.pos(i), a.size(i), cbuff, cm);
			}
		}
		// store results: { var | mean, count, m2 correction, mean correction }
		c.set(0, 0, cbuff.getRequiredResult(AggregateOperationTypes.VARIANCE));
		c.set(0, 1, cbuff.mean._sum);
		c.set(0, 2, cbuff.w);
		c.set(0, 3, cbuff.m2._correction);
		c.set(0, 4, cbuff.mean._correction);
	}

	/**
	 * ROWVAR, opcode: uarvar, sparse input.
	 *
	 * @param a Sparse array of values.
	 * @param c Output array to store variance, mean, count,
	 *          m2 correction factor, and mean correction factor.
	 * @param n Number of values per row.
	 * @param cbuff A CM_COV_Object to hold various intermediate
	 *              values for the variance calculation.
	 * @param cm A CM object of type Variance to perform the variance
	 *           calculation.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void s_uarvar(SparseBlock a, DenseBlock c, int n, CmCovObject cbuff, CM cm, int rl, int ru) {
		// calculate aggregated variance for each row
		for( int i=rl; i<ru; i++ ) {
			cbuff.reset(); // reset buffer for each row
			// compute and store count of empty cells in this row before aggregation
			int count = (a.isEmpty(i)) ? n : n-a.size(i);
			cbuff.w = count;
			if (!a.isEmpty(i))
				var(a.values(i), a.pos(i), a.size(i), cbuff, cm);
			// store results: { var | mean, count, m2 correction, mean correction }
			c.set(i, 0, cbuff.getRequiredResult(AggregateOperationTypes.VARIANCE));
			c.set(i, 1, cbuff.mean._sum);
			c.set(i, 2, cbuff.w);
			c.set(i, 3, cbuff.m2._correction);
			c.set(i, 4, cbuff.mean._correction);
		}
	}

	/**
	 * COLVAR, opcode: uacvar, sparse input.
	 *
	 * @param a Sparse array of values.
	 * @param c Output array to store variance, mean, count,
	 *          m2 correction factor, and mean correction factor.
	 * @param n Number of values per row.
	 * @param cbuff A CM_COV_Object to hold various intermediate
	 *              values for the variance calculation.
	 * @param cm A CM object of type Variance to perform the variance
	 *           calculation.
	 * @param rl Lower row limit.
	 * @param ru Upper row limit.
	 */
	private static void s_uacvar(SparseBlock a, DenseBlock c, int n, CmCovObject cbuff, CM cm, int rl, int ru) {
		// compute and store counts of empty cells per column before aggregation
		// note: column results are { var | mean, count, m2 correction, mean correction }
		// - first, store total possible column counts in 3rd row of output
		c.set(2, 3, 0, n, ru-rl); // counts stored in 3rd row
		// - then subtract one from the column count for each dense value in the column
		double[] lc = c.values(2);
		int cpos = c.pos(2);
		if( a.isContiguous() ) {
			countDisAgg(a.values(rl), lc, a.indexes(rl), a.pos(rl), cpos, (int)a.size(rl, ru)); 
		}
		else {
			for (int i=rl; i<ru; i++) {
				if (!a.isEmpty(i)) // counts stored in 3rd row
					countDisAgg(a.values(i), lc, a.indexes(i), a.pos(i), cpos, a.size(i)); 
			}
		}

		// calculate aggregated variance for each column
		if( a.isContiguous() ) {
			varAgg(a.values(rl), c, a.indexes(rl), a.pos(rl), (int)a.size(rl, ru), n, cbuff, cm);
		}
		else {
			for (int i=rl; i<ru; i++) {
				if (!a.isEmpty(i))
					varAgg(a.values(i), c, a.indexes(i), a.pos(i), a.size(i), n, cbuff, cm);
			}
		}
	}

	/**
	 * PROD, opcode: ua*, sparse input.
	 * 
	 * @param a Sparse block to process
	 * @param c Dense result block
	 * @param n Number of columns
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uam( SparseBlock a, DenseBlock c, int n, int rl, int ru ) {
		double ret = 1;
		for( int i=rl; i<ru; i++ ) {
			if( !a.isEmpty(i) ) {
				int alen = a.size(i);
				ret *= product(a.values(i), 0, alen);
				ret *= (alen<n) ? 0 : 1;
			}
			else
				ret *= (n==0) ? 1 : 0;
			
			//early abort (note: in case of NaNs this is an invalid optimization)
			if( !NAN_AWARENESS && ret==0 ) break;
		}
		c.set(0, 0, ret);
	}
	
	/**
	 * ROWPROD, opcode: uar*, sparse input.
	 * 
	 * @param a Sparse block to process
	 * @param c Dense result block
	 * @param n Number of columns
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uarm( SparseBlock a, DenseBlock c, int n, int rl, int ru ) {
		double[] lc = c.valuesAt(0);
		for( int i=rl; i<ru; i++ ) {
			if( !a.isEmpty(i) ) {
				final int alen = a.size(i);
				if(alen < n)
					lc[i] = 0;
				else{
					final int apos = a.pos(i);
					double tmp = product(a.values(i), apos, alen);
					lc[i] = tmp * ((alen<n) ? 0 : 1);
				}
			}
			else
				lc[i] = (n==0) ? 1 : 0;
		}
	}
	
	/**
	 * COLPROD, opcode: uac*, sparse input.
	 * 
	 * @param a Sparse block to process
	 * @param c Dense result block
	 * @param n Number of columns
	 * @param rl row lower index
	 * @param ru row upper index
	 */
	private static void s_uacm( SparseBlock a, DenseBlock c, int n, int rl, int ru ) {
		double[] lc = c.set(1).valuesAt(0);
		int[] cnt = new int[n];
		for(int i = rl; i < ru; i++) {
			if(a.isEmpty(i)) // if there is a empty row then stop processing since it is equivalient to multiplying with zero on all columns
				continue;
			countAgg(a.values(i), cnt, a.indexes(i), a.pos(i), a.size(i));
			LibMatrixMult.vectMultiplyWrite(lc, a.values(i), lc, a.indexes(i), 0, a.pos(i), 0, a.size(i));
		}
		for(int j = 0; j < n; j++)
			if(cnt[j] < ru - rl)
				lc[j] = 0;
	}
	
	
	////////////////////////////////////////////
	// performance-relevant utility functions //
	////////////////////////////////////////////
	
	private static void sum(double[] a, int ai, final int len, KahanObject kbuff, KahanFunction kplus) {
		for (int i=ai; i<ai+len; i++)
			kplus.execute2(kbuff, a[i]);
	}

	private static void sumAgg(double[] a, DenseBlock c, int ai, final int len, KahanObject kbuff, KahanFunction kplus) {
		//note: output might span multiple physical blocks
		double[] sum = c.values(0);
		double[] corr = c.values(1);
		int pos0 = c.pos(0), pos1 = c.pos(1);
		for (int i=0; i<len; i++) {
			kbuff._sum = sum[pos0+i];
			kbuff._correction = corr[pos1+i];
			kplus.execute2(kbuff, a[ai+i]);
			sum[pos0+i] = kbuff._sum;
			corr[pos1+i] = kbuff._correction;
		}
	}
	
	private static void sumAgg(double[] a, DenseBlock c, int[] aix, int ai, final int len, final int n, KahanObject kbuff, KahanFunction kplus) {
		//note: output might span multiple physical blocks
		double[] sum = c.values(0);
		double[] corr = c.values(1);
		int pos0 = c.pos(0), pos1 = c.pos(1);
		for (int i=ai; i<ai+len; i++) {
			int ix = aix[i];
			kbuff._sum = sum[pos0+ix];
			kbuff._correction = corr[pos1+ix];
			kplus.execute2(kbuff, a[i]);
			sum[pos0+ix] = kbuff._sum;
			corr[pos1+ix] = kbuff._correction;
		}
	}

	private static void sumAgg(double[] a, double[] c, int[] aix, int ai, final int len, final int n) {
		for(int i = ai; i < ai + len; i++)
			c[aix[i]] += a[i];
	}
	
	private static double product( double[] a, int ai, final int len ) {
		double val = 1;
		if( NAN_AWARENESS ) {
			//product without early abort
			//even if val is 0, it might turn into NaN.
			for( int i=0; i<len; i++, ai++ )
				val *= a[ ai ];	
		}
		else {
			//product with early abort (if 0)
			//note: this will not work with NaNs (invalid optimization)
			for( int i=0; i<len && val!=0; i++, ai++ )
				val *= a[ ai ];
		}
		return val;
	}

	private static void productAgg( double[] a, double[] c, int ai, int ci, final int len ) {
		//always w/ NAN_AWARENESS: product without early abort;
		//even if val is 0, it might turn into NaN.
		//(early abort would require column-flags and branches)
		for( int i=0; i<len; i++, ai++, ci++ )
			c[ ci ] *= a[ ai ];
	}

	private static void productAgg( double[] a, double[] c, int[] aix, int ai, int ci, final int len ) {
		//always w/ NAN_AWARENESS: product without early abort;
		//even if val is 0, it might turn into NaN.
		//(early abort would require column-flags and branches)
		for( int i=ai; i<ai+len; i++ )
			c[ ci + aix[i] ] *= a[ i ];
	}

	private static void mean( double[] a, int ai, final int len, int count, KahanObject kbuff, Mean mean ) {
		//delta: (newvalue-buffer._sum)/count
		for( int i=0; i<len; i++, ai++, count++ )
			mean.execute2(kbuff, a[ai], count+1);
	}

	private static void meanAgg( double[] a, DenseBlock c, int ai, final int len, KahanObject kbuff, Mean mean ) {
		//note: output might span multiple physical blocks
		double[] sum = c.values(0);
		double[] count = c.values(1);
		double[] corr = c.values(2);
		int pos0 = c.pos(0), pos1 = c.pos(1), pos2 = c.pos(2);
		for( int i=0; i<len; i++ ) {
			kbuff._sum = sum[pos0+i];
			double lcount = count[pos1+i] + 1;
			kbuff._correction = corr[pos2+i];
			mean.execute2(kbuff, a[ai+i], lcount);
			sum[pos0+i] = kbuff._sum;
			count[pos1+i] = lcount;
			corr[pos2+i] = kbuff._correction;
		}
	}

	private static void meanAgg( double[] a, DenseBlock c, int[] aix, int ai, final int len, final int n, KahanObject kbuff, Mean mean ) {
		//note: output might span multiple physical blocks
		double[] sum = c.values(0);
		double[] count = c.values(1);
		double[] corr = c.values(2);
		int pos0 = c.pos(0), pos1 = c.pos(1), pos2 = c.pos(2);
		for( int i=ai; i<ai+len; i++ ) {
			int ix = aix[i];
			kbuff._sum = sum[pos0+ix];
			double lcount = count[pos1+ix] + 1;
			kbuff._correction = corr[pos2+ix];
			mean.execute2(kbuff, a[ i ], lcount);
			sum[pos0+ix] = kbuff._sum;
			count[pos1+ix] = lcount;
			corr[pos2+ix] = kbuff._correction;
		}
	}

	private static void var(double[] a, int ai, final int len, CmCovObject cbuff, CM cm) {
		for(int i=0; i<len; i++, ai++)
			cbuff = (CmCovObject) cm.execute(cbuff, a[ai]);
	}

	private static void varAgg(double[] a, DenseBlock c, int ai, final int len, CmCovObject cbuff, CM cm) {
		//note: output might span multiple physical blocks
		double[] var = c.values(0);
		double[] mean = c.values(1);
		double[] count = c.values(2);
		double[] m2corr = c.values(3);
		double[] mcorr = c.values(4);
		int pos0 = c.pos(0), pos1 = c.pos(1),
		pos2 = c.pos(2), pos3 = c.pos(3), pos4 = c.pos(4);
		for (int i=0; i<len; i++) {
			// extract current values: { var | mean, count, m2 correction, mean correction }
			cbuff.w = count[pos2+i]; // count
			cbuff.m2._sum = var[pos0+i] * (cbuff.w - 1); // m2 = var * (n - 1)
			cbuff.mean._sum = mean[pos1+i]; // mean
			cbuff.m2._correction = m2corr[pos3+i];
			cbuff.mean._correction = mcorr[pos4+i];
			// calculate incremental aggregated variance
			cbuff = (CmCovObject) cm.execute(cbuff, a[ai+i]);
			// store updated values: { var | mean, count, m2 correction, mean correction }
			var[pos0+i] = cbuff.getRequiredResult(AggregateOperationTypes.VARIANCE);
			mean[pos1+i] = cbuff.mean._sum;
			count[pos2+i] = cbuff.w;
			m2corr[pos3+i] = cbuff.m2._correction;
			mcorr[pos4+i] = cbuff.mean._correction;
		}
	}

	private static void varAgg(double[] a, DenseBlock c, int[] aix, int ai, final int len, final int n, CmCovObject cbuff, CM cm)
	{
		//note: output might span multiple physical blocks
		double[] var = c.values(0);
		double[] mean = c.values(1);
		double[] count = c.values(2);
		double[] m2corr = c.values(3);
		double[] mcorr = c.values(4);
		int pos0 = c.pos(0), pos1 = c.pos(1),
		pos2 = c.pos(2), pos3 = c.pos(3), pos4 = c.pos(4);
		for (int i=ai; i<ai+len; i++) {
			// extract current values: { var | mean, count, m2 correction, mean correction }
			int ix = aix[i];
			cbuff.w = count[pos2+ix]; // count
			cbuff.m2._sum = var[pos0+ix] * (cbuff.w - 1); // m2 = var * (n - 1)
			cbuff.mean._sum = mean[pos1+ix]; // mean
			cbuff.m2._correction = m2corr[pos3+ix];
			cbuff.mean._correction = mcorr[pos4+ix];
			// calculate incremental aggregated variance
			cbuff = (CmCovObject) cm.execute(cbuff, a[i]);
			// store updated values: { var | mean, count, m2 correction, mean correction }
			var[pos0+ix] = cbuff.getRequiredResult(AggregateOperationTypes.VARIANCE);
			mean[pos1+ix] = cbuff.mean._sum;
			count[pos2+ix] = cbuff.w;
			m2corr[pos3+ix] = cbuff.m2._correction;
			mcorr[pos4+ix] = cbuff.mean._correction;
		}
	}
	
	private static double builtin( double[] a, int ai, final double init, final int len, Builtin aggop ) {
		double val = init;
		for( int i=0; i<len; i++, ai++ )
			val = aggop.execute( val, a[ ai ] );
		return val;
	}

	private static void builtinAgg( double[] a, double[] c, int ai, final int len, Builtin aggop ) {
		for( int i=0; i<len; i++ )
			c[ i ] = aggop.execute( c[ i ], a[ ai+i ] );
	}

	private static void builtinAgg( double[] a, double[] c, int[] aix, int ai, final int len, Builtin aggop ) {
		for( int i=ai; i<ai+len; i++ )
			c[ aix[i] ] = aggop.execute( c[ aix[i] ], a[ i ] );
	}

	private static int indexmax( double[] a, int ai, final double init, final int len, Builtin aggop ) {
		double maxval = init;
		int maxindex = -1;
		for( int i=ai; i<ai+len; i++ ) {
			maxindex = (a[i]>=maxval) ? i-ai : maxindex;
			maxval = (a[i]>=maxval) ? a[i] : maxval;
		}
		//note: robustness for all-NaN rows
		return Math.max(maxindex, 0);
	}

	private static int indexmin( double[] a, int ai, final double init, final int len, Builtin aggop ) {
		double minval = init;
		int minindex = -1;
		for( int i=ai; i<ai+len; i++ ) {
			minindex = (a[i]<=minval) ? i-ai : minindex;
			minval = (a[i]<=minval) ? a[i] : minval;
		}
		//note: robustness for all-NaN rows
		return Math.max(minindex, 0);
	}

	public static void countAgg( double[] a, int[] c, int[] aix, int ai, final int len ) {
		final int bn = len%8;
		//compute rest, not aligned to 8-block
		for( int i=ai; i<ai+bn; i++ )
			c[ aix[i] ]++;
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=ai+bn; i<ai+len; i+=8 ) {
			c[ aix[ i+0 ] ] ++;
			c[ aix[ i+1 ] ] ++;
			c[ aix[ i+2 ] ] ++;
			c[ aix[ i+3 ] ] ++;
			c[ aix[ i+4 ] ] ++;
			c[ aix[ i+5 ] ] ++;
			c[ aix[ i+6 ] ] ++;
			c[ aix[ i+7 ] ] ++;
		}
	}
	
	public static void countAgg( double[] a, int[] c, int ai, final int len ) {
		final int bn = len%8;
		//compute rest, not aligned to 8-block
		for( int i=0; i<bn; i++ )
			c[i] += a[ai+i]!=0 ? 1 : 0;
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=bn; i<len; i+=8 ) {
			c[i+0] += a[ai+i+0]!=0 ? 1 : 0;
			c[i+1] += a[ai+i+1]!=0 ? 1 : 0;
			c[i+2] += a[ai+i+2]!=0 ? 1 : 0;
			c[i+3] += a[ai+i+3]!=0 ? 1 : 0;
			c[i+4] += a[ai+i+4]!=0 ? 1 : 0;
			c[i+5] += a[ai+i+5]!=0 ? 1 : 0;
			c[i+6] += a[ai+i+6]!=0 ? 1 : 0;
			c[i+7] += a[ai+i+7]!=0 ? 1 : 0;
		}
	}
	
	private static void countDisAgg( double[] a, double[] c, int[] aix, int ai, final int ci, final int len ) {
		final int bn = len%8;
		//compute rest, not aligned to 8-block
		for( int i=ai; i<ai+bn; i++ )
			c[ ci+aix[i] ]--;
		//unrolled 8-block (for better instruction level parallelism)
		for( int i=ai+bn; i<ai+len; i+=8 ) {
			c[ ci+aix[ i+0 ] ] --;
			c[ ci+aix[ i+1 ] ] --;
			c[ ci+aix[ i+2 ] ] --;
			c[ ci+aix[ i+3 ] ] --;
			c[ ci+aix[ i+4 ] ] --;
			c[ ci+aix[ i+5 ] ] --;
			c[ ci+aix[ i+6 ] ] --;
			c[ ci+aix[ i+7 ] ] --;
		}
	}


	//////////////////////////////////////////////////////
	// Duplicated dense block related utility functions //
	/////////////////////////////////////////////////////


	private static void uakpDedup (DenseBlockFP64DEDUP a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru) {
		HashMap<double[], Integer> counts = new HashMap<>();
		if(a.getNrEmbsPerRow() != 1){
			//TODO: currently impossible case, since Dedup reshape is not supported yet, once it is, this method needs
			// to be implemented
			throw new NotImplementedException("Check TODO");
		}
		for(int i = rl; i < ru; i++) {
			double[] row = a.getDedupDirectly(i);
			Integer count = counts.getOrDefault(row, 0);
			count += 1;
			counts.put(row, count);
		}
		counts.forEach((row, count) -> {
			for(double r : row) {
				kplus.execute3(kbuff, r, count);
			}
		});
	}

	private static void uarkpDedup( DenseBlockFP64DEDUP a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		HashMap<double[], double[]> cache = new HashMap<>();
		if(a.getNrEmbsPerRow() != 1){
			//TODO: currently impossible case, since Dedup reshape is not supported yet, once it is, this method needs
			// to be implemented
			throw new NotImplementedException("Check TODO");
		}
		for(int i = rl; i < ru; i++) {
			double[] row = a.getDedupDirectly(i);
			double[] kbuff_array = cache.computeIfAbsent(row, lambda_row -> {
				kbuff.set(0, 0);
				sum(lambda_row, 0, n, kbuff, kplus);
				return new double[] {kbuff._sum, kbuff._correction};
			});
			cache.putIfAbsent(row, kbuff_array);
			c.set(i, 0, kbuff_array[0]);
			c.set(i, 1, kbuff_array[1]);
		}
	}

	private static void uackpDedup( DenseBlockFP64DEDUP a, DenseBlock c, int n, KahanObject kbuff, KahanPlus kplus, int rl, int ru ) {
		HashMap<double[], Integer> counts = new HashMap<>();
		if(a.getNrEmbsPerRow() != 1){
			//TODO: currently impossible case, since Dedup reshape is not supported yet, once it is, this method needs
			// to be implemented
			throw new NotImplementedException("Check TODO");
		}
		for(int i = rl; i < ru; i++) {
			double[] row = a.getDedupDirectly(i);
			Integer count = counts.getOrDefault(row, 0);
			count += 1;
			counts.put(row, count);
		}
		double[] sum = new double[n];
		double[] corr = new double[n];
		counts.forEach((row, count) -> {
			for(int i = 0; i < row.length; i++) {
				kbuff._sum = sum[i];
				kbuff._correction = corr[i];
				kplus.execute3(kbuff, row[i], count);
				sum[i] = kbuff._sum;
				corr[i] = kbuff._correction;
			}
		});
		double[] out_sum = c.values(0);
		double[] out_corr = c.values(1);
		int pos0 = c.pos(0), pos1 = c.pos(1);
		for(int i = 0; i < n; i++) {
			double tmp_sum = out_sum[pos0 + i] + sum[i];
			if(Math.abs(out_sum[pos0 + i]) > Math.abs(sum[i]))
				out_corr[pos1 + i] = ((out_sum[pos0 + i] - tmp_sum) + sum[i]) + out_corr[pos1 + i] + corr[i];
			else
				out_corr[pos1 + i] = ((sum[i] - tmp_sum) + out_sum[pos0 + i]) + out_corr[pos1 + i] + corr[i];
			out_sum[pos0 + i] = tmp_sum + out_corr[pos1 + i];
		}
	}


	public static MatrixBlock prepareAggregateUnaryOutput(MatrixBlock in, AggregateUnaryOperator op, MatrixValue result, int blen){
		CellIndex tempCellIndex = new CellIndex(-1,-1);
		final int rlen = in.getNumRows();
		final int clen = in.getNumColumns();
		op.indexFn.computeDimension(rlen, clen, tempCellIndex);
		if(op.aggOp.existsCorrection())
		{
			switch(op.aggOp.correction)
			{
				case LASTROW: 
					tempCellIndex.row++;
					break;
				case LASTCOLUMN: 
					tempCellIndex.column++;
					break;
				case LASTTWOROWS: 
					tempCellIndex.row+=2;
					break;
				case LASTTWOCOLUMNS: 
					tempCellIndex.column+=2;
					break;
				case LASTFOURROWS:
					tempCellIndex.row+=4;
					break;
				case LASTFOURCOLUMNS:
					tempCellIndex.column+=4;
					break;
				default:
					throw new DMLRuntimeException("unrecognized correctionLocation: "+op.aggOp.correction);
			}
		}
		
		//prepare result matrix block
		if(result==null)
			result=new MatrixBlock(tempCellIndex.row, tempCellIndex.column, false);
		else
			result.reset(tempCellIndex.row, tempCellIndex.column, false);
		return (MatrixBlock)result;
	}


	/////////////////////////////////////////////////////////
	// Task Implementations for Multi-Threaded Operations  //
	/////////////////////////////////////////////////////////
	
	private static abstract class AggTask implements Callable<Object> {}

	private static class RowAggTask extends AggTask 
	{
		private MatrixBlock _in  = null;
		private MatrixBlock _ret = null;
		private AggType _aggtype = null;
		private AggregateUnaryOperator _uaop = null;
		private int _rl = -1;
		private int _ru = -1;

		protected RowAggTask( MatrixBlock in, MatrixBlock ret, AggType aggtype, AggregateUnaryOperator uaop, int rl, int ru )
		{
			_in = in;
			_ret = ret;
			_aggtype = aggtype;
			_uaop = uaop;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() {
			if( !_in.sparse )
				aggregateUnaryMatrixDense(_in, _ret, _aggtype, _uaop.aggOp.increOp.fn, _uaop.indexFn, _rl, _ru);
			else
				aggregateUnaryMatrixSparse(_in, _ret, _aggtype, _uaop.aggOp.increOp.fn, _uaop.indexFn, _rl, _ru);
			return _ret.recomputeNonZeros(_rl, _ru-1);
		}
	}

	private static class PartialAggTask extends AggTask 
	{
		private MatrixBlock _in  = null;
		private MatrixBlock _ret = null;
		private AggType _aggtype = null;
		private AggregateUnaryOperator _uaop = null;
		private int _rl = -1;
		private int _ru = -1;

		protected PartialAggTask( MatrixBlock in, MatrixBlock ret, AggType aggtype, AggregateUnaryOperator uaop, int rl, int ru ) {
			_in = in;
			_ret = ret;
			_aggtype = aggtype;
			_uaop = uaop;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Object call() {
			//thead-local allocation for partial aggregation
			_ret = new MatrixBlock(_ret.rlen, _ret.clen, false);
			_ret.allocateDenseBlock();
			
			if( !_in.sparse )
				aggregateUnaryMatrixDense(_in, _ret, _aggtype, _uaop.aggOp.increOp.fn, _uaop.indexFn, _rl, _ru);
			else
				aggregateUnaryMatrixSparse(_in, _ret, _aggtype, _uaop.aggOp.increOp.fn, _uaop.indexFn, _rl, _ru);
			
			//recompute non-zeros of partial result
			_ret.recomputeNonZeros();
			
			return null;
		}
		
		public MatrixBlock getResult() {
			return _ret;
		}
	}

	private static class CumAggTask implements Callable<Long> 
	{
		private MatrixBlock _in  = null;
		private double[] _agg = null;
		private MatrixBlock _ret = null;
		private AggType _aggtype = null;
		private UnaryOperator _uop = null;
		private int _rl = -1;
		private int _ru = -1;

		protected CumAggTask( MatrixBlock in, double[] agg, MatrixBlock ret, AggType aggtype, UnaryOperator uop, int rl, int ru ) {
			_in = in;
			_agg = agg;
			_ret = ret;
			_aggtype = aggtype;
			_uop = uop;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() {
			//compute partial cumulative aggregate
			if( !_in.sparse )
				cumaggregateUnaryMatrixDense(_in, _ret, _aggtype, _uop.fn, _agg, _rl, _ru);
			else
				cumaggregateUnaryMatrixSparse(_in, _ret, _aggtype, _uop.fn, _agg, _rl, _ru);
			//recompute partial non-zeros (ru exlusive)
			return _ret.recomputeNonZeros(_rl, _ru-1, 0, _ret.getNumColumns()-1);
		}
	}

	private static class AggTernaryTask implements Callable<MatrixBlock>
	{
		private final MatrixBlock _in1;
		private final MatrixBlock _in2;
		private final MatrixBlock _in3;
		private MatrixBlock _ret = null;
		private final IndexFunction _ixFn;
		private final int _rl;
		private final int _ru;

		protected AggTernaryTask( MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, MatrixBlock ret, IndexFunction ixFn, int rl, int ru ) {
			_in1 = in1;
			_in2 = in2;
			_in3 = in3;
			_ret = ret;
			_ixFn = ixFn;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public MatrixBlock call() {
			//thead-local allocation for partial aggregation
			_ret = new MatrixBlock(_ret.rlen, _ret.clen, false);
			_ret.allocateDenseBlock();
			
			if( !_in1.sparse && !_in2.sparse && (_in3==null||!_in3.sparse) ) //DENSE
				aggregateTernaryDense(_in1, _in2, _in3, _ret, _ixFn, _rl, _ru);
			else //GENERAL CASE
				aggregateTernaryGeneric(_in1, _in2, _in3, _ret, _ixFn, _rl, _ru);
			
			//recompute non-zeros of partial result
			_ret.recomputeNonZeros();
			
			return _ret;
		}
	}
	
	private static class GrpAggTask extends AggTask 
	{
		private MatrixBlock _groups  = null;
		private MatrixBlock _target  = null;
		private MatrixBlock _weights  = null;
		private MatrixBlock _ret  = null;
		private int _numGroups = -1;
		private Operator _op = null;
		private int _cl = -1;
		private int _cu = -1;

		protected GrpAggTask( MatrixBlock groups, MatrixBlock target, MatrixBlock weights, MatrixBlock ret, int numGroups, Operator op, int cl, int cu ) {
			_groups = groups;
			_target = target;
			_weights = weights;
			_ret = ret;
			_numGroups = numGroups;
			_op = op;
			_cl = cl;
			_cu = cu;
		}
		
		@Override
		public Object call() {
			//CM operator for count, mean, variance
			//note: current support only for column vectors
			if( _op instanceof CMOperator ) {
				CMOperator cmOp = (CMOperator) _op;
				groupedAggregateCM(_groups, _target, _weights, _ret, _numGroups, cmOp, _cl, _cu);
			}
			//Aggregate operator for sum (via kahan sum)
			//note: support for row/column vectors and dense/sparse
			else if( _op instanceof AggregateOperator ) {
				AggregateOperator aggop = (AggregateOperator) _op;
				groupedAggregateKahanPlus(_groups, _target, _weights, _ret, _numGroups, aggop, _cl, _cu);
			}
			return null;
		}
	}
	
	private static class AggCmCovTask implements Callable<CmCovObject> {
		private final MatrixBlock _in1, _in2, _in3;
		private final ValueFunction _fn;
		private final int _rl, _ru;

		protected AggCmCovTask(MatrixBlock in1, MatrixBlock in2, MatrixBlock in3, ValueFunction fn, int rl, int ru) {
			_in1 = in1;
			_in2 = in2;
			_in3 = in3;
			_fn = fn;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public CmCovObject call() {
			//deep copy stateful CM function (has Kahan objects inside)
			//for correctness and to avoid cache thrashing among threads
			ValueFunction fn = (_fn instanceof CM) ? CM.getCMFnObject((CM)_fn) : _fn;
			//execute aggregate for row partition
			return aggregateCmCov(_in1, _in2, _in3, fn, _rl, _ru);
		}
	}
}

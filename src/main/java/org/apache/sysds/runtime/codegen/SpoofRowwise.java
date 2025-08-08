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

package org.apache.sysds.runtime.codegen;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.MemoryEstimates;


public abstract class SpoofRowwise extends SpoofOperator
{
	private static final long serialVersionUID = 6242910797139642998L;
	
	// Enum with explicit integer values
	// Thanks to https://codingexplained.com/coding/java/enum-to-integer-and-integer-to-enum
	// these values need to match with their native counterparts (spoof cuda ops)
	public enum RowType {
		NO_AGG(0),       //no aggregation
		NO_AGG_B1(1),    //no aggregation w/ matrix mult B1
		NO_AGG_CONST(2), //no aggregation w/ expansion/contraction
		FULL_AGG(3),     //full row/col aggregation
		ROW_AGG(4),      //row aggregation (e.g., rowSums() or X %*% v)
		COL_AGG (5),      //col aggregation (e.g., colSums() or t(y) %*% X)
		COL_AGG_T(6),    //transposed col aggregation (e.g., t(X) %*% y)
		COL_AGG_B1(7),   //col aggregation w/ matrix mult B1
		COL_AGG_B1_T(8), //transposed col aggregation w/ matrix mult B1
		COL_AGG_B1R(9),  //col aggregation w/ matrix mult B1 to row vector
		COL_AGG_CONST (10);//col aggregation w/ expansion/contraction
		
		private final int value;
		private final static HashMap<Integer, RowType> map = new HashMap<>();
		
		RowType(int value) {
			this.value = value;
		}
		
		static {
			for (RowType rowType : RowType.values()) {
				map.put(rowType.value, rowType);
			}
		}
		
		public static RowType valueOf(int rowType) {
			return map.get(rowType);
		}
		
		public int getValue() {
			return value;
		}
		
		public boolean isColumnAgg() {
			return this == COL_AGG || this == COL_AGG_T
				|| this == COL_AGG_B1 || this == COL_AGG_B1_T
				|| this == COL_AGG_B1R || this == COL_AGG_CONST;
		}
		public boolean isRowTypeB1() {
			return this == NO_AGG_B1 || this == COL_AGG_B1 
				|| this == COL_AGG_B1_T || this == COL_AGG_B1R;
		}
		public boolean isRowTypeB1ColumnAgg() {
			return (this == COL_AGG_B1) || (this == COL_AGG_B1_T);
		}
		public boolean isConstDim2(long dim2) {
			return (this == NO_AGG_CONST || this == COL_AGG_CONST)
				|| (dim2>=0 && isRowTypeB1());
		}
	}
	
	protected final RowType _type;
	protected final long _constDim2;
	protected final boolean _tB1;
	protected final int _reqVectMem;
	
	public SpoofRowwise(RowType type, long constDim2, boolean tB1, int reqVectMem) {
		_type = type;
		_constDim2 = constDim2;
		_tB1 = tB1;
		_reqVectMem = reqVectMem;
	}
	
	public RowType getRowType() {
		return _type;
	}
	
	public long getConstDim2() {
		return _constDim2;
	}
	
	public int getNumIntermediates() {
		return _reqVectMem;
	}
	
	public long getTmpMemoryReq(int k, long cols, long... cols2) {
		boolean hasMatrixSideInputs = IntStream.range(1, cols2.length)
			.mapToLong(i -> cols2[i]).anyMatch(n -> n > 1);
		long minCols = IntStream.range(1, cols2.length)
			.mapToLong(i -> cols2[i]).filter(c -> c > 1).min().orElse(1);
		long n = cols;
		long n2 = _type.isConstDim2(_constDim2) ? (int)_constDim2 : 
			_type.isRowTypeB1() || hasMatrixSideInputs ? minCols : -1;
		return (long)(k * _reqVectMem * ((n2>0 && n!=n2) ?
			(MemoryEstimates.doubleArrayCost(n) + MemoryEstimates.doubleArrayCost(n2)) :
			MemoryEstimates.doubleArrayCost(n)));
	}

	@Override
	public String getSpoofType() {
		return "RA" +  getClass().getName().split("\\.")[1];
	}
	
	@Override
	public SpoofCUDAOperator createCUDAInstrcution(Integer opID, SpoofCUDAOperator.PrecisionProxy ep) {
		return new SpoofCUDARowwise(_type, _constDim2, _tB1, _reqVectMem, opID, ep);
	}
	
	@Override
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, int k) {
		MatrixBlock out = ( k > 1 ) ?
			execute(inputs, scalarObjects, new MatrixBlock(1,1,false), k) :
			execute(inputs, scalarObjects, new MatrixBlock(1,1,false));
		return new DoubleObject(out.get(0, 0));
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out) {
		return execute(inputs, scalarObjects, out, true, false, 0);
	}
	
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, boolean allocTmp, boolean aggIncr, long rix) {
		//sanity check
		if( inputs==null || inputs.size() < 1 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//result allocation and preparations
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int n2 = _type.isConstDim2(_constDim2) ? (int)_constDim2 : 
			_type.isRowTypeB1() || hasMatrixSideInput(inputs) ?
			getMinColsMatrixSideInputs(inputs) : -1;
		if( !aggIncr || !out.isAllocated() )
			allocateOutputMatrix(m, n, n2, out);
		DenseBlock c = out.getDenseBlock();
		final boolean flipOut = _type.isRowTypeB1ColumnAgg()
			&& LibSpoofPrimitives.isFlipOuter(out.getNumRows(), out.getNumColumns());
		
		//input preparation
		SideInput[] b = prepInputMatrices(inputs, 1, inputs.size()-1, false, _tB1);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//setup thread-local memory if necessary
		if( allocTmp &&_reqVectMem > 0 )
			if(inputs.get(0).isInSparseFormat() && DMLScript.SPARSE_INTERMEDIATE) {
				LibSpoofPrimitives.setupSparseThreadLocalMemory(_reqVectMem, n/2, n2);
				LibSpoofPrimitives.setupThreadLocalMemory(_reqVectMem, n, n2);
			} else {
				LibSpoofPrimitives.setupThreadLocalMemory(_reqVectMem, n, n2);
			}
		
		//core sequential execute
		MatrixBlock a = inputs.get(0);
		if(a instanceof CompressedMatrixBlock)
			a = CompressedMatrixBlock.getUncompressed(a);

		if( !a.isInSparseFormat() )
			executeDense(a.getDenseBlock(), b, scalars, c, n, 0, m, rix);
		else
			executeSparse(a.getSparseBlock(), b, scalars, c, n, 0, m, rix);
		
		//post-processing
		if( allocTmp &&_reqVectMem > 0 )
			if(inputs.get(0).isInSparseFormat() && DMLScript.SPARSE_INTERMEDIATE) {
				LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
				LibSpoofPrimitives.cleanupThreadLocalMemory();
			} else {
				LibSpoofPrimitives.cleanupThreadLocalMemory();
			}
		if( flipOut ) {
			fixTransposeDimensions(out);
			out = LibMatrixReorg.transpose(out, new MatrixBlock(
				out.getNumColumns(), out.getNumRows(), false));
		}
		if( !aggIncr ) {
			out.recomputeNonZeros();
			out.examSparsity();
		}
		return out;
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k)
	{
		//redirect to serial execution
		if( k <= 1 || (_type.isColumnAgg() && !LibMatrixMult.satisfiesMultiThreadingConstraints(inputs.get(0), k))
			|| getTotalInputSize(inputs) < PAR_NUMCELL_THRESHOLD ) {
			return execute(inputs, scalarObjects, out);
		}
		
		//sanity check
		if( inputs==null || inputs.size() < 1 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//result allocation and preparations
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int n2 = _type.isConstDim2(_constDim2) ? (int)_constDim2 : 
			_type.isRowTypeB1() || hasMatrixSideInput(inputs) ?
			getMinColsMatrixSideInputs(inputs) : -1;
		allocateOutputMatrix(m, n, n2, out);
		final boolean flipOut = _type.isRowTypeB1ColumnAgg()
			&& LibSpoofPrimitives.isFlipOuter(out.getNumRows(), out.getNumColumns());
		
		//input preparation
		MatrixBlock a = inputs.get(0);
		SideInput[] b = prepInputMatrices(inputs, 1, inputs.size()-1, false, _tB1);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core parallel execute
		ExecutorService pool = CommonThreadPool.get(k);
		
		try {
			ArrayList<Integer> blklens = UtilFunctions
				.getBalancedBlockSizesDefault(m, k, (long)m*n<16*PAR_NUMCELL_THRESHOLD);
			if( _type.isColumnAgg() || _type == RowType.FULL_AGG ) {
				//execute tasks
				ArrayList<ParColAggTask> tasks = new ArrayList<>();
				int outLen = out.getNumRows() * out.getNumColumns();
				for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
					tasks.add(new ParColAggTask(a, b, scalars, n, n2, outLen, lb, lb+blklens.get(i)));
				List<Future<DenseBlock>> taskret = pool.invokeAll(tasks);
				//aggregate partial results
				int len = _type.isColumnAgg() ? out.getNumRows()*out.getNumColumns() : 1;
				for( Future<DenseBlock> task : taskret )
					LibMatrixMult.vectAdd(task.get().valuesAt(0), out.getDenseBlockValues(), 0, 0, len);
				out.recomputeNonZeros();
			}
			else {
				//execute tasks
				ArrayList<ParExecTask> tasks = new ArrayList<>();
				for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
					tasks.add(new ParExecTask(a, b, out, scalars, n, n2, lb, lb+blklens.get(i)));
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				//aggregate nnz, no need to aggregate results
				long nnz = 0;
				for( Future<Long> task : taskret )
					nnz += task.get();
				out.setNonZeros(nnz);
			}
			
			if( flipOut ) {
				fixTransposeDimensions(out);
				out = LibMatrixReorg.transpose(out, new MatrixBlock(
					out.getNumColumns(), out.getNumRows(), false));
			}
			out.examSparsity();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		finally{
			pool.shutdown();
		}
		
		return out;
	}
	
	public static boolean hasMatrixSideInput(ArrayList<MatrixBlock> inputs) {
		return IntStream.range(1, inputs.size())
			.mapToObj(i -> inputs.get(i))
			.anyMatch(in -> in.getNumColumns()>1);
	}
	
	protected static int getMinColsMatrixSideInputs(ArrayList<MatrixBlock> inputs) {
		//For B1 types, get the output number of columns as the minimum
		//number of columns of side input matrices other than vectors.
		return IntStream.range(1, inputs.size())
			.map(i -> inputs.get(i).getNumColumns())
			.filter(ncol -> ncol > 1).min().orElse(1);
	}
	
	public static boolean hasMatrixObjectSideInput(ArrayList<MatrixObject> inputs) {
		return IntStream.range(1, inputs.size())
			.mapToObj(i -> inputs.get(i))
			.anyMatch(in -> in.getNumColumns()>1);
	}
	
	protected static int getMinColsMatrixObjectSideInputs(ArrayList<MatrixObject> inputs) {
		//For B1 types, get the output number of columns as the minimum
		//number of columns of side input matrices other than vectors.
		return IntStream.range(1, inputs.size())
			.map(i -> (int) inputs.get(i).getNumColumns())
			.filter(ncol -> ncol > 1).min().orElse(1);
	}
	
	protected class OutputDimensions {
		public final int rows;
		public final int cols;
		OutputDimensions(int m, int n, int n2) {
			switch(_type) {
				case NO_AGG:  		rows = m;	cols = n; break;
				case NO_AGG_B1: 	rows = m;	cols = n2; break;
				case NO_AGG_CONST: 	rows = m;	cols = (int) SpoofRowwise.this._constDim2; break;
				case FULL_AGG: 		rows = 1;	cols = 1; break;
				case ROW_AGG: 		rows = m;	cols = 1; break;
				case COL_AGG: 		rows = 1;	cols = n; break;
				case COL_AGG_T: 	rows = n;	cols = 1; break;
				case COL_AGG_B1: 	rows = n2;	cols = n; break;
				case COL_AGG_B1_T: 	rows = n;	cols = n2; break;
				case COL_AGG_B1R: 	rows = 1;	cols = n2; break;
				case COL_AGG_CONST: rows = 1;	cols = (int) SpoofRowwise.this._constDim2; break;
				default: 			rows = 0;	cols = 0;
			}
		}
	}
	
	
	private void allocateOutputMatrix(int m, int n, int n2, MatrixBlock out) {
		OutputDimensions dims = new OutputDimensions(m, n, n2);
		out.reset(dims.rows, dims.cols, false);
		out.allocateDenseBlock();
	}
	
	private static void fixTransposeDimensions(MatrixBlock out) {
		int rlen = out.getNumRows();
		out.setNumRows(out.getNumColumns());
		out.setNumColumns(rlen);
		out.setNonZeros(out.getNumRows()*out.getNumColumns());
	}
	
	private void executeDense(DenseBlock a, SideInput[] b, double[] scalars, DenseBlock c, int n, int rl, int ru, long rix) {
		//forward empty block to sparse
		if( a == null ) {
			executeSparse(null, b, scalars, c, n, rl, ru, rix);
			return;
		}
		
		SideInput[] lb = createSparseSideInputs(b, true);
		for( int i=rl; i<ru; i++ ) {
			genexec(a.values(i), a.pos(i), lb, scalars,
				c.values(i), c.pos(i), n, rix+i, i );
		}
	}
	
	private void executeSparse(SparseBlock a, SideInput[] b, double[] scalars, DenseBlock c, int n, int rl, int ru, long rix) {
		SideInput[] lb = createSparseSideInputs(b, true);
		SparseRow empty = new SparseRowVector(1);
		for( int i=rl; i<ru; i++ ) {
			if( a!=null && !a.isEmpty(i) ) {
				//call generated method
				genexec(a.values(i), a.indexes(i), a.pos(i), lb, scalars,
					c.values(i), c.pos(i), a.size(i), n, rix+i, i);
			}
			else
				genexec(empty.values(), empty.indexes(), 0, lb, scalars,
					c.values(i), c.pos(i), 0, n, rix+i, i);
		}
	}
	
	//methods to be implemented by generated operators of type SpoofRowAggrgate 
	
	//local execution where grix==rix
	protected final void genexec(double[] a, int ai, 
		SideInput[] b, double[] scalars, double[] c, int ci, int len, int rix) {
		genexec(a, ai, b, scalars, c, ci, len, rix, rix);
	}
	
	protected final void genexec(double[] avals, int[] aix, int ai, 
		SideInput[] b, double[] scalars, double[] c, int ci, int alen, int n, int rix) {
		genexec(avals, aix, ai, b, scalars, c, ci, alen, n, rix, rix);
	}
	
	//distributed execution with additional global row index
	protected abstract void genexec(double[] a, int ai, 
		SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix);
	
	protected abstract void genexec(double[] avals, int[] aix, int ai, 
		SideInput[] b, double[] scalars, double[] c, int ci, int alen, int n, long grix, int rix);

	
	/**
	 * Task for multi-threaded column aggregation operations.
	 */
	private class ParColAggTask implements Callable<DenseBlock> 
	{
		private final MatrixBlock _a;
		private final SideInput[] _b;
		private final double[] _scalars;
		private final int _clen, _clen2, _outLen;
		private final int _rl, _ru;

		protected ParColAggTask( MatrixBlock a, SideInput[] b, double[] scalars, int clen, int clen2, int outLen, int rl, int ru ) {
			_a = a;
			_b = b;
			_scalars = scalars;
			_clen = clen;
			_clen2 = clen2;
			_outLen = outLen;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public DenseBlock call() {
			
			//allocate vector intermediates and partial output
			if( _reqVectMem > 0 )
				LibSpoofPrimitives.setupThreadLocalMemory(_reqVectMem, _clen, _clen2);
			DenseBlock c = DenseBlockFactory.createDenseBlock(1, _outLen);
			
			if( !_a.isInSparseFormat() )
				executeDense(_a.getDenseBlock(), _b, _scalars, c, _clen, _rl, _ru, 0);
			else
				executeSparse(_a.getSparseBlock(), _b, _scalars, c, _clen, _rl, _ru, 0);
			
			if( _reqVectMem > 0 )
				LibSpoofPrimitives.cleanupThreadLocalMemory();
			return c;
		}
	}
	
	/**
	 * Task for multi-threaded execution with no or row aggregation.
	 */
	private class ParExecTask implements Callable<Long> 
	{
		private final MatrixBlock _a;
		private final SideInput[] _b;
		private final MatrixBlock _c;
		private final double[] _scalars;
		private final int _clen;
		private final int _clen2;
		private final int _rl;
		private final int _ru;

		protected ParExecTask( MatrixBlock a, SideInput[] b, MatrixBlock c, double[] scalars, int clen, int clen2, int rl, int ru ) {
			_a = a;
			_b = b;
			_c = c;
			_scalars = scalars;
			_clen = clen;
			_clen2 = clen2;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public Long call() {
			//allocate vector intermediates
			if( _reqVectMem > 0 )
				LibSpoofPrimitives.setupThreadLocalMemory(_reqVectMem, _clen, _clen2);
			
			if( !_a.isInSparseFormat() )
				executeDense(_a.getDenseBlock(), _b, _scalars, _c.getDenseBlock(), _clen, _rl, _ru, 0);
			else
				executeSparse(_a.getSparseBlock(), _b, _scalars, _c.getDenseBlock(), _clen, _rl, _ru, 0);
			
			if( _reqVectMem > 0 )
				LibSpoofPrimitives.cleanupThreadLocalMemory();
			
			//maintain nnz for row partition
			return _c.recomputeNonZeros(_rl, _ru-1, 0, _c.getNumColumns()-1);
		}
	}
}

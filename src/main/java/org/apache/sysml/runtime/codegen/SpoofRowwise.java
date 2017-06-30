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

package org.apache.sysml.runtime.codegen;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.instructions.cp.DoubleObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.LibMatrixMult;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.data.SparseRow;
import org.apache.sysml.runtime.matrix.data.SparseRowVector;
import org.apache.sysml.runtime.util.UtilFunctions;


public abstract class SpoofRowwise extends SpoofOperator
{
	private static final long serialVersionUID = 6242910797139642998L;
	private static final long PAR_NUMCELL_THRESHOLD = 1024*1024;   //Min 1M elements
	
	public enum RowType {
		NO_AGG,    //no aggregation
		NO_AGG_B1, //no aggregation w/ matrix mult B1
		FULL_AGG,  //full row/col aggregation
		ROW_AGG,   //row aggregation (e.g., rowSums() or X %*% v)
		COL_AGG,   //col aggregation (e.g., colSums() or t(y) %*% X)
		COL_AGG_T, //transposed col aggregation (e.g., t(X) %*% y)
		COL_AGG_B1,   //col aggregation w/ matrix mult B1
		COL_AGG_B1_T; //transposed col aggregation w/ matrix mult B1
		
		public boolean isColumnAgg() {
			return (this == COL_AGG || this == COL_AGG_T)
				|| (this == COL_AGG_B1) || (this == COL_AGG_B1_T);
		}
		public boolean isRowTypeB1() {
			return (this == NO_AGG_B1) || (this == COL_AGG_B1) || (this == COL_AGG_B1_T);
		} 
	}
	
	protected final RowType _type;
	protected final boolean _cbind0;
	protected final boolean _tB1;
	protected final int _reqVectMem;
	
	public SpoofRowwise(RowType type, boolean cbind0, boolean tB1, int reqVectMem) {
		_type = type;
		_cbind0 = cbind0;
		_tB1 = tB1;
		_reqVectMem = reqVectMem;
	}
	
	public RowType getRowType() {
		return _type;
	}
	
	public boolean isCBind0() {
		return _cbind0;
	}
	
	public int getNumIntermediates() {
		return _reqVectMem;
	}

	@Override
	public String getSpoofType() {
		return "RA" +  getClass().getName().split("\\.")[1];
	}
	
	@Override
	public ScalarObject execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, int k) 
		throws DMLRuntimeException 
	{
		MatrixBlock out = new MatrixBlock(1, 1, false);
		if( k > 1 )
			execute(inputs, scalarObjects, out, k);
		else
			execute(inputs, scalarObjects, out);
		return new DoubleObject(out.quickGetValue(0, 0));
	}
	
	@Override
	public void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out)	
		throws DMLRuntimeException 
	{
		execute(inputs, scalarObjects, out, true, false);
	}
	
	public void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, boolean allocTmp, boolean aggIncr) 
		throws DMLRuntimeException	
	{
		//sanity check
		if( inputs==null || inputs.size() < 1 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//result allocation and preparations
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int n2 = _type.isRowTypeB1() ? inputs.get(1).getNumColumns() : -1;
		if( !aggIncr || !out.isAllocated() )
			allocateOutputMatrix(m, n, n2, out);
		double[] c = out.getDenseBlock();
		
		//input preparation
		SideInput[] b = prepInputMatrices(inputs, 1, inputs.size()-1, true, _tB1);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//setup thread-local memory if necessary
		if( allocTmp )
			LibSpoofPrimitives.setupThreadLocalMemory(_reqVectMem, n, n2);
		
		//core sequential execute
		MatrixBlock a = inputs.get(0);
		if( a instanceof CompressedMatrixBlock )
			executeCompressed((CompressedMatrixBlock)a, b, scalars, c, n, 0, m);
		else if( !a.isInSparseFormat() )
			executeDense(a.getDenseBlock(), b, scalars, c, n, 0, m);
		else
			executeSparse(a.getSparseBlock(), b, scalars, c, n, 0, m);
		
		//post-processing
		if( allocTmp )
			LibSpoofPrimitives.cleanupThreadLocalMemory();
		out.recomputeNonZeros();
		out.examSparsity();
	}
	
	@Override
	public void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k)	
		throws DMLRuntimeException
	{
		//redirect to serial execution
		if( k <= 1 || (long)inputs.get(0).getNumRows()*inputs.get(0).getNumColumns()<PAR_NUMCELL_THRESHOLD ) {
			execute(inputs, scalarObjects, out);
			return;
		}
		
		//sanity check
		if( inputs==null || inputs.size() < 1 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//result allocation and preparations
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		final int n2 = _type.isRowTypeB1() ? inputs.get(1).getNumColumns() : -1;
		allocateOutputMatrix(m, n, n2, out);
		
		//input preparation
		SideInput[] b = prepInputMatrices(inputs, 1, inputs.size()-1, true, _tB1);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core parallel execute
		ExecutorService pool = Executors.newFixedThreadPool( k );
		int nk = UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
		int blklen = (int)(Math.ceil((double)m/nk));
		try
		{
			if( _type.isColumnAgg() || _type == RowType.FULL_AGG ) {
				//execute tasks
				ArrayList<ParColAggTask> tasks = new ArrayList<ParColAggTask>();
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParColAggTask(inputs.get(0), b, scalars, n, n2, i*blklen, Math.min((i+1)*blklen, m)));
				List<Future<double[]>> taskret = pool.invokeAll(tasks);	
				//aggregate partial results
				int len = _type.isColumnAgg() ? out.getNumRows()*out.getNumColumns() : 1;
				for( Future<double[]> task : taskret )
					LibMatrixMult.vectAdd(task.get(), out.getDenseBlock(), 0, 0, len);
				out.recomputeNonZeros();
			}
			else {
				//execute tasks
				ArrayList<ParExecTask> tasks = new ArrayList<ParExecTask>();
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParExecTask(inputs.get(0), b, out, scalars, n, n2, i*blklen, Math.min((i+1)*blklen, m)));
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				//aggregate nnz, no need to aggregate results
				long nnz = 0;
				for( Future<Long> task : taskret )
					nnz += task.get();
				out.setNonZeros(nnz);
			}
			
			pool.shutdown();
			out.examSparsity();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
	
	private void allocateOutputMatrix(int m, int n, int n2, MatrixBlock out) {
		switch( _type ) {
			case NO_AGG:       out.reset(m, n, false); break;
			case NO_AGG_B1:    out.reset(m, n2, false); break;
			case FULL_AGG:     out.reset(1, 1, false); break;
			case ROW_AGG:      out.reset(m, 1+(_cbind0?1:0), false); break;
			case COL_AGG:      out.reset(1, n, false); break;
			case COL_AGG_T:    out.reset(n, 1, false); break;
			case COL_AGG_B1:   out.reset(n2, n, false); break;
			case COL_AGG_B1_T: out.reset(n, n2, false); break;
			
		}
		out.allocateDenseBlock();
	}
	
	private void executeDense(double[] a, SideInput[] b, double[] scalars, double[] c, int n, int rl, int ru) 
	{
		if( a == null )
			return;
		
		for( int i=rl, aix=rl*n; i<ru; i++, aix+=n ) {
			//call generated method
			genexec( a, aix, b, scalars, c, n, i );
		}
	}
	
	private void executeSparse(SparseBlock sblock, SideInput[] b, double[] scalars, double[] c, int n, int rl, int ru) 
	{
		SparseRow empty = new SparseRowVector(1);
		for( int i=rl; i<ru; i++ ) {
			if( sblock!=null && !sblock.isEmpty(i) ) {
				double[] avals = sblock.values(i);
				int[] aix = sblock.indexes(i);
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				
				//call generated method
				genexec(avals, aix, apos, b, scalars, c, alen, n, i);
			}
			else
				genexec(empty.values(), 
					empty.indexes(), 0, b, scalars, c, 0, n, i);	
		}
	}
	
	private void executeCompressed(CompressedMatrixBlock a, SideInput[] b, double[] scalars, double[] c, int n, int rl, int ru) 
	{
		if( a.isEmptyBlock(false) )
			return;
		
		if( !a.isInSparseFormat() ) { //DENSE
			Iterator<double[]> iter = a.getDenseRowIterator(rl, ru);
			for( int i=rl; iter.hasNext(); i++ ) {
				genexec(iter.next(), 0, b, scalars, c, n, i);
			}
		}
		else { //SPARSE
			Iterator<SparseRow> iter = a.getSparseRowIterator(rl, ru);
			SparseRow empty = new SparseRowVector(1);
			for( int i=rl; iter.hasNext(); i++ ) {
				SparseRow row = iter.next();
				if( !row.isEmpty() )
					genexec(row.values(), 
						row.indexes(), 0, b, scalars, c, row.size(), n, i);
				else
					genexec(empty.values(), 
						empty.indexes(), 0, b, scalars, c, 0, n, i);
			}
		}
	}
	
	//methods to be implemented by generated operators of type SpoofRowAggrgate 
	
	protected abstract void genexec(double[] a, int ai, 
		SideInput[] b, double[] scalars, double[] c, int len, int rowIndex);
	
	protected abstract void genexec(double[] avals, int[] aix, int ai, 
		SideInput[] b, double[] scalars, double[] c, int alen, int n, int rowIndex);

	
	/**
	 * Task for multi-threaded column aggregation operations.
	 */
	private class ParColAggTask implements Callable<double[]> 
	{
		private final MatrixBlock _a;
		private final SideInput[] _b;
		private final double[] _scalars;
		private final int _clen;
		private final int _clen2;
		private final int _rl;
		private final int _ru;

		protected ParColAggTask( MatrixBlock a, SideInput[] b, double[] scalars, int clen, int clen2, int rl, int ru ) {
			_a = a;
			_b = b;
			_scalars = scalars;
			_clen = clen;
			_clen2 = clen2;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public double[] call() throws DMLRuntimeException {
			
			//allocate vector intermediates and partial output
			LibSpoofPrimitives.setupThreadLocalMemory(_reqVectMem, _clen, _clen2);
			double[] c = new double[(_clen2>0)?_clen*_clen2 : _clen];
			
			if( _a instanceof CompressedMatrixBlock )
				executeCompressed((CompressedMatrixBlock)_a, _b, _scalars, c, _clen, _rl, _ru);
			else if( !_a.isInSparseFormat() )
				executeDense(_a.getDenseBlock(), _b, _scalars, c, _clen, _rl, _ru);
			else
				executeSparse(_a.getSparseBlock(), _b, _scalars, c, _clen, _rl, _ru);
			
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
		public Long call() throws DMLRuntimeException {
			//allocate vector intermediates
			LibSpoofPrimitives.setupThreadLocalMemory(_reqVectMem, _clen, _clen2);
			
			if( _a instanceof CompressedMatrixBlock )
				executeCompressed((CompressedMatrixBlock)_a, _b, _scalars, _c.getDenseBlock(), _clen, _rl, _ru);
			else if( !_a.isInSparseFormat() )
				executeDense(_a.getDenseBlock(), _b, _scalars, _c.getDenseBlock(), _clen, _rl, _ru);
			else
				executeSparse(_a.getSparseBlock(), _b, _scalars, _c.getDenseBlock(), _clen, _rl, _ru);
			LibSpoofPrimitives.cleanupThreadLocalMemory();
			
			//maintain nnz for row partition
			return _c.recomputeNonZeros(_rl, _ru-1, 0, _c.getNumColumns()-1);
		}
	}
}

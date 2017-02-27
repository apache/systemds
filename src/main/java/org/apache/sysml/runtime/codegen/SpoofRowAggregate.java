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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.LibMatrixMult;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.util.UtilFunctions;


public abstract class SpoofRowAggregate extends SpoofOperator
{
	private static final long serialVersionUID = 6242910797139642998L;
	private static final long PAR_NUMCELL_THRESHOLD = 1024*1024;   //Min 1M elements
	
	protected boolean _colVector = false;
	
	public SpoofRowAggregate() {

	}

	@Override
	public void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out)	
		throws DMLRuntimeException
	{
		//sanity check
		if( inputs==null || inputs.size() < 1 || out==null )
			throw new RuntimeException("Invalid input arguments.");
		
		//result allocation and preparations
		out.reset(_colVector ? inputs.get(0).getNumColumns() : 1, 
			_colVector ? 1 : inputs.get(0).getNumColumns(), false);
		out.allocateDenseBlock();
		double[] c = out.getDenseBlock();
		
		//input preparation
		double[][] b = prepInputMatrices(inputs);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core sequential execute
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();		
		if( !inputs.get(0).isInSparseFormat() )
			executeDense(inputs.get(0).getDenseBlock(), b, scalars, c, n, 0, m);
		else
			executeSparse(inputs.get(0).getSparseBlock(), b, scalars, c, n, 0, m);
	
		//post-processing
		out.recomputeNonZeros();	
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
		out.reset(_colVector ? inputs.get(0).getNumColumns() : 1, 
			_colVector ? 1 : inputs.get(0).getNumColumns(), false);
		out.allocateDenseBlock();
		
		//input preparation
		double[][] b = prepInputMatrices(inputs);
		double[] scalars = prepInputScalars(scalarObjects);
		
		//core parallel execute
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();		
		try {
			ExecutorService pool = Executors.newFixedThreadPool( k );
			ArrayList<ParExecTask> tasks = new ArrayList<ParExecTask>();
			int nk = UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
			int blklen = (int)(Math.ceil((double)m/nk));
			for( int i=0; i<nk & i*blklen<m; i++ )
				tasks.add(new ParExecTask(inputs.get(0), b, scalars, n, i*blklen, Math.min((i+1)*blklen, m)));
			//execute tasks
			List<Future<double[]>> taskret = pool.invokeAll(tasks);	
			pool.shutdown();
			//aggregate partial results
			for( Future<double[]> task : taskret )
				LibMatrixMult.vectAdd(task.get(), out.getDenseBlock(), 0, 0, n);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//post-processing
		out.recomputeNonZeros();	
	}
	
	private void executeDense(double[] a, double[][] b, double[] scalars, double[] c, int n, int rl, int ru) 
	{
		for( int i=rl, aix=rl*n; i<ru; i++, aix+=n ) {
			//call generated method
			genexecRowDense( a, aix, b, scalars, c, n, i );
		}
	}
	
	private void executeSparse(SparseBlock sblock, double[][] b, double[] scalars, double[] c, int n, int rl, int ru) 
	{
		for( int i=rl; i<ru; i++ ) {
			if( !sblock.isEmpty(i) ) {
				double[] avals = sblock.values(i);
				int[] aix = sblock.indexes(i);
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				
				//call generated method
				genexecRowSparse(avals, aix, apos, b, scalars, c, alen, i);
			}
		}
	}
	
	//methods to be implemented by generated operators of type SpoofRowAggrgate 
	
	protected abstract void genexecRowDense( double[] a, int ai, double[][] b, double[] scalars, double[] c, int len, int rowIndex );
	
	protected abstract void genexecRowSparse( double[] avals, int[] aix, int ai, double[][] b, double[] scalars, double[] c, int len, int rowIndex );

	
	/**
	 * Task for multi-threaded operations.
	 */
	private class ParExecTask implements Callable<double[]> 
	{
		private final MatrixBlock _a;
		private final double[][] _b;
		private final double[] _scalars;
		private final int _clen;
		private final int _rl;
		private final int _ru;

		protected ParExecTask( MatrixBlock a, double[][] b, double[] scalars, int clen, int rl, int ru ) {
			_a = a;
			_b = b;
			_scalars = scalars;
			_clen = clen;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public double[] call() throws DMLRuntimeException {
			double[] c = new double[_clen];
			if( !_a.isInSparseFormat() )
				executeDense(_a.getDenseBlock(), _b, _scalars, c, _clen, _rl, _ru);
			else
				executeSparse(_a.getSparseBlock(), _b, _scalars, c, _clen, _rl, _ru);
				
			return c;
		}
	}
}

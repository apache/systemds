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

import java.io.Serializable;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.SpoofCellwise.AggOp;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class SpoofMultiAggregate extends SpoofOperator implements Serializable
{
	private static final long serialVersionUID = -6164871955591089349L;
	
	private final AggOp[] _aggOps;
	private final boolean _sparseSafe;
	
	public SpoofMultiAggregate(boolean sparseSafe, AggOp... aggOps) {
		_sparseSafe = sparseSafe;
		_aggOps = aggOps;
	}
	
	public AggOp[] getAggOps() {
		return _aggOps;
	}
	
	public boolean isSparseSafe() {
		return _sparseSafe;
	}
	
	@Override
	public String getSpoofType() {
		return "MA" +  getClass().getName().split("\\.")[1];
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out) {
		return execute(inputs, scalarObjects, out, 1, 0);
	}
	
	@Override
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k) {
		return execute(inputs, scalarObjects, out, k, 0);
	}
	
	public MatrixBlock execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k, long rix) {
		//sanity check
		if( inputs==null || inputs.size() < 1  )
			throw new RuntimeException("Invalid input arguments.");
		
		long inputSize = isSparseSafe() ?
			getTotalInputNnz(inputs) : getTotalInputSize(inputs);
		if( inputSize < PAR_NUMCELL_THRESHOLD ) {
			k = 1; //serial execution
		}
	
		//result allocation and preparations
		out.reset(1, _aggOps.length, false);
		out.allocateDenseBlock();
		double[] c = out.getDenseBlockValues(); //1x<num_agg>
		setInitialOutputValues(c);
		
		//input preparation
		SideInput[] b = prepInputMatrices(inputs);
		double[] scalars = prepInputScalars(scalarObjects);
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		boolean sparseSafe = isSparseSafe();
		
		if( k <= 1 ) //SINGLE-THREADED
		{
			if( !inputs.get(0).isInSparseFormat() )
				executeDense(inputs.get(0).getDenseBlock(), b, scalars, c, m, n, sparseSafe, 0, m, rix);
			else
				executeSparse(inputs.get(0).getSparseBlock(), b, scalars, c, m, n, sparseSafe, 0, m, rix);
		}
		else  //MULTI-THREADED
		{
			try {
				ExecutorService pool = CommonThreadPool.get(k);
				ArrayList<ParAggTask> tasks = new ArrayList<>();
				int nk = UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
				int blklen = (int)(Math.ceil((double)m/nk));
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParAggTask(inputs.get(0), b, scalars,
						m, n, sparseSafe, i*blklen, Math.min((i+1)*blklen, m))); 
				//execute tasks
				List<Future<double[]>> taskret = pool.invokeAll(tasks);	
				pool.shutdown();
			
				//aggregate partial results
				ArrayList<double[]> pret = new ArrayList<>();
				for( Future<double[]> task : taskret )
					pret.add(task.get());
				aggregatePartialResults(c, pret);
			}
			catch(Exception ex) {
				throw new DMLRuntimeException(ex);
			}
		}
	
		//post-processing
		out.recomputeNonZeros();
		out.examSparsity();
		return out;
	}
	
	private void executeDense(DenseBlock a, SideInput[] b, double[] scalars, double[] c, int m, int n, boolean sparseSafe, int rl, int ru, long rix)
	{
		SideInput[] lb = createSparseSideInputs(b);
		
		//core dense aggregation operation
		if( a == null && !sparseSafe ) {
			for( int i=rl; i<ru; i++ )
				for( int j=0; j<n; j++ )
					genexec( 0, lb, scalars, c, m, n, rix+i, i, j );
		}
		else if( a != null ) {
			for( int i=rl; i<ru; i++ ) { 
				double[] avals = a.values(i);
				int aix = a.pos(i);
				for( int j=0; j<n; j++ )
					genexec( avals[aix+j], lb, scalars, c, m, n, rix+i, i, j );
			}
		}
	}
	
	private void executeSparse(SparseBlock sblock, SideInput[] b, double[] scalars,
			double[] c, int m, int n, boolean sparseSafe, int rl, int ru, long rix) 
	{
		if( sblock == null && sparseSafe )
			return;
		
		SideInput[] lb = createSparseSideInputs(b);
		
		//note: sequential scan algorithm for both sparse-safe and -unsafe
		//in order to avoid binary search for sparse-unsafe
		for(int i=rl; i<ru; i++) {
			int lastj = -1;
			//handle non-empty rows
			if( sblock != null && !sblock.isEmpty(i) ) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] avals = sblock.values(i);
				for(int k=apos; k<apos+alen; k++) {
					//process zeros before current non-zero
					if( !sparseSafe )
						for(int j=lastj+1; j<aix[k]; j++)
							genexec(0, lb, scalars, c, m, n, rix+i, i, j);
					//process current non-zero
					lastj = aix[k];
					genexec(avals[k], lb, scalars, c, m, n, rix+i, i, lastj);
				}
			}
			//process empty rows or remaining zeros
			if( !sparseSafe )
				for(int j=lastj+1; j<n; j++)
					genexec(0, lb, scalars, c, m, n, rix+i, i, j);
		}
	}
	
	//local execution where grix==rix
	protected final void genexec( double a, SideInput[] b,
		double[] scalars, double[] c, int m, int n, int rix, int cix) {
		genexec(a, b, scalars, c, m, n, rix, rix, cix);
	}
	
	//distributed execution with additional global row index
	protected abstract void genexec( double a, SideInput[] b,
		double[] scalars, double[] c, int m, int n, long grix, int rix, int cix);
	
	
	private void setInitialOutputValues(double[] c) {
		for( int k=0; k<_aggOps.length; k++ )
			c[k] = getInitialValue(_aggOps[k]);
	}
	
	public static double getInitialValue(AggOp aggop) {
		switch( aggop ) {
			case SUM:
			case SUM_SQ: return 0; 
			case MIN:    return Double.POSITIVE_INFINITY;
			case MAX:    return Double.NEGATIVE_INFINITY;
		}
		return 0;
	}
	

	private void aggregatePartialResults(double[] c, ArrayList<double[]> pret) {
		ValueFunction[] vfun = getAggFunctions(_aggOps); 
		for( int k=0; k<_aggOps.length; k++ ) {
			if( vfun[k] instanceof KahanFunction ) {
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				for(double[] tmp : pret)
					kplus.execute2(kbuff, tmp[k]);
				c[k] = kbuff._sum;
			}
			else {
				for(double[] tmp : pret)
					c[k] = vfun[k].execute(c[k], tmp[k]);
			}
		}
	}
	
	public static void aggregatePartialResults(AggOp[] aggOps, MatrixBlock c, MatrixBlock b) {
		ValueFunction[] vfun = getAggFunctions(aggOps);
		
		for( int k=0; k< aggOps.length; k++ ) {
			if( vfun[k] instanceof KahanFunction ) {
				KahanObject kbuff = new KahanObject(c.quickGetValue(0, k), 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				kplus.execute2(kbuff, b.quickGetValue(0, k));
				c.quickSetValue(0, k, kbuff._sum);
			}
			else {
				double cval = c.quickGetValue(0, k);
				double bval = b.quickGetValue(0, k);
				c.quickSetValue(0, k, vfun[k].execute(cval, bval));
			}
		}
	}

	public static ValueFunction[] getAggFunctions(AggOp[] aggOps) {
		ValueFunction[] fun = new ValueFunction[aggOps.length];
		for( int i=0; i<aggOps.length; i++ ) {
			switch( aggOps[i] ) {
				case SUM: fun[i] = KahanPlus.getKahanPlusFnObject(); break;
				case SUM_SQ: fun[i] = KahanPlusSq.getKahanPlusSqFnObject(); break;
				case MIN: fun[i] = Builtin.getBuiltinFnObject(BuiltinCode.MIN); break;
				case MAX: fun[i] = Builtin.getBuiltinFnObject(BuiltinCode.MAX); break;
				default:
					throw new RuntimeException("Unsupported "
							+ "aggregation type: "+aggOps[i].name());
			}
		}
		return fun;
	}
	
	private class ParAggTask implements Callable<double[]> 
	{
		private final MatrixBlock _a;
		private final SideInput[] _b;
		private final double[] _scalars;
		private final int _rlen;
		private final int _clen;
		private final boolean _safe;
		private final int _rl;
		private final int _ru;

		protected ParAggTask( MatrixBlock a, SideInput[] b, double[] scalars, 
				int rlen, int clen, boolean safe, int rl, int ru ) {
			_a = a;
			_b = b;
			_scalars = scalars;
			_rlen = rlen;
			_clen = clen;
			_safe = safe;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public double[] call() {
			double[] c = new double[_aggOps.length];
			setInitialOutputValues(c);
			if( !_a.isInSparseFormat() )
				executeDense(_a.getDenseBlock(), _b, _scalars, c, _rlen, _clen, _safe, _rl, _ru, 0);
			else
				executeSparse(_a.getSparseBlock(), _b, _scalars, c, _rlen, _clen, _safe, _rl, _ru, 0);
			return c;
		}
	}
}

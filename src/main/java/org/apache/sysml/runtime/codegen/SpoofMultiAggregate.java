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

import java.io.Serializable;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.codegen.SpoofCellwise.AggOp;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.functionobjects.KahanFunction;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.KahanPlusSq;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.instructions.cp.ScalarObject;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.util.UtilFunctions;

public abstract class SpoofMultiAggregate extends SpoofOperator implements Serializable
{
	private static final long serialVersionUID = -6164871955591089349L;
	private static final long PAR_NUMCELL_THRESHOLD = 1024*1024;   //Min 1M elements
	
	private final AggOp[] _aggOps;
	
	public SpoofMultiAggregate(AggOp... aggOps) {
		_aggOps = aggOps;
	}
	
	public AggOp[] getAggOps() {
		return _aggOps;
	}
	
	@Override
	public String getSpoofType() {
		return "MA" +  getClass().getName().split("\\.")[1];
	}
	
	@Override
	public void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out) 
		throws DMLRuntimeException
	{
		execute(inputs, scalarObjects, out, 1);
	}
	
	@Override
	public void execute(ArrayList<MatrixBlock> inputs, ArrayList<ScalarObject> scalarObjects, MatrixBlock out, int k)	
		throws DMLRuntimeException
	{
		//sanity check
		if( inputs==null || inputs.size() < 1  )
			throw new RuntimeException("Invalid input arguments.");
		
		if( inputs.get(0).getNumRows()*inputs.get(0).getNumColumns()<PAR_NUMCELL_THRESHOLD ) {
			k = 1; //serial execution
		}
	
		//result allocation and preparations
		out.reset(1, _aggOps.length, false);
		out.allocateDenseBlock();
		double[] c = out.getDenseBlock();
		setInitialOutputValues(c);
		
		//input preparation
		SideInput[] b = prepInputMatrices(inputs);
		double[] scalars = prepInputScalars(scalarObjects);
		final int m = inputs.get(0).getNumRows();
		final int n = inputs.get(0).getNumColumns();
		
		if( k <= 1 ) //SINGLE-THREADED
		{
			if( inputs.get(0) instanceof CompressedMatrixBlock )
				executeCompressed((CompressedMatrixBlock)inputs.get(0), b, scalars, c, m, n, 0, m);
			else if( !inputs.get(0).isInSparseFormat() )
				executeDense(inputs.get(0).getDenseBlock(), b, scalars, c, m, n, 0, m);
			else	
				executeSparse(inputs.get(0).getSparseBlock(), b, scalars, c, m, n, 0, m);
		}
		else  //MULTI-THREADED
		{
			try {
				ExecutorService pool = Executors.newFixedThreadPool( k );
				ArrayList<ParAggTask> tasks = new ArrayList<ParAggTask>();
				int nk = UtilFunctions.roundToNext(Math.min(8*k,m/32), k);
				int blklen = (int)(Math.ceil((double)m/nk));
				for( int i=0; i<nk & i*blklen<m; i++ )
					tasks.add(new ParAggTask(inputs.get(0), b, scalars, m, n, i*blklen, Math.min((i+1)*blklen, m))); 
				//execute tasks
				List<Future<double[]>> taskret = pool.invokeAll(tasks);	
				pool.shutdown();
			
				//aggregate partial results
				ArrayList<double[]> pret = new ArrayList<double[]>();
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
	}
	
	private void executeDense(double[] a, SideInput[] b, double[] scalars, double[] c, int m, int n, int rl, int ru) throws DMLRuntimeException 
	{
		//core dense aggregation operation
		for( int i=rl, ix=rl*n; i<ru; i++ ) { 
			for( int j=0; j<n; j++, ix++ ) {
				double in = (a != null) ? a[ix] : 0;
				genexec( in, b, scalars, c, m, n, i, j );
			}
		}
	}
	
	private void executeSparse(SparseBlock sblock, SideInput[] b, double[] scalars, double[] c, int m, int n, int rl, int ru) 
		throws DMLRuntimeException 
	{
		//core dense aggregation operation
		for( int i=rl; i<ru; i++ )
			for( int j=0; j<n; j++ ) {
				double in = (sblock != null) ? sblock.get(i, j) : 0;
				genexec( in, b, scalars, c, m, n, i, j );
			}
	}

	private void executeCompressed(CompressedMatrixBlock a, SideInput[] b, double[] scalars, double[] c, int m, int n, int rl, int ru) throws DMLRuntimeException 
	{
		//core compressed aggregation operation
		Iterator<IJV> iter = a.getIterator(rl, ru, true);
		while( iter.hasNext() ) {
			IJV cell = iter.next();
			genexec(cell.getV(), b, scalars, c, m, n, cell.getI(), cell.getJ());
		}
	}
	
	protected abstract void genexec( double a, SideInput[] b, double[] scalars, double[] c, int m, int n, int rowIndex, int colIndex);
	
	
	private void setInitialOutputValues(double[] c) {
		for( int k=0; k<_aggOps.length; k++ )
			c[k] = getInitialValue(_aggOps[k]);
	}
	
	public static double getInitialValue(AggOp aggop) {
		switch( aggop ) {
			case SUM:
			case SUM_SQ: return 0; 
			case MIN:    return Double.MAX_VALUE;
			case MAX:    return -Double.MAX_VALUE;
		}
		return 0;
	}
	

	private void aggregatePartialResults(double[] c, ArrayList<double[]> pret) 
		throws DMLRuntimeException 
	{
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
		
	public static void aggregatePartialResults(AggOp[] aggOps, MatrixBlock c, MatrixBlock b) 
		throws DMLRuntimeException 
	{
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
		private final int _rl;
		private final int _ru;

		protected ParAggTask( MatrixBlock a, SideInput[] b, double[] scalars, 
				int rlen, int clen, int rl, int ru ) {
			_a = a;
			_b = b;
			_scalars = scalars;
			_rlen = rlen;
			_clen = clen;
			_rl = rl;
			_ru = ru;
		}
		
		@Override
		public double[] call() throws DMLRuntimeException {
			double[] c = new double[_aggOps.length];
			setInitialOutputValues(c);
			if( _a instanceof CompressedMatrixBlock )
				executeCompressed((CompressedMatrixBlock)_a, _b, _scalars, c, _rlen, _clen, _rl, _ru);
			else if( !_a.isInSparseFormat() )
				executeDense(_a.getDenseBlock(), _b, _scalars, c, _rlen, _clen, _rl, _ru);
			else	
				executeSparse(_a.getSparseBlock(), _b, _scalars, c, _rlen, _clen, _rl, _ru);
			return c;
		}
	}
}

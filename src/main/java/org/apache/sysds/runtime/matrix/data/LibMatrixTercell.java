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
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Library for ternary cellwise operations.
 * 
 */
public class LibMatrixTercell 
{
	private static final long PAR_NUMCELL_THRESHOLD = 8*1024;
	
	private LibMatrixTercell() {
		//prevent instantiation via private constructor
	}
	
	public static void tercellOp(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret, TernaryOperator op)
	{
		final boolean s1 = (m1.rlen==1 && m1.clen==1);
		final boolean s2 = (m2.rlen==1 && m2.clen==1);
		final boolean s3 = (m3.rlen==1 && m3.clen==1);
		final double d1 = s1 ? m1.quickGetValue(0, 0) : Double.NaN;
		final double d2 = s2 ? m2.quickGetValue(0, 0) : Double.NaN;
		final double d3 = s3 ? m3.quickGetValue(0, 0) : Double.NaN;
		
		//allocate dense/sparse output
		ret.allocateBlock();
		
		//execute ternary cell operations
		if( op.getNumThreads() > 1 && ret.getLength() > PAR_NUMCELL_THRESHOLD) {
			try {
				//execute binary cell operations
				ExecutorService pool = CommonThreadPool.get(op.getNumThreads());
				ArrayList<TercellTask> tasks = new ArrayList<>();
				ArrayList<Integer> blklens = UtilFunctions
					.getBalancedBlockSizesDefault(ret.rlen, op.getNumThreads(), false);
				for( int i=0, lb=0; i<blklens.size(); lb+=blklens.get(i), i++ )
					tasks.add(new TercellTask(m1, m2, m3, ret, op, s1, s2, s3, d1, d2, d3, lb, lb+blklens.get(i)));
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				
				//aggregate non-zeros
				ret.nonZeros = 0; //reset after execute
				for( Future<Long> task : taskret )
					ret.nonZeros += task.get();
				pool.shutdown();
			}
			catch(InterruptedException | ExecutionException ex) {
				throw new DMLRuntimeException(ex);
			}
		}
		else {
			long nnz = unsafeTernary(m1, m2, m3, ret, op,
				s1, s2, s3, d1, d2, d3, 0, ret.rlen);
			ret.setNonZeros(nnz);
		}
	}
	
	private static long unsafeTernary(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret,
		TernaryOperator op, boolean s1, boolean s2, boolean s3, double d1, double d2, double d3, int rl, int ru)
	{
		//basic ternary operations (all combinations sparse/dense)
		int n = ret.clen;
		long lnnz = 0;
		for( int i=rl; i<ru; i++ )
			for( int j=0; j<n; j++ ) {
				double in1 = s1 ? d1 : m1.quickGetValue(i, j);
				double in2 = s2 ? d2 : m2.quickGetValue(i, j);
				double in3 = s3 ? d3 : m3.quickGetValue(i, j);
				double val = op.fn.execute(in1, in2, in3);
				lnnz += (val != 0) ? 1 : 0;
				ret.appendValuePlain(i, j, val);
			}
		
		//set global output nnz once
		return lnnz;
	}
	
	private static class TercellTask implements Callable<Long> {
		private final MatrixBlock _m1, _m2, _m3;
		private final boolean _s1, _s2, _s3;
		private final double _d1, _d2, _d3;
		private final MatrixBlock _ret;
		private final TernaryOperator _op;
		private final int _rl, _ru;

		protected TercellTask(MatrixBlock m1, MatrixBlock m2, MatrixBlock m3, MatrixBlock ret, TernaryOperator op,
			boolean s1, boolean s2, boolean s3, double d1, double d2, double d3, int rl, int ru) {
			_m1 = m1; _m2 = m2; _m3 = m3;
			_s1 = s1; _s2 = s2; _s3 = s3;
			_d1 = d1; _d2 = d2; _d3 = d3;
			_ret = ret;
			_op = op;
			_rl = rl; _ru = ru;
		}
		
		@Override
		public Long call() {
			// execute binary operation on row partition
			// (including nnz maintenance)
			return unsafeTernary(_m1, _m2, _m3, _ret, _op,
				_s1, _s2, _s3, _d1, _d2, _d3, _rl, _ru);
		}
	}
}

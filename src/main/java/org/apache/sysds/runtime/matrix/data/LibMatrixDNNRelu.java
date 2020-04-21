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
import java.util.concurrent.Callable;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;

/**
 * This class contains the different implementation of rotate180 operation
 */
public class LibMatrixDNNRelu
{
	/**
	 * Factory method that returns list of callable tasks for performing relu backward operation
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing relu backward operation
	 */
	public static ArrayList<Callable<Long>> getReluBackwardWorkers(DnnParameters params) {
		ArrayList<Callable<Long>> ret = new ArrayList<>();
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		for(int i = 0; i*taskSize < params.N; i++)
			ret.add(new ReluBackward(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
		return ret;
	}
	
	/**
	 * Performs the operation: (X gt 0) * dout
	 */
	public static class ReluBackward implements Callable<Long>
	{
		public final int _rl, _ru; 
		private final DnnParameters _params;
		public ReluBackward(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			MatrixBlock m1 = _params.input1;
			MatrixBlock m2 = _params.input2;
			MatrixBlock out = _params.output;
			final int n = m1.getNumColumns();
			if( m1.isEmptyBlock(false) || m2.isEmptyBlock(false) )
				return 0L; //nothing to do
			
			//compute c = (a > 0) * b
			//(if there is at least one sparse input, the output is allocated in sparse)
			if(!m1.isInSparseFormat() && !m2.isInSparseFormat())
				reluBackwardDenseDense(m1.getDenseBlock(), m2.getDenseBlock(), out.getDenseBlock(), n, _rl, _ru);
			else if(!m1.isInSparseFormat() && m2.isInSparseFormat())
				reluBackwardDenseSparse(m1.getDenseBlock(), m2.getSparseBlock(), out.getSparseBlock(), _rl, _ru);
			else if(m1.isInSparseFormat() && !m2.isInSparseFormat())
				reluBackwardSparseDense(m1.getSparseBlock(), m2.getDenseBlock(), out.getSparseBlock(), _rl, _ru);
			else //sparse-sparse
				reluBackwardSparseSparse(m1.getSparseBlock(), m2.getSparseBlock(), out.getSparseBlock(), _rl, _ru);
			
			//thread-local nnz maintenance
			return out.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	private static void reluBackwardDenseDense(DenseBlock a, DenseBlock b, DenseBlock c, int n, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			double[] avals = a.values(i), bvals = b.values(i);
			double[] cvals = c.values(i);
			int ix = a.pos(i);
			for(int j=0; j<n; j++)
				cvals[ix+j] = (avals[ix+j] > 0) ? bvals[ix +j] : 0;
		}
	}
	
	private static void reluBackwardDenseSparse(DenseBlock a, SparseBlock b, SparseBlock c, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			if( b.isEmpty(i) ) continue;
			int bpos = b.pos(i), blen = b.size(i);
			int[] bix = b.indexes(i);
			double[] bvals = b.values(i), avals = a.values(i);
			int aix = a.pos(i);
			c.allocate(i, blen);
			for(int k=bpos; k<bpos+blen; k++)
				c.append(i, bix[k], (avals[aix+bix[k]] > 0) ? bvals[k] : 0);
		}
	}
	
	private static void reluBackwardSparseDense(SparseBlock a, DenseBlock b, SparseBlock c, int rl, int ru) {
		for(int i = rl; i < ru; i++) {
			if( a.isEmpty(i) ) continue;
			int apos = a.pos(i), alen = a.size(i);
			int[] aix = a.indexes(i);
			double[] avals = a.values(i), bvals = b.values(i);
			int bix = b.pos(i);
			c.allocate(i, alen);
			for(int k=apos; k<apos+alen; k++)
				c.append(i, aix[k], (avals[k] > 0) ? bvals[bix+aix[k]] : 0);
		}
	}
	
	private static void reluBackwardSparseSparse(SparseBlock a, SparseBlock b, SparseBlock c, int rl, int ru) {
		//b is the driver as it has likely less non-zeros
		for(int i = rl; i < ru; i++) {
			if( a.isEmpty(i) || b.isEmpty(i) ) continue;
			int bpos = b.pos(i), blen = b.size(i);
			int[] bix = b.indexes(i);
			double[] bvals = b.values(i);
			c.allocate(i, blen);
			for(int k=bpos; k<bpos+blen; k++)
				c.append(i, bix[k], (a.get(i, bix[k]) > 0) ? bvals[k] : 0);
		}
	}
}

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
package org.apache.sysml.runtime.matrix.data;

import java.util.ArrayList;
import java.util.concurrent.Callable;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;

/**
 * This class contains the different implementation of rotate180 operation
 */
public class LibMatrixDNNRelu
{
	private static ScalarOperator GT0 = InstructionUtils.parseScalarBinaryOperator(">", false, 0);

	
	/**
	 * Factory method that returns list of callable tasks for performing relu backward operation
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing relu backward operation
	 * @throws DMLRuntimeException if error occurs
	 */
	public static ArrayList<Callable<Long>> getReluBackwardWorkers(ConvolutionParameters params) throws DMLRuntimeException {
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
		private final ConvolutionParameters _params;
		public ReluBackward(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			//note: X (m x n), dout (m x n) -> out (m x n)
			DenseBlock out = _params.output.getDenseBlock();
			final int n = _params.input1.getNumColumns();
			if(!_params.input1.isInSparseFormat() && !_params.input2.isInSparseFormat()) {
				DenseBlock x = _params.input1.getDenseBlock();
				DenseBlock dout = _params.input2.getDenseBlock();
				for(int i = _rl; i < _ru; i++) {
					double[] xvals = x.values(i), doutvals = dout.values(i), cvals = out.values(i);
					int xpos = x.pos(i), doutpos = dout.pos(i), cpos = out.pos(i);
					for(int j=0; j<n; j++)
						cvals[cpos+j] = xvals[xpos+j] > 0 ? doutvals[doutpos +j] : 0;
				}
			}
			else {
				scalarOperations(_params.input1, out, n, _rl, _ru, GT0);      // (X > 0)
				binaryOperationInPlacePlus(_params.input2, out, n, _rl, _ru); // (X > 0) * dout
			}
			return 0L;
		}
	}
	
	private static void scalarOperations(MatrixBlock src, DenseBlock c,
			int destNumCols, int src_rl, int src_ru, ScalarOperator op)
		throws DMLRuntimeException
	{
		if(src.isInSparseFormat()) {
			for(int i = src_rl; i < src_ru; i++) {
				if( src.getSparseBlock().isEmpty(i) ) continue;
				int apos = src.getSparseBlock().pos(i);
				int alen = src.getSparseBlock().size(i);
				int[] aix = src.getSparseBlock().indexes(i);
				double[] avals = src.getSparseBlock().values(i);
				double[] cvals = c.values(i);
				int cix = c.pos(i);
				for(int j = apos; j < apos+alen; j++)
					cvals[ cix+aix[j] ] = op.executeScalar(avals[j]);
			}
		}
		else {
			DenseBlock a = src.getDenseBlock();
			for(int i = src_rl; i < src_ru; i++) {
				double[] avals = a.values(i), cvals = c.values(i);
				int aix = a.pos(i), cix = c.pos(i);
				for(int j=0; j<destNumCols; j++)
					cvals[cix+j] = op.executeScalar(avals[aix+j]);
			}
		}
	}
	
	private static void binaryOperationInPlacePlus(MatrixBlock src,
		DenseBlock c, int destNumCols, int src_rl, int src_ru)
		throws DMLRuntimeException 
	{
		if( src.isEmptyBlock(false) )
			return; //do nothing (add 0);
		
		if(src.isInSparseFormat()) {
			for(int i = src_rl; i < src_ru; i++) {
				if( src.getSparseBlock().isEmpty(i) ) continue;
				int apos = src.getSparseBlock().pos(i);
				int alen = src.getSparseBlock().size(i);
				int[] aix = src.getSparseBlock().indexes(i);
				double[] avals = src.getSparseBlock().values(i);
				double[] cvals = c.values(i);
				int cix = c.pos(i);
				for(int j = apos; j < apos+alen; j++)
					cvals[ cix+aix[j] ] += avals[j];
			}
		}
		else { //DENSE
			DenseBlock a = src.getDenseBlock();
			for(int i = src_rl; i < src_ru; i++) {
				double[] avals = a.values(i), cvals = c.values(i);
				int aix = a.pos(i), cix = c.pos(i);
				for(int j=0; j<destNumCols; j++)
					cvals[cix+j] += avals[aix+j];
			}
		}
	}
}

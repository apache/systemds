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
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;

/**
 * This class contains the different implementation of rotate180 operation
 */
public class LibMatrixDNNRelu
{
	private static BinaryOperator PLUS = new BinaryOperator(Plus.getPlusFnObject());

	
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
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		double [] outputArray; int numOutCols;
		public ReluBackward(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
			outputArray= params.output.getDenseBlockValues();
			numOutCols = params.input1.getNumColumns();
		}
		
		@Override
		public Long call() throws Exception {
			if(!_params.input1.isInSparseFormat() && !_params.input2.isInSparseFormat()) {
				double [] inputArr = _params.input1.getDenseBlockValues();
				double [] doutArr = _params.input2.getDenseBlockValues();
				for(int i = _rl*numOutCols; i < _ru*numOutCols; i++) {
					outputArray[i] = inputArr[i] > 0 ? doutArr[i] : 0;
				}
			}
			else {
				// Perform (X > 0)
				ConvolutionUtils.scalarOperations(_params.input1, outputArray, _rl*numOutCols, numOutCols, _rl, _ru, 
					InstructionUtils.parseScalarBinaryOperator(">", false, 0));
				// Then perform (X > 0) * dout
				ConvolutionUtils.binaryOperationInPlace(_params.input2, outputArray, _rl*numOutCols, numOutCols, _rl, _ru, PLUS);
			}
			return 0L;
		}
	}
}

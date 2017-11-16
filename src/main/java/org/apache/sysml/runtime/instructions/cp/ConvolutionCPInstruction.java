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

package org.apache.sysml.runtime.instructions.cp;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.ConvolutionParameters;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysml.runtime.matrix.data.LibMatrixNative;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.NativeHelper;

public class ConvolutionCPInstruction extends UnaryCPInstruction {
	private static final Log LOG = LogFactory.getLog(ConvolutionCPInstruction.class.getName());
	private static boolean warnedUnderUtilitization = false;
	
	private final CPOperand _in2;
	private final CPOperand _in3;
	private final ArrayList<CPOperand> _input_shape;
	private final ArrayList<CPOperand> _filter_shape;
	private final ArrayList<CPOperand> _stride;
	private final ArrayList<CPOperand> _padding;
	private final int _numThreads;
	private final double _intermediateMemoryBudget;
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand in2, CPOperand in3, CPOperand out, 
			ArrayList<CPOperand> stride, ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget, String opcode, String istr) {
		super(CPType.Convolution, null, in, out, opcode, istr);
		_in2 = in2;
		_in3 = in3;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
		_numThreads = numThreads;
		_intermediateMemoryBudget = intermediateMemoryBudget;
	}
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode, String istr, int numThreads, double intermediateMemoryBudget) throws DMLRuntimeException {
		this(in, in2, null, out, null, null, null, null, numThreads, intermediateMemoryBudget, opcode, istr);
		if( !(opcode.equals("bias_add") || opcode.equals("relu_backward") || opcode.equals("bias_multiply") ) ) {
			throw new DMLRuntimeException("Incorrect usage. Expected the opcode to be bias_add or bias_multiply or relu_backward, but found " + opcode);
		}
	}
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand in2, CPOperand in3, CPOperand out, String opcode, String istr, int numThreads, double intermediateMemoryBudget) throws DMLRuntimeException {
		this(in, in2, in3, out, null, null, null, null, numThreads, intermediateMemoryBudget, opcode, istr);
		if( !opcode.equals("channel_sums") ) {
			throw new DMLRuntimeException("Incorrect usage. Expected the opcode to be channel_sums, but found " + opcode);
		}
	}

	private ConvolutionCPInstruction(CPOperand in, CPOperand out, String opcode, String istr,
			ArrayList<CPOperand> stride, ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget) {
		this(in, null, null, out, stride, padding, input_shape, filter_shape, numThreads, intermediateMemoryBudget, opcode, istr);
	}
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand in2, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget) {
		this(in, in2, null, out, stride, padding, input_shape, filter_shape, numThreads, intermediateMemoryBudget, opcode, istr);
	}
	
	public ConvolutionCPInstruction(CPOperand in, CPOperand in2, CPOperand in3, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape, int numThreads, double intermediateMemoryBudget) {
		this(in, in2, in3, out, stride, padding, input_shape, filter_shape, numThreads, intermediateMemoryBudget, opcode, istr);
	}

	public static ConvolutionCPInstruction parseInstruction(String str)
			throws DMLRuntimeException {

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("maxpooling") || opcode.equalsIgnoreCase("relu_maxpooling")) {
			InstructionUtils.checkNumFields(parts, 16);
			// stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			CPOperand in = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[14]);

			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
			stride.add(new CPOperand(parts[2]));
			stride.add(new CPOperand(parts[3]));
			padding.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			input_shape.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			filter_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			int k = Integer.parseInt(parts[15]);

			return new ConvolutionCPInstruction(in, out, opcode, str, stride,
					padding, input_shape, filter_shape, k, Double.parseDouble(parts[16]));
		} 
		else if (opcode.equalsIgnoreCase("maxpooling_backward") || opcode.equalsIgnoreCase("relu_maxpooling_backward")
				|| opcode.equalsIgnoreCase("conv2d")
				|| opcode.equalsIgnoreCase("conv2d_backward_filter")
				|| opcode.equalsIgnoreCase("conv2d_backward_data")) {
			InstructionUtils.checkNumFields(parts, 17);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[15]);

			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
			stride.add(new CPOperand(parts[3]));
			stride.add(new CPOperand(parts[4]));
			padding.add(new CPOperand(parts[5]));
			padding.add(new CPOperand(parts[6]));
			input_shape.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			input_shape.add(new CPOperand(parts[10]));
			filter_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			filter_shape.add(new CPOperand(parts[14]));
			int k = Integer.parseInt(parts[16]);

			return new ConvolutionCPInstruction(in, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape, k, Double.parseDouble(parts[17]));
		}
		else if (opcode.equalsIgnoreCase("conv2d_bias_add")) {
			InstructionUtils.checkNumFields(parts, 18);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[16]);

			ArrayList<CPOperand> stride = new ArrayList<>();
			ArrayList<CPOperand> padding = new ArrayList<>();
			ArrayList<CPOperand> input_shape = new ArrayList<>();
			ArrayList<CPOperand> filter_shape = new ArrayList<>();
			stride.add(new CPOperand(parts[4]));
			stride.add(new CPOperand(parts[5]));
			padding.add(new CPOperand(parts[6]));
			padding.add(new CPOperand(parts[7]));
			input_shape.add(new CPOperand(parts[8]));
			input_shape.add(new CPOperand(parts[9]));
			input_shape.add(new CPOperand(parts[10]));
			input_shape.add(new CPOperand(parts[11]));
			filter_shape.add(new CPOperand(parts[12]));
			filter_shape.add(new CPOperand(parts[13]));
			filter_shape.add(new CPOperand(parts[14]));
			filter_shape.add(new CPOperand(parts[15]));
			int k = Integer.parseInt(parts[17]);

			return new ConvolutionCPInstruction(in, in2, in3, out, opcode, str, stride,
					padding, input_shape, filter_shape, k, Double.parseDouble(parts[18]));
		}
		else if (opcode.equalsIgnoreCase("bias_add") || opcode.equals("relu_backward") || opcode.equalsIgnoreCase("bias_multiply") ) {
			InstructionUtils.checkNumFields(parts, 5);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			int k = Integer.parseInt(parts[4]);
			return new ConvolutionCPInstruction(in, in2, out, opcode, str, k, Double.parseDouble(parts[5]));
		}
		else if (opcode.equalsIgnoreCase("channel_sums")) {
			InstructionUtils.checkNumFields(parts, 4);
			CPOperand in = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand in3 = new CPOperand(parts[3]);
			CPOperand out = new CPOperand(parts[4]);
			return new ConvolutionCPInstruction(in, in2, in3, out, opcode, str, -1, 0);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ConvolutionCPInstruction: " + str);
		}
	}

	private static int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL, int index) 
			throws DMLRuntimeException {
		return (int) ec.getScalarInput(aL.get(index).getName(),
			aL.get(index).getValueType(), aL.get(index).isLiteral()).getLongValue();
	}
	
	public void processReluBackwardInstruction(ExecutionContext ec) throws DMLRuntimeException {
		// (X > 0) * dout
		MatrixBlock input = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock outputBlock;
		
		if( !input.isEmpty() && !dout.isEmpty() ) {
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), false);
			outputBlock.allocateDenseBlock();
			LibMatrixDNN.reluBackward(input, dout, outputBlock, _numThreads);
		}
		else
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), true);
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	public void processBiasAddInstruction(ExecutionContext ec) throws DMLRuntimeException {
		MatrixBlock input = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock bias = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock outputBlock = null;
		
		if(bias.getNumColumns() != 1) {
			throw new DMLRuntimeException("Expected the number of columns of bias matrix to be 1, but found " + bias.getNumColumns());
		}
		
		if(input.isEmpty() && bias.isEmpty()) {
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), true);
		}
		else if(bias.isEmpty()) {
			outputBlock = new MatrixBlock(input);
		}
		else {
			// As we always fill the output first with bias
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), false);
			outputBlock.allocateDenseBlock();
			LibMatrixDNN.biasAdd(input, bias, outputBlock, _numThreads);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	public void processBiasMultiplyInstruction(ExecutionContext ec) throws DMLRuntimeException {
		MatrixBlock input = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		MatrixBlock bias = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
		MatrixBlock outputBlock = null;
		
		if(bias.getNumColumns() != 1) {
			throw new DMLRuntimeException("Expected the number of columns of bias matrix to be 1, but found " + bias.getNumColumns());
		}
		
		if(bias.isEmpty()) {
			// Anything multiplied by zero is zero
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), true);
		}
		else {
			// As we always fill the output first with bias
			outputBlock = new MatrixBlock(input.getNumRows(), input.getNumColumns(), 
				input.isInSparseFormat()).allocateBlock();
			LibMatrixDNN.biasMultiply(input, bias, outputBlock, _numThreads);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	public void processChannelSumsInstruction(ExecutionContext ec) throws DMLRuntimeException {
		MatrixBlock input = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		int C = (int) ec.getScalarInput(_in2.getName(), _in2.getValueType(), _in2.isLiteral()).getLongValue();
		int HW = (int) ec.getScalarInput(_in3.getName(), _in3.getValueType(), _in3.isLiteral()).getLongValue();
		if(C*HW != input.getNumColumns()) {
			throw new DMLRuntimeException("Expected rows*cols" + C + "*" + HW + " to be equal to number of columns of input " + input.getNumColumns());
		}
		MatrixBlock outputBlock = null;
		if(input.isEmpty()) {
			outputBlock = new MatrixBlock(C, 1, true);
		}
		else {
			outputBlock = new MatrixBlock(C, 1, false).allocateBlock();
			double [] output = outputBlock.getDenseBlock();
			if(input.isInSparseFormat()) {
				SparseBlock sblock = input.getSparseBlock();
				for(int n = 0; n < input.getNumRows(); n++) {
					if( sblock.isEmpty(n) )
						continue;
					int apos = sblock.pos(n);
					int alen = sblock.size(n);
					int[] aix = sblock.indexes(n);
					double[] avals = sblock.values(n);
					
					// Iterate over the sparse block
					for(int j=apos; j<apos+alen; j++) {
						// Note: the input is of shape [N, CHW]
						int chw = aix[j];
						
						// Get individual zero-based c,h,w indexes from zero-based 'chw'
						int c = chw / HW;
						output[c] += avals[j];
					}
				}
			}
			else {
				double [] inArr = input.getDenseBlock();
				if(inArr != null) {
					KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
					for(int c = 0; c < C; c++) {
						KahanObject sum = new KahanObject(0.0, 0.0);
						for(int n = 0; n < input.getNumRows(); n++) {
							int index =  n*C*HW + c*HW;
							for(int hw = 0; hw < HW; hw++, index++) {
								kplus.execute2(sum, inArr[index]);
							}
						}
						output[c] = sum._sum;
					}
				}
			}
			outputBlock.recomputeNonZeros(getExtendedOpcode());
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	// Assumption: enableNative && NativeHelper.isNativeLibraryLoaded() is true
	// This increases the number of native calls. For example:the cases where filter is sparse but input is dense
	private static boolean isFilterSparse(MatrixBlock filter) throws DMLRuntimeException {
		long numElems = filter.getNumRows()*filter.getNumColumns();
		// if filter is less than 10 MB in dense format (which handles almost all the cases).
		// In fact, using threshold of 1 MB is still sufficient for common CNNs.
		if(filter.isInSparseFormat() && numElems < 10e+6)
			filter.sparseToDense(); 
		return filter.isInSparseFormat();
	}
	
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException {
		if (instOpcode.equalsIgnoreCase("bias_add")) {
			processBiasAddInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("bias_multiply")) {
			processBiasMultiplyInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("relu_backward")) {
			processReluBackwardInstruction(ec);
			return;
		}
		else if (instOpcode.equalsIgnoreCase("channel_sums")) {
			processChannelSumsInstruction(ec);
			return;
		}
		
		// acquire inputs
		MatrixBlock outputBlock = null;
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName(), getExtendedOpcode());
		int pad_h = getScalarInput(ec, _padding, 0);
		int pad_w = getScalarInput(ec, _padding, 1);
		int stride_h = getScalarInput(ec, _stride, 0);
		int stride_w = getScalarInput(ec, _stride, 1);

		int N = getScalarInput(ec, _input_shape, 0);
		int C = getScalarInput(ec, _input_shape, 1);
		int H = getScalarInput(ec, _input_shape, 2);
		int W = getScalarInput(ec, _input_shape, 3);

		int K = getScalarInput(ec, _filter_shape, 0);
		
		int R = getScalarInput(ec, _filter_shape, 2);
		int S = getScalarInput(ec, _filter_shape, 3);
		int P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
		int Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
		
		ConvolutionParameters params = new ConvolutionParameters(N, C, H, W, K, R, S, stride_h, stride_w, pad_h, pad_w, _numThreads);
		params.enableNative = NativeHelper.isNativeLibraryLoaded();
		if (instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("relu_maxpooling")) {
			if(matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(N, C*P*Q, true);
			}
			else {
				outputBlock = new MatrixBlock(N, C*P*Q, false).allocateBlock();
				if(instOpcode.equalsIgnoreCase("relu_maxpooling"))
					params.minValForMaxPoolOperations = 0;
				LibMatrixDNN.maxpooling(matBlock, outputBlock, params);
			}
		}
		else if (instOpcode.equalsIgnoreCase("maxpooling_backward") || instOpcode.equalsIgnoreCase("relu_maxpooling_backward")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(matBlock.isEmpty() || dout.isEmpty()) {
				outputBlock = new MatrixBlock(N, C*H*W, true);
			}
			else {
				outputBlock = new MatrixBlock(N, C*H*W, false).allocateBlock();
				if(instOpcode.equalsIgnoreCase("relu_maxpooling_backward"))
					params.minValForMaxPoolOperations = 0;
				LibMatrixDNN.maxpoolingBackward(matBlock, dout, outputBlock, params, 
					!instOpcode.equalsIgnoreCase("maxpooling_backward"));
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d")) {
			resetNumThreads(params, C*R*S, P*Q, matBlock.getNonZeros() / (matBlock.getNumRows()*matBlock.getNumColumns()));
			MatrixBlock filter = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(filter.isEmpty() || matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(N, K*P*Q, true);
			}
			else {
				outputBlock = new MatrixBlock(N, K*P*Q, false).allocateBlock();
				if(params.enableNative && !isFilterSparse(filter) && !matBlock.isInSparseFormat())
					LibMatrixNative.conv2d(matBlock, filter, outputBlock, params);
				else
					LibMatrixDNN.conv2d(matBlock, filter, outputBlock, params);
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_bias_add")) {
			resetNumThreads(params, C*R*S, P*Q, matBlock.getNonZeros() / (matBlock.getNumRows()*matBlock.getNumColumns()));
			MatrixBlock filter = ec.getMatrixInput(_in3.getName(), getExtendedOpcode());
			MatrixBlock bias = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(bias.getNumRows() != params.K || bias.getNumColumns() != 1) {
				throw new DMLRuntimeException("Incorrect shape of bias matrix: [" + bias.getNumRows() + " " + bias.getNumColumns() + "]. "
						+ "Expected: [" + params.K + ", 1]");
			}
			boolean isOutputConvEmpty = filter.isEmpty() || matBlock.isEmpty();
			if(isOutputConvEmpty && bias.isEmpty()) {
				// bias_add(empty mb, empty mb) = empty mb
				outputBlock = new MatrixBlock(N, K*P*Q, true);
			}
			else if(isOutputConvEmpty && !bias.isEmpty()) {
				// Add bias to empty output block
				// bias_add(empty mb, bias)
				outputBlock = new MatrixBlock(N, K*P*Q, false).allocateBlock();
				for(int n = 0;  n < params.N; n++) 
					ConvolutionUtils.fillBias(bias, outputBlock.getDenseBlock(), n, n+1, params.N, params.K, params.P*params.Q);
			}
			else {
				outputBlock = new MatrixBlock(N, K*P*Q, false).allocateBlock();
				if(!bias.isEmpty()) {
					// Handle situation where both input and filter are non empty, but bias is empty
					params.bias = bias;
				}
				if(params.enableNative && !isFilterSparse(filter) && !matBlock.isInSparseFormat())
					LibMatrixNative.conv2d(matBlock, filter, outputBlock, params);
				else
					LibMatrixDNN.conv2d(matBlock, filter, outputBlock, params);
			}
			ec.releaseMatrixInput(_in3.getName(), getExtendedOpcode());
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_backward_filter")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(dout.isEmpty() || matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(K, C*R*S, true);
			}
			else {
				outputBlock = new MatrixBlock(K, C*R*S, false).allocateBlock();
				if(params.enableNative && !matBlock.isInSparseFormat() && !dout.isInSparseFormat())
					LibMatrixNative.conv2dBackwardFilter(matBlock, dout, outputBlock, params);
				else
					LibMatrixDNN.conv2dBackwardFilter(matBlock, dout, outputBlock, params);
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else if (instOpcode.equalsIgnoreCase("conv2d_backward_data")) {
			MatrixBlock dout = ec.getMatrixInput(_in2.getName(), getExtendedOpcode());
			if(dout.isEmpty() || matBlock.isEmpty()) {
				outputBlock = new MatrixBlock(N, C * H * W, true);
			}
			else {
				outputBlock = new MatrixBlock(N, C * H * W, false).allocateBlock();
				if(params.enableNative && !isFilterSparse(matBlock) && !dout.isInSparseFormat())
					LibMatrixNative.conv2dBackwardData(matBlock, dout, outputBlock, params);
				else
					LibMatrixDNN.conv2dBackwardData(matBlock, dout, outputBlock, params);
			}
			ec.releaseMatrixInput(_in2.getName(), getExtendedOpcode());
		}
		else {
			throw new DMLRuntimeException("Unsupported op code " + instOpcode);
		}
		
		// release inputs/outputs
		ec.releaseMatrixInput(input1.getName(), getExtendedOpcode());
		ec.setMatrixOutput(getOutputVariableName(), outputBlock, getExtendedOpcode());
	}
	
	/**
	 * Reset the number of thread to respect the intermediate CP memory budget
	 * 
	 * @param params convolution parameters
	 * @param numRows number of rows of intermediate matrix used per thread
	 * @param numCols number of rows of intermediate matrix used per thread
	 * @param sparsity sparsity of intermediate matrix used per thread
	 */
	private void resetNumThreads(ConvolutionParameters params, int numRows, int numCols, double sparsity) {
		if(DMLScript.USE_ACCELERATOR) {
			double memBudget1Thread = OptimizerUtils.estimateSizeExactSparsity(numRows, numCols, sparsity);
			int limitedDegreeOfParallelism = (int) Math.floor(_intermediateMemoryBudget / memBudget1Thread);
			if(params.numThreads > limitedDegreeOfParallelism) {
				params.numThreads = limitedDegreeOfParallelism;
				if(!warnedUnderUtilitization)
					LOG.warn("CPU Under-utilization to respect the intermediate memory budget. To avoid this, please try reducing the mini-batch or forcing gpu execution.");
				warnedUnderUtilitization = true;
			}
		}
	}
}

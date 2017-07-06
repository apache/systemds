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
package org.apache.sysml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.functionobjects.SwapIndex;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysml.runtime.instructions.spark.functions.ExtractBlockForBinaryReblock;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.ConvolutionParameters;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNN;
import org.apache.sysml.runtime.matrix.data.LibMatrixNative;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.operators.ReorgOperator;
import org.apache.sysml.runtime.util.ConvolutionUtils;
import org.apache.sysml.utils.NativeHelper;

import scala.Tuple2;

public class ConvolutionSPInstruction extends UnarySPInstruction {
	private CPOperand _in2;
	private CPOperand _in3; 
	private ArrayList<CPOperand> _input_shape;
	private ArrayList<CPOperand> _filter_shape;
	private ArrayList<CPOperand> _stride = new ArrayList<CPOperand>();
	private ArrayList<CPOperand> _padding = new ArrayList<CPOperand>();

	public ConvolutionSPInstruction(CPOperand in, CPOperand out, String opcode,
			String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}

	public ConvolutionSPInstruction(CPOperand in, CPOperand in2, CPOperand out,
			String opcode, String istr, ArrayList<CPOperand> stride,
			ArrayList<CPOperand> padding, ArrayList<CPOperand> input_shape,
			ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_in2 = in2;
		_sptype = SPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}

	public ConvolutionSPInstruction(CPOperand in, CPOperand in2, CPOperand in3,
			CPOperand out, String opcode, String istr,
			ArrayList<CPOperand> stride, ArrayList<CPOperand> padding,
			ArrayList<CPOperand> input_shape, ArrayList<CPOperand> filter_shape) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_in2 = in2;
		_in3 = in3;
		_sptype = SPINSTRUCTION_TYPE.Convolution;
		_stride = stride;
		_padding = padding;
		_input_shape = input_shape;
		_filter_shape = filter_shape;
	}

	public ConvolutionSPInstruction(CPOperand in, CPOperand in2, CPOperand out,
			String opcode, String istr) {
		super(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out,
				opcode, istr);
		_in2 = in2;
		_sptype = SPINSTRUCTION_TYPE.Convolution;
	}

	public static ConvolutionSPInstruction parseInstruction( String str ) throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);

		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if (opcode.equalsIgnoreCase("maxpooling") || opcode.equalsIgnoreCase("relu_maxpooling")) {
			InstructionUtils.checkNumFields(parts, 14);
			// stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			in.split(parts[1]);
			out.split(parts[14]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
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

			return new ConvolutionSPInstruction(in, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		} 
		else if (opcode.equalsIgnoreCase("maxpooling_backward")
				|| opcode.equalsIgnoreCase("conv2d")
				|| opcode.equalsIgnoreCase("conv2d_backward_filter")
				|| opcode.equalsIgnoreCase("conv2d_backward_data")) {
			InstructionUtils.checkNumFields(parts, 15);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			in.split(parts[1]);
			CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in2.split(parts[2]);
			out.split(parts[15]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
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

			return new ConvolutionSPInstruction(in, in2, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		}
		else if (opcode.equalsIgnoreCase("conv2d_bias_add")) {
			InstructionUtils.checkNumFields(parts, 16);
			// dout, stride1, stride2, padding1, padding2
			// input_shape1, input_shape2, input_shape3, input_shape4,
			// filter_shape1, filter_shape2, filter_shape3, filter_shape4, k
			in.split(parts[1]);
			CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in2.split(parts[2]);
			CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in3.split(parts[3]);
			out.split(parts[16]);

			ArrayList<CPOperand> stride = new ArrayList<CPOperand>();
			ArrayList<CPOperand> padding = new ArrayList<CPOperand>();
			ArrayList<CPOperand> input_shape = new ArrayList<CPOperand>();
			ArrayList<CPOperand> filter_shape = new ArrayList<CPOperand>();
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

			return new ConvolutionSPInstruction(in, in2, in3, out, opcode, str, stride,
					padding, input_shape, filter_shape);
		}
		else if (opcode.equalsIgnoreCase("bias_add")) {
			InstructionUtils.checkNumFields(parts, 3);
			in.split(parts[1]);
			CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			in2.split(parts[2]);
			out.split(parts[3]);
			return new ConvolutionSPInstruction(in, in2, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ConvolutionCPInstruction: " + str);
		}
	}
	
	private JavaPairRDD<MatrixIndexes,MatrixBlock> reblockAsRectangularMatrices(SparkExecutionContext sec, String name, int numRowsPerBlock) throws DMLRuntimeException {
		JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( name );
		MatrixCharacteristics mcRdd = sec.getMatrixCharacteristics(name);
		if(mcRdd.getColsPerBlock() < mcRdd.getCols() || mcRdd.getRowsPerBlock() != 1) {
			MatrixCharacteristics mcOut = new MatrixCharacteristics(mcRdd);
			mcOut.setColsPerBlock((int)mcRdd.getCols());
			mcOut.setRowsPerBlock(numRowsPerBlock); 
			in1 = RDDAggregateUtils.mergeByKey(in1.flatMapToPair(new ExtractBlockForBinaryReblock(mcRdd, mcOut)));
			// TODO: Inject checkpoint to avoid doing this repeated for validation set
//			sec.setRDDHandleForVariable(name, in1);
//			sec.setMetaData(name, new MatrixDimensionsMetaData(mcOut));
		}
		return in1;
	}
	
	private Broadcast<MatrixBlock> getBroadcast(SparkExecutionContext sec, String name) throws DMLRuntimeException {
		MatrixBlock mb = sec.getMatrixInput( name );
		sec.releaseMatrixInput(name);
		return sec.getSparkContext().broadcast(mb);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		if(instOpcode.equalsIgnoreCase("conv2d") || instOpcode.equalsIgnoreCase("conv2d_bias_add")
			|| instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("relu_maxpooling")) {
			String rddVar = input1.getName();
			int numRowsPerBlock = 1;
			JavaPairRDD<MatrixIndexes,MatrixBlock> inputRDD = reblockAsRectangularMatrices(sec, rddVar, numRowsPerBlock);
			MatrixCharacteristics mcRdd = sec.getMatrixCharacteristics(rddVar);
			
			// ------------------------------------
			// TODO: Handle large filters > 2G
			Broadcast<MatrixBlock> filterBroadcast = null;
			Broadcast<MatrixBlock> biasBroadcast = null;
			if(instOpcode.equalsIgnoreCase("conv2d")) {
				filterBroadcast = getBroadcast(sec, _in2.getName());
			}
			else if(instOpcode.equalsIgnoreCase("conv2d_bias_add")) {
				filterBroadcast = getBroadcast(sec, _in3.getName());
				biasBroadcast = getBroadcast(sec, _in2.getName());
			}
			// ------------------------------------
			
			int pad_h = getScalarInput(ec, _padding, 0);
			int pad_w = getScalarInput(ec, _padding, 1);
			int stride_h = getScalarInput(ec, _stride, 0);
			int stride_w = getScalarInput(ec, _stride, 1);

			// int N = getScalarInput(ec, _input_shape, 0);
			int C = getScalarInput(ec, _input_shape, 1);
			int H = getScalarInput(ec, _input_shape, 2);
			int W = getScalarInput(ec, _input_shape, 3);

			int K = getScalarInput(ec, _filter_shape, 0);
			int R = getScalarInput(ec, _filter_shape, 2);
			int S = getScalarInput(ec, _filter_shape, 3);
			int P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
			int Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
			
			ConvolutionParameters params = new ConvolutionParameters(numRowsPerBlock, C, H, W, K, R, S, stride_h, stride_w, pad_h, pad_w, 1);
			boolean enableNativeBLAS = NativeHelper.isNativeLibraryLoaded(); 
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = inputRDD.mapPartitionsToPair(new RDDConv2dMapMMFunction(filterBroadcast, params, instOpcode, biasBroadcast, mcRdd.getRows(), enableNativeBLAS), true);
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
			
			long nnz = -1; // TODO: Handle nnz
			long numCols = ((long)K)*((long)P)*((long)Q);
			if(instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("relu_maxpooling")) {
				numCols = ((long)C)*((long)P)*((long)Q);
			}
			if(numCols > Integer.MAX_VALUE) {
				throw new DMLRuntimeException("The current operator doesnot support large outputs.");
			}
			sec.setMetaData(output.getName(), 
					new MatrixFormatMetaData(new MatrixCharacteristics(mcRdd.getRows(), numCols, numRowsPerBlock, (int)numCols, nnz), OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo));
		}
		else {
			throw new DMLRuntimeException("Not implemented: " + instOpcode);
		}
	}

	private int getScalarInput(ExecutionContext ec, ArrayList<CPOperand> aL,
			int index) throws DMLRuntimeException {
		return (int) ec.getScalarInput(aL.get(index).getName(),
				aL.get(index).getValueType(), aL.get(index).isLiteral())
				.getLongValue();
	}
	
	private static class RDDConv2dMapMMFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes, MatrixBlock>>, MatrixIndexes, MatrixBlock> {
	// PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = -2106155380020232155L;
		Broadcast<MatrixBlock> filterBroadcast = null;
		Broadcast<MatrixBlock> biasBroadcast = null;
		ConvolutionParameters params = null;
		String instOpcode = null; boolean enableNative;
		long numRows = 0;
		public RDDConv2dMapMMFunction(Broadcast<MatrixBlock> filterBroadcast, 
				ConvolutionParameters params, String instOpcode, Broadcast<MatrixBlock> biasBroadcast, long numRows, boolean enableNativeBLAS) {
			this.filterBroadcast = filterBroadcast;
			this.params = params;
			this.instOpcode = instOpcode;
			this.biasBroadcast = biasBroadcast;
			this.numRows = numRows;
			this.enableNative = enableNativeBLAS;
		}
		
		private MatrixBlock processRectangularBlock(MatrixBlock matBlock) throws Exception {
			MatrixBlock outputBlock = null;
			if(instOpcode.equalsIgnoreCase("conv2d")) {
				MatrixBlock filter = filterBroadcast.getValue();
				if(filter.isEmptyBlock() || matBlock.isEmptyBlock()) {
					outputBlock = new MatrixBlock(params.N, params.K*params.P*params.Q, true);
				}
				else {
					outputBlock = getDenseOutputBlock(params.N, params.K*params.P*params.Q);
					if(enableNative)
						LibMatrixNative.conv2d(matBlock, filter, outputBlock, params);
					else
						LibMatrixDNN.conv2d(matBlock, filter, outputBlock, params);
				}
			}
			else if (instOpcode.equalsIgnoreCase("conv2d_bias_add")) {
				MatrixBlock filter = filterBroadcast.getValue();
				MatrixBlock bias = biasBroadcast.getValue();
				if((filter.isEmptyBlock() || matBlock.isEmptyBlock()) && bias.isEmptyBlock()) {
					outputBlock = new MatrixBlock(params.N, params.K*params.P*params.Q, true);
				}
				else {
					outputBlock = getDenseOutputBlock(params.N, params.K*params.P*params.Q);
					if(!bias.isEmptyBlock())
						params.bias = bias;
					if(enableNative)
						LibMatrixNative.conv2d(matBlock, filter, outputBlock, params);
					else
						LibMatrixDNN.conv2d(matBlock, filter, outputBlock, params);
				}
			}
			else if(instOpcode.equalsIgnoreCase("maxpooling") || instOpcode.equalsIgnoreCase("relu_maxpooling")) {
				if(matBlock.isEmptyBlock()) {
					outputBlock = new MatrixBlock(params.N, params.C*params.P*params.Q, true);
				}
				else {
					outputBlock = getDenseOutputBlock(params.N, params.C*params.P*params.Q);
					if(instOpcode.equalsIgnoreCase("maxpooling"))
						Arrays.fill(outputBlock.getDenseBlock(), -Double.MAX_VALUE);
					LibMatrixDNN.maxpooling(matBlock, outputBlock, params);
				}
			}
			else {
				throw new RuntimeException("Not implemented");
			}
			return outputBlock;
		}
		
		private MatrixBlock getDenseOutputBlock(int numRows, int numCols) throws DMLRuntimeException {
			MatrixBlock outputBlock = new MatrixBlock(numRows, numCols, false);
			outputBlock.allocateDenseBlock();
			return outputBlock;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(
				Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
				throws Exception {
			return new MapsideConvolutionPartitionIterator(arg0);
		}
		
		// Avoid materialization of partitions
		private class MapsideConvolutionPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>> {
			public MapsideConvolutionPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg) throws Exception {
				if(arg._1.getRowIndex() > numRows || arg._1.getColumnIndex() != 1) {
					throw new RuntimeException("Expected the inputs to be reblocked as rectangular RDD");
				}
				MatrixBlock out = processRectangularBlock(arg._2);
				if(out.getNumRows() != 1) {
					throw new RuntimeException("Expected the output to have 1 row");
				}
				return new Tuple2<MatrixIndexes, MatrixBlock>(arg._1, out);
			}
		}
		
	}
}

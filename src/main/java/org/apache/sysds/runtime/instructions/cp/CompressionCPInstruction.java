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

package org.apache.sysds.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.SingletonLookupHashMap;
import org.apache.sysds.runtime.compress.lib.CLALibBinCompress;
import org.apache.sysds.runtime.compress.workload.WTreeRoot;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameLibCompress;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class CompressionCPInstruction extends ComputationCPInstruction {
	private static final Log LOG = LogFactory.getLog(CompressionCPInstruction.class.getName());

	private final int _singletonLookupID;
	private final int _numThreads;

	/** This is set to true only for quantization-fused compression */
	private final boolean _quantizationFused;

	/** This is only for binned compression with 2 outputs */
	protected final List<CPOperand> _outputs;

	private CompressionCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr,
		int singletonLookupID, int numThreads) {
		super(CPType.Compression, op, in, null, null, out, opcode, istr);
		_outputs = null;
		this._singletonLookupID = singletonLookupID;
		this._numThreads = numThreads;
		this._quantizationFused = false;
	}

	private CompressionCPInstruction(Operator op, CPOperand in1, CPOperand in2, List<CPOperand> out, String opcode,
		String istr, int singletonLookupID, int numThreads) {
		super(CPType.Compression, op, in1, in2, null, out.get(0), opcode, istr);
		_outputs = out;
		this._singletonLookupID = singletonLookupID;
		this._numThreads = numThreads;
		this._quantizationFused = false;
	}

	private CompressionCPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand out, String opcode,
	String istr, int singletonLookupID, int numThreads) {
		super(CPType.QuantizeCompression, op, in1, in2, null, out, opcode, istr);
		_outputs = null;
		this._singletonLookupID = singletonLookupID;
		this._numThreads = numThreads;
		this._quantizationFused = true;
	}	

	public static CompressionCPInstruction parseInstruction(String str) {
		InstructionUtils.checkNumFields(str, 3, 4, 5);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		if(parts.length == 6) {
			/** Compression with bins that returns two outputs */
			List<CPOperand> outputs = new ArrayList<>();
			outputs.add(new CPOperand(parts[3]));
			outputs.add(new CPOperand(parts[4]));
			int numThreads = Integer.parseInt(parts[5]);
			return new CompressionCPInstruction(null, in1, out, outputs, opcode, str, 0, numThreads);
		}
		else if(parts.length == 5) {
			int treeNodeID = Integer.parseInt(parts[3]);
			int numThreads = Integer.parseInt(parts[4]);
			return new CompressionCPInstruction(null, in1, out, opcode, str, treeNodeID, numThreads);
		}
		else {

			int numThreads = Integer.parseInt(parts[3]);
			return new CompressionCPInstruction(null, in1, out, opcode, str, 0, numThreads);
		}
	}

	public static CompressionCPInstruction parseQuantizationFusedInstruction(String str) {
		InstructionUtils.checkNumFields(str, 3, 4, 5);
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
  		CPOperand out = new CPOperand(parts[3]);
		int numThreads = Integer.parseInt(parts[4]);
		return new CompressionCPInstruction(null, in1, in2, out, opcode, str, 0, numThreads);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(input2 == null)
			processSimpleCompressInstruction(ec);
		else if (this._quantizationFused == true)
			processSimpleQuantizationFusedCompressInstruction(ec);
		else
			processCompressByBinInstruction(ec);
	}

	private void processCompressByBinInstruction(ExecutionContext ec) {
		final MatrixBlock d = ec.getMatrixInput(input2.getName());

		final int k = OptimizerUtils.getConstrainedNumThreads(-1);

		Pair<MatrixBlock, FrameBlock> out;

		if(ec.isMatrixObject(input1.getName())) {
			final MatrixBlock X = ec.getMatrixInput(input1.getName());
			out = CLALibBinCompress.binCompress(X, d, k);
			ec.releaseMatrixInput(input1.getName());
		}
		else {
			final FrameBlock X = ec.getFrameInput(input1.getName());
			out = CLALibBinCompress.binCompress(X, d, k);
			ec.releaseFrameInput(input1.getName());
		}

		// Set output and release input
		ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(_outputs.get(0).getName(), out.getKey());
		ec.setFrameOutput(_outputs.get(1).getName(), out.getValue());
	}

	private void processSimpleCompressInstruction(ExecutionContext ec) {
		// final MatrixBlock in = ec.getMatrixInput(input1.getName());
		final SingletonLookupHashMap m = SingletonLookupHashMap.getMap();

		// Get and clear workload tree entry for this compression instruction.
		final WTreeRoot root = (_singletonLookupID != 0) ? (WTreeRoot) m.get(_singletonLookupID) : null;
		// We used to remove the key from the hash map, 
		// however this is not correct since the compression statement 
		// can be reused in multiple for loops.


		if(ec.isFrameObject(input1.getName()))
			processFrameBlockCompression(ec, ec.getFrameInput(input1.getName()), _numThreads, root);
		else if(ec.isMatrixObject(input1.getName()))
			processMatrixBlockCompression(ec, ec.getMatrixInput(input1.getName()), _numThreads, root);
		else {
			LOG.warn("Compression on Scalar should not happen");
			ScalarObject Scalar = ec.getScalarInput(input1);
			ec.setScalarOutput(output.getName(),Scalar);
		}
	}

	private void processSimpleQuantizationFusedCompressInstruction(ExecutionContext ec) {
		// final MatrixBlock in = ec.getMatrixInput(input1.getName());
		final SingletonLookupHashMap m = SingletonLookupHashMap.getMap();

		// Get and clear workload tree entry for this compression instruction.
		final WTreeRoot root = (_singletonLookupID != 0) ? (WTreeRoot) m.get(_singletonLookupID) : null;
		// We used to remove the key from the hash map, 
		// however this is not correct since the compression statement 
		// can be reused in multiple for loops.

		ScalarObject scalarIn2 = null;
		MatrixBlock matrixIn2 = null;

		if (input2.isScalar() == true) {
			scalarIn2 = ec.getScalarInput(input2);
			processMatrixBlockQuantizationFusedCompression(ec, ec.getMatrixInput(input1.getName()), scalarIn2, _numThreads, root);
		} else if (input2.isMatrix() == true) {
			matrixIn2 = ec.getMatrixInput(input2.getName());
			processMatrixBlockQuantizationFusedCompression(ec, ec.getMatrixInput(input1.getName()), matrixIn2, _numThreads, root);
		}
	}

	private void processMatrixBlockCompression(ExecutionContext ec, MatrixBlock in, int k, WTreeRoot root) {
		Pair<MatrixBlock, CompressionStatistics> compResult = CompressedMatrixBlockFactory.compress(in, k, root);
		if(LOG.isTraceEnabled())
			LOG.trace(compResult.getRight());
		MatrixBlock out = compResult.getLeft();
		if(LOG.isInfoEnabled())
			LOG.info("Compression output class: " + out.getClass().getSimpleName());
		// Set output and release input
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), out);
	}

	private void processFrameBlockCompression(ExecutionContext ec, FrameBlock in, int k, WTreeRoot root) {
		FrameBlock compResult = FrameLibCompress.compress(in, k, root);
		// Set output and release input
		ec.releaseFrameInput(input1.getName());
		ec.setFrameOutput(output.getName(), compResult);
	}

	private void processMatrixBlockQuantizationFusedCompression(ExecutionContext ec, MatrixBlock in1, MatrixBlock in2, int k, WTreeRoot root) {
		Pair<MatrixBlock, CompressionStatistics> compResult = CompressedMatrixBlockFactory.compress(in1, in2, k, root);
		if(LOG.isTraceEnabled())
			LOG.trace(compResult.getRight());
		MatrixBlock out = compResult.getLeft();
		if(LOG.isInfoEnabled())
			LOG.info("Compression output class: " + out.getClass().getSimpleName());
		// Set output and release input
		ec.releaseMatrixInput(input1.getName());
		ec.releaseMatrixInput(input2.getName());
		ec.setMatrixOutput(output.getName(), out);
	}

	private void processMatrixBlockQuantizationFusedCompression(ExecutionContext ec, MatrixBlock in1, ScalarObject in2, int k, WTreeRoot root) {
		Pair<MatrixBlock, CompressionStatistics> compResult = CompressedMatrixBlockFactory.compress(in1, in2, k, root);
		if(LOG.isTraceEnabled())
			LOG.trace(compResult.getRight());
		MatrixBlock out = compResult.getLeft();
		if(LOG.isInfoEnabled())
			LOG.info("Compression output class: " + out.getClass().getSimpleName());
		// Set output and release input
		ec.releaseMatrixInput(input1.getName());
		if (input2.isMatrix()) {
			ec.releaseMatrixInput(input2.getName());
		}
		ec.setMatrixOutput(output.getName(), out);
	}
}

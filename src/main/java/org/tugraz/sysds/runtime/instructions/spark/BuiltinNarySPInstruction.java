/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.functionobjects.Builtin;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObject;
import org.tugraz.sysds.runtime.instructions.spark.AppendGSPInstruction.ShiftMatrix;
import org.tugraz.sysds.runtime.instructions.spark.functions.MapInputSignature;
import org.tugraz.sysds.runtime.instructions.spark.functions.MapJoinSignature;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.operators.SimpleOperator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.List;

public class BuiltinNarySPInstruction extends SPInstruction 
{
	private CPOperand[] inputs;
	private CPOperand output;
	
	protected BuiltinNarySPInstruction(CPOperand[] in, CPOperand out, String opcode, String istr) {
		super(SPType.BuiltinNary, opcode, istr);
		inputs = in;
		output = out;
	}

	public static BuiltinNarySPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand output = new CPOperand(parts[parts.length - 1]);
		CPOperand[] inputs = null;
		inputs = new CPOperand[parts.length - 2];
		for (int i = 1; i < parts.length-1; i++)
			inputs[i-1] = new CPOperand(parts[i]);
		return new BuiltinNarySPInstruction(inputs, output, opcode, str);
	}

	@Override 
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		DataCharacteristics mcOut = null;
		
		if( getOpcode().equals("cbind") || getOpcode().equals("rbind") ) {
			//compute output characteristics
			boolean cbind = getOpcode().equals("cbind");
			mcOut = computeAppendOutputDataCharacteristics(sec, inputs, cbind);
			
			//get consolidated input via union over shifted and padded inputs
			DataCharacteristics off = new MatrixCharacteristics(0, 0, mcOut.getBlocksize(), 0);
			for( CPOperand input : inputs ) {
				DataCharacteristics mcIn = sec.getDataCharacteristics(input.getName());
				JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec
					.getBinaryMatrixBlockRDDHandleForVariable( input.getName() )
					.flatMapToPair(new ShiftMatrix(off, mcIn, cbind))
					.mapToPair(new PadBlocksFunction(mcOut)); //just padding
				out = (out != null) ? out.union(in) : in;
				updateAppendDataCharacteristics(mcIn, off, cbind);
			}
			
			//aggregate partially overlapping blocks w/ single shuffle
			int numPartOut = SparkUtils.getNumPreferredPartitions(mcOut);
			out = RDDAggregateUtils.mergeByKey(out, numPartOut, false);
		}
		else if( getOpcode().equals("nmin") || getOpcode().equals("nmax") ) {
			//compute output characteristics
			mcOut = computeMinMaxOutputDataCharacteristics(sec, inputs);
			
			//get scalars and consolidated input via join
			List<ScalarObject> scalars = sec.getScalarInputs(inputs);
			JavaPairRDD<MatrixIndexes, MatrixBlock[]> in = null;
			for( CPOperand input : inputs ) {
				if( !input.getDataType().isMatrix() ) continue;
				JavaPairRDD<MatrixIndexes, MatrixBlock> tmp = sec
					.getBinaryMatrixBlockRDDHandleForVariable(input.getName());
				in = (in == null) ? tmp.mapValues(new MapInputSignature()) :
					in.join(tmp).mapValues(new MapJoinSignature());
			}
			
			//compute nary min/max (partitioning-preserving)
			out = in.mapValues(new MinMaxFunction(getOpcode(), scalars));
		}
		
		//set output RDD and add lineage
		sec.getDataCharacteristics(output.getName()).set(mcOut);
		sec.setRDDHandleForVariable(output.getName(), out);
		for( CPOperand input : inputs )
			if( !input.isScalar() )
				sec.addLineageRDD(output.getName(), input.getName());
	}
	
	private static DataCharacteristics computeAppendOutputDataCharacteristics(SparkExecutionContext sec, CPOperand[] inputs, boolean cbind) {
		DataCharacteristics mcIn1 = sec.getDataCharacteristics(inputs[0].getName());
		DataCharacteristics mcOut = new MatrixCharacteristics(0, 0, mcIn1.getBlocksize(), 0);
		for( CPOperand input : inputs ) {
			DataCharacteristics mcIn = sec.getDataCharacteristics(input.getName());
			updateAppendDataCharacteristics(mcIn, mcOut, cbind);
		}
		return mcOut;
	}
	
	private static void updateAppendDataCharacteristics(DataCharacteristics in, DataCharacteristics out, boolean cbind) {
		out.setDimension(cbind ? Math.max(out.getRows(), in.getRows()) : out.getRows()+in.getRows(),
			cbind ? out.getCols()+in.getCols() : Math.max(out.getCols(), in.getCols()));
		out.setNonZeros((out.getNonZeros()!=-1 && in.dimsKnown(true)) ? out.getNonZeros()+in.getNonZeros() : -1);
	}
	
	private static DataCharacteristics computeMinMaxOutputDataCharacteristics(SparkExecutionContext sec, CPOperand[] inputs) {
		DataCharacteristics mcOut = new MatrixCharacteristics();
		for( CPOperand input : inputs ) {
			if( !input.getDataType().isMatrix() ) continue;
			DataCharacteristics mcIn = sec.getDataCharacteristics(input.getName());
			mcOut.setRows(Math.max(mcOut.getRows(), mcIn.getRows()));
			mcOut.setCols(Math.max(mcOut.getCols(), mcIn.getCols()));
			mcOut.setBlocksize(mcIn.getBlocksize());
		}
		return mcOut;
	}
	
	public static class PadBlocksFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 1291358959908299855L;
		
		private final DataCharacteristics _mcOut;
		
		public PadBlocksFunction(DataCharacteristics mcOut) {
			_mcOut = mcOut;
		}

		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			int brlen = UtilFunctions.computeBlockSize(_mcOut.getRows(), ix.getRowIndex(), _mcOut.getBlocksize());
			int bclen = UtilFunctions.computeBlockSize(_mcOut.getCols(), ix.getColumnIndex(), _mcOut.getBlocksize());
			
			//check for pass-through
			if( brlen == mb.getNumRows() && bclen == mb.getNumColumns() )
				return arg0;
			
			//cbind or rbind to pad to right blocksize
			if( brlen > mb.getNumRows() ) //rbind
				mb = mb.append(new MatrixBlock(brlen-mb.getNumRows(),bclen,true), new MatrixBlock(), false);
			else if( bclen > mb.getNumColumns() ) //cbind
				mb = mb.append(new MatrixBlock(brlen,bclen-mb.getNumColumns(),true), new MatrixBlock(), true);
			return new Tuple2<>(ix, mb);
		}
	}
	
	private static class MinMaxFunction implements Function<MatrixBlock[], MatrixBlock> {
		private static final long serialVersionUID = -4227447915387484397L;
		
		private final SimpleOperator _op;
		private final ScalarObject[] _scalars;
		
		public MinMaxFunction(String opcode, List<ScalarObject> scalars) {
			_scalars = scalars.toArray(new ScalarObject[0]);
			_op = new SimpleOperator(Builtin.getBuiltinFnObject(opcode.substring(1)));
		}
		
		@Override
		public MatrixBlock call(MatrixBlock[] v1) throws Exception {
			return MatrixBlock.naryOperations(_op, v1, _scalars, new MatrixBlock());
		}
	}
}

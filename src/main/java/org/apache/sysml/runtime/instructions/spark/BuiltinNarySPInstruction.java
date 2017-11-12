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

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.AppendGSPInstruction.ShiftMatrix;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.UtilFunctions;

import scala.Tuple2;

public class BuiltinNarySPInstruction extends SPInstruction 
{
	private CPOperand[] inputs;
	private CPOperand output;
	
	protected BuiltinNarySPInstruction(CPOperand[] in, CPOperand out, String opcode, String istr) {
		super(opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.BuiltinNary;
		inputs = in;
		output = out;
	}

	public static BuiltinNarySPInstruction parseInstruction ( String str ) 
			throws DMLRuntimeException 
	{
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
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		boolean cbind = getOpcode().equals("cbind");
		
		//compute output characteristics
		MatrixCharacteristics mcOut = computeOutputMatrixCharacteristics(sec, inputs, cbind);
		
		//get consolidated input via union over shifted and padded inputs
		MatrixCharacteristics off = new MatrixCharacteristics(
			0, 0, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), 0);
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		for( CPOperand input : inputs ) {
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(input.getName());
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec
				.getBinaryBlockRDDHandleForVariable( input.getName() )
				.flatMapToPair(new ShiftMatrix(off, mcIn, cbind))
				.mapToPair(new PadBlocksFunction(mcOut)); //just padding
			out = (out != null) ? out.union(in) : in;
			updateMatrixCharacteristics(mcIn, off, cbind);
		}
		
		//aggregate partially overlapping blocks w/ single shuffle
		int numPartOut = SparkUtils.getNumPreferredPartitions(mcOut);
		out = RDDAggregateUtils.mergeByKey(out, numPartOut, false);
		
		//set output RDD and add lineage
		sec.getMatrixCharacteristics(output.getName()).set(mcOut);
		sec.setRDDHandleForVariable(output.getName(), out);
		for( CPOperand input : inputs )
			sec.addLineageRDD(output.getName(), input.getName());
	}
	
	private static MatrixCharacteristics computeOutputMatrixCharacteristics(SparkExecutionContext sec, CPOperand[] inputs, boolean cbind) 
		throws DMLRuntimeException 
	{
		MatrixCharacteristics mcIn1 = sec.getMatrixCharacteristics(inputs[0].getName());
		MatrixCharacteristics mcOut = new MatrixCharacteristics(
			0, 0, mcIn1.getRowsPerBlock(), mcIn1.getColsPerBlock(), 0);
		for( CPOperand input : inputs ) {
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(input.getName());
			updateMatrixCharacteristics(mcIn, mcOut, cbind);
		}
		return mcOut;
	}
	
	private static void updateMatrixCharacteristics(MatrixCharacteristics in, MatrixCharacteristics out, boolean cbind) {
		out.setDimension(cbind ? Math.max(out.getRows(), in.getRows()) : out.getRows()+in.getRows(),
			cbind ? out.getCols()+in.getCols() : Math.max(out.getCols(), in.getCols()));
		out.setNonZeros((out.getNonZeros()!=-1 && in.dimsKnown(true)) ? out.getNonZeros()+in.getNonZeros() : -1);
	}
	
	public static class PadBlocksFunction implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 1291358959908299855L;
		
		private final MatrixCharacteristics _mcOut;
		
		public PadBlocksFunction(MatrixCharacteristics mcOut) {
			_mcOut = mcOut;
		}

		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			int brlen = UtilFunctions.computeBlockSize(_mcOut.getRows(), ix.getRowIndex(), _mcOut.getRowsPerBlock());
			int bclen = UtilFunctions.computeBlockSize(_mcOut.getCols(), ix.getColumnIndex(), _mcOut.getColsPerBlock());
			
			//check for pass-through
			if( brlen == mb.getNumRows() && bclen == mb.getNumColumns() )
				return arg0;
			
			//cbind or rbind to pad to right blocksize
			if( brlen > mb.getNumRows() ) //rbind
				mb = mb.appendOperations(new MatrixBlock(brlen-mb.getNumRows(),bclen,true), new MatrixBlock(), false);
			else if( bclen > mb.getNumColumns() ) //cbind
				mb = mb.appendOperations(new MatrixBlock(brlen,bclen-mb.getNumColumns(),true), new MatrixBlock(), true);
			return new Tuple2<>(ix, mb);
		}
	}
}

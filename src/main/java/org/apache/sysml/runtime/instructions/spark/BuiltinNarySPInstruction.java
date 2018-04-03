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
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.UtilFunctions;

import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

public class BuiltinNarySPInstruction extends SPInstruction 
{
	private CPOperand[] inputs;
	private CPOperand output;
	
	protected BuiltinNarySPInstruction(CPOperand[] in, CPOperand out, String opcode, String istr) {
		super(SPType.BuiltinNary, opcode, istr);
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
		//decide upon broadcast side inputs
		boolean[] bcVect = determineBroadcastInputs(sec, inputs);
		
		//compute output characteristics
		MatrixCharacteristics mcOut = computeOutputMatrixCharacteristics(sec, inputs, cbind);
		
		//get consolidated input via union over shifted and padded inputs
		MatrixCharacteristics off = new MatrixCharacteristics(
			0, 0, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), 0);
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		for(int i = 0; i < inputs.length; i++){
			CPOperand input = inputs[i];
			MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(input.getName());
			//get broadcast matrix
			PartitionedBroadcast<MatrixBlock> bcMatrix = bcVect[i]?sec.getBroadcastForVariable(input.getName()):null;
			JavaPairRDD<MatrixIndexes, MatrixBlock> in = sec
					.getBinaryBlockRDDHandleForVariable(input.getName())
					.flatMapToPair(new ShiftMatrixFunction(off, mcIn, cbind, bcMatrix))
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
		for(int i = 0; i<inputs.length; i++){
			CPOperand input = inputs[i];
			sec.addLineage(output.getName(), input.getName(), bcVect[i]);
		}
	}

	private static boolean[] determineBroadcastInputs(SparkExecutionContext sec, CPOperand[] inputs)
			throws DMLRuntimeException
	{
		boolean[] ret = new boolean[inputs.length];
		double localBudget = OptimizerUtils.getLocalMemBudget()
				- CacheableData.getBroadcastSize(); //account for other broadcasts
		double bcBudget = SparkExecutionContext.getBroadcastMemoryBudget();

		//decided for each matrix input if it fits into remaining memory
		//budget; the major input, i.e., inputs[0] is always an RDD
		for( int i=0; i<inputs.length; i++ )
			if( inputs[i].getDataType().isMatrix() ) {
				MatrixCharacteristics mc = sec.getMatrixCharacteristics(inputs[i].getName());
				double sizeL = OptimizerUtils.estimateSizeExactSparsity(mc);
				double sizeP = OptimizerUtils.estimatePartitionedSizeExactSparsity(mc);
				//account for partitioning and local/remote budgets
				ret[i] = localBudget > (sizeL + sizeP) && bcBudget > sizeP;
				localBudget -= ret[i] ? sizeP : 0; //in local block manager
				bcBudget -= ret[i] ? sizeP : 0; //in remote block managers
			}

		return ret;
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

	public static class ShiftMatrixFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock>
	{
		private static final long serialVersionUID = 3524189212798209172L;

		private boolean _cbind;
		private long _startIx;
		private int _shiftBy;
		private int _blen;
		private long _outlen;
		private PartitionedBroadcast<MatrixBlock> _bcMatrix;

		public ShiftMatrixFunction(MatrixCharacteristics mc1, MatrixCharacteristics mc2, boolean cbind, PartitionedBroadcast<MatrixBlock> bcMatrix)
		{
			_cbind = cbind;
			_startIx = cbind ? UtilFunctions.computeBlockIndex(mc1.getCols(), mc1.getColsPerBlock()) :
					UtilFunctions.computeBlockIndex(mc1.getRows(), mc1.getRowsPerBlock());
			_blen = (int) (cbind ? mc1.getColsPerBlock() : mc1.getRowsPerBlock());
			_shiftBy = (int) (cbind ? mc1.getCols()%_blen : mc1.getRows()%_blen);
			_outlen = cbind ? mc1.getCols()+mc2.getCols() : mc1.getRows()+mc2.getRows();
			_bcMatrix = bcMatrix;
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv)
				throws Exception
		{

			//common preparation
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<>();
			MatrixIndexes ix = kv._1();
			MatrixBlock in;
			if(_bcMatrix != null){
				int rowIndex = (int)(_bcMatrix.getNumRowBlocks()>=ix.getRowIndex()?ix.getRowIndex():1);
				int colIndex = (int) (_bcMatrix.getNumColumnBlocks()>=ix.getColumnIndex()?ix.getColumnIndex():1);
				in = _bcMatrix.getBlock(rowIndex, colIndex);
			}
			else {
				in = kv._2();
			}

			int cutAt = _blen - _shiftBy;

			if( _cbind )
			{
				MatrixIndexes firstIndex = new MatrixIndexes(ix.getRowIndex(), ix.getColumnIndex()+_startIx-1);
				MatrixIndexes secondIndex = new MatrixIndexes(ix.getRowIndex(), ix.getColumnIndex()+_startIx);

				int lblen1 = UtilFunctions.computeBlockSize(_outlen, firstIndex.getColumnIndex(), _blen);
				if(cutAt >= in.getNumColumns()) {
					// The block is too small to be cut
					MatrixBlock firstBlk = new MatrixBlock(in.getNumRows(), lblen1, true);
					firstBlk = firstBlk.leftIndexingOperations(in, 0, in.getNumRows()-1, lblen1-in.getNumColumns(), lblen1-1, new MatrixBlock(), MatrixObject.UpdateType.INPLACE_PINNED);
					retVal.add(new Tuple2<>(firstIndex, firstBlk));
				}
				else {
					// Since merge requires the dimensions matching, shifting = slicing + left indexing
					MatrixBlock firstSlicedBlk = in.sliceOperations(0, in.getNumRows()-1, 0, cutAt-1, new MatrixBlock());
					MatrixBlock firstBlk = new MatrixBlock(in.getNumRows(), lblen1, true);
					firstBlk = firstBlk.leftIndexingOperations(firstSlicedBlk, 0, in.getNumRows()-1, _shiftBy, _blen-1, new MatrixBlock(), MatrixObject.UpdateType.INPLACE_PINNED);

					MatrixBlock secondSlicedBlk = in.sliceOperations(0, in.getNumRows()-1, cutAt, in.getNumColumns()-1, new MatrixBlock());
					int llen2 = UtilFunctions.computeBlockSize(_outlen, secondIndex.getColumnIndex(), _blen);
					MatrixBlock secondBlk = new MatrixBlock(in.getNumRows(), llen2, true);
					secondBlk = secondBlk.leftIndexingOperations(secondSlicedBlk, 0, in.getNumRows()-1, 0, secondSlicedBlk.getNumColumns()-1, new MatrixBlock(), MatrixObject.UpdateType.INPLACE_PINNED);
					retVal.add(new Tuple2<>(firstIndex, firstBlk));
					retVal.add(new Tuple2<>(secondIndex, secondBlk));
				}
			}
			else //rbind
			{
				MatrixIndexes firstIndex = new MatrixIndexes(ix.getRowIndex()+_startIx-1, ix.getColumnIndex());
				MatrixIndexes secondIndex = new MatrixIndexes(ix.getRowIndex()+_startIx, ix.getColumnIndex());

				int lblen1 = UtilFunctions.computeBlockSize(_outlen, firstIndex.getRowIndex(), _blen);
				if(cutAt >= in.getNumRows()) {
					// The block is too small to be cut
					MatrixBlock firstBlk = new MatrixBlock(lblen1, in.getNumColumns(), true);
					firstBlk = firstBlk.leftIndexingOperations(in, lblen1-in.getNumRows(), lblen1-1, 0, in.getNumColumns()-1, new MatrixBlock(), MatrixObject.UpdateType.INPLACE_PINNED);
					retVal.add(new Tuple2<>(firstIndex, firstBlk));
				}
				else {
					// Since merge requires the dimensions matching, shifting = slicing + left indexing
					MatrixBlock firstSlicedBlk = in.sliceOperations(0, cutAt-1, 0, in.getNumColumns()-1, new MatrixBlock());
					MatrixBlock firstBlk = new MatrixBlock(lblen1, in.getNumColumns(), true);
					firstBlk = firstBlk.leftIndexingOperations(firstSlicedBlk, _shiftBy, _blen-1, 0, in.getNumColumns()-1, new MatrixBlock(), MatrixObject.UpdateType.INPLACE_PINNED);

					MatrixBlock secondSlicedBlk = in.sliceOperations(cutAt, in.getNumRows()-1, 0, in.getNumColumns()-1, new MatrixBlock());
					int lblen2 = UtilFunctions.computeBlockSize(_outlen, secondIndex.getRowIndex(), _blen);
					MatrixBlock secondBlk = new MatrixBlock(lblen2, in.getNumColumns(), true);
					secondBlk = secondBlk.leftIndexingOperations(secondSlicedBlk, 0, secondSlicedBlk.getNumRows()-1, 0, in.getNumColumns()-1, new MatrixBlock(), MatrixObject.UpdateType.INPLACE_PINNED);

					retVal.add(new Tuple2<>(firstIndex, firstBlk));
					retVal.add(new Tuple2<>(secondIndex, secondBlk));
				}
			}

			return retVal.iterator();
		}
	}
}

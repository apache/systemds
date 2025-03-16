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

package org.apache.sysds.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.functions.IsFrameBlockInRange;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * This class implements the frame indexing functionality inside Spark.
 *  
 */
public class FrameIndexingSPInstruction extends IndexingSPInstruction {

	protected FrameIndexingSPInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, SparkAggType aggtype, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, aggtype, opcode, istr);
	}

	protected FrameIndexingSPInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
			CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();
		
		//get indexing range
		long rl = ec.getScalarInput(rowLower).getLongValue();
		long ru = ec.getScalarInput(rowUpper).getLongValue();
		long cl = ec.getScalarInput(colLower).getLongValue();
		long cu = ec.getScalarInput(colUpper).getLongValue();
		IndexRange ixrange = new IndexRange(rl, ru, cl, cu);
		
		//right indexing
		if( opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) )
		{
			//update and check output dimensions
			DataCharacteristics mcIn = sec.getDataCharacteristics(input1.getName());
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
			mcOut.set(ru-rl+1, cu-cl+1, mcIn.getBlocksize(), mcIn.getBlocksize());
			checkValidOutputDimensions(mcOut);
			
			//execute right indexing operation (partitioning-preserving if possible)
			JavaPairRDD<Long,FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<Long,FrameBlock> out = null;
			if( isPartitioningPreservingRightIndexing(mcIn, ixrange) ) {
				out = in1.mapPartitionsToPair(
						new SliceBlockPartitionFunction(ixrange, mcOut), true);
			}
			else{
				out = in1.filter(new IsFrameBlockInRange(rl, ru, mcOut))
			             .mapToPair(new SliceBlock(ixrange, mcOut));
			}
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			
			//update schema of output with subset of input schema
			sec.getFrameObject(output.getName()).setSchema(
				sec.getFrameObject(input1.getName()).getSchema((int)cl, (int)cu));
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPLEFTINDEX.toString()))
		{
			JavaPairRDD<Long,FrameBlock> in1 = sec.getFrameBinaryBlockRDDHandleForVariable( input1.getName() );
			PartitionedBroadcast<FrameBlock> broadcastIn2 = null;
			JavaPairRDD<Long,FrameBlock> in2 = null;
			JavaPairRDD<Long,FrameBlock> out = null;
			
			//update and check output dimensions
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
			DataCharacteristics mcLeft = ec.getDataCharacteristics(input1.getName());
			mcOut.set(mcLeft.getRows(), mcLeft.getCols(), mcLeft.getBlocksize(), mcLeft.getBlocksize());
			checkValidOutputDimensions(mcOut);
			
			//note: always frame rhs, scalars are preprocessed via cast to 1x1 frame
			DataCharacteristics mcRight = ec.getDataCharacteristics(input2.getName());
				
			//sanity check matching index range and rhs dimensions
			if(!mcRight.dimsKnown()) {
				throw new DMLRuntimeException("The right input frame dimensions are not specified for FrameIndexingSPInstruction");
			}
			if(!(ru-rl+1 == mcRight.getRows() && cu-cl+1 == mcRight.getCols())) {
				throw new DMLRuntimeException("Invalid index range of leftindexing: ["+rl+":"+ru+","+cl+":"+cu+"] vs ["+mcRight.getRows()+"x"+mcRight.getCols()+"]." );
			}
			
			if(opcode.equalsIgnoreCase(Opcodes.MAPLEFTINDEX.toString()))
			{
				broadcastIn2 = sec.getBroadcastForFrameVariable( input2.getName());
				
				//partitioning-preserving mappartitions (key access required for broadcast loopkup)
				out = in1.mapPartitionsToPair(
						new LeftIndexPartitionFunction(broadcastIn2, ixrange, mcOut), true);
			}
			else { //general case
				
				// zero-out lhs
				in1 = in1.flatMapToPair(new ZeroOutLHS(false, ixrange, mcLeft));
				
				// slice rhs, shift and merge with lhs
				in2 = sec.getFrameBinaryBlockRDDHandleForVariable( input2.getName() )
					    .flatMapToPair(new SliceRHSForLeftIndexing(ixrange, mcLeft));

				out = FrameRDDAggregateUtils.mergeByKey(in1.union(in2));
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			if( broadcastIn2 != null)
				sec.addLineageBroadcast(output.getName(), input2.getName());
			if(in2 != null) 
				sec.addLineageRDD(output.getName(), input2.getName());
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in FrameIndexingSPInstruction.");		
	}

	private static boolean isPartitioningPreservingRightIndexing(DataCharacteristics mcIn, IndexRange ixrange) {
		return ( mcIn.dimsKnown() &&
			(ixrange.rowStart==1 && ixrange.rowEnd==mcIn.getRows() ));   //Entire Column/s
	}

	private static void checkValidOutputDimensions(DataCharacteristics mcOut) {
		if(!mcOut.dimsKnown()) {
			throw new DMLRuntimeException("FrameIndexingSPInstruction: The updated output dimensions are invalid: " + mcOut);
		}
	}

	private static class SliceRHSForLeftIndexing implements PairFlatMapFunction<Tuple2<Long,FrameBlock>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = 5724800998701216440L;

		private IndexRange _ixrange = null; 
		private int _blen = -1;
		private long _rlen = -1;
		private long _clen = -1;
		
		public SliceRHSForLeftIndexing(IndexRange ixrange, DataCharacteristics mcLeft) {
			_ixrange = ixrange;
			_rlen = mcLeft.getRows();
			_clen = mcLeft.getCols();
			_blen = (int) Math.min(OptimizerUtils.getDefaultFrameSize(), _rlen);
			_blen = (int) mcLeft.getCols();
		}

		@Override
		public Iterator<Tuple2<Long, FrameBlock>> call(Tuple2<Long, FrameBlock> rightKV) 
			throws Exception 
		{
			Pair<Long,FrameBlock> in = SparkUtils.toIndexedFrameBlock(rightKV);
			ArrayList<Pair<Long,FrameBlock>> out = new ArrayList<>();
			OperationsOnMatrixValues.performShift(in, _ixrange, _blen, _rlen, _clen, out);
			return SparkUtils.fromIndexedFrameBlock(out).iterator();
		}
	}

	private static class ZeroOutLHS implements PairFlatMapFunction<Tuple2<Long,FrameBlock>, Long,FrameBlock> 
	{
		private static final long serialVersionUID = -2672267231152496854L;

		private boolean _complement = false;
		private IndexRange _ixrange = null;
		private int _blen = -1;
		private long _rlen = -1;
		
		public ZeroOutLHS(boolean complement, IndexRange range, DataCharacteristics mcLeft) {
			_complement = complement;
			_ixrange = range;
			_blen = OptimizerUtils.getDefaultFrameSize();
			_blen = (int) mcLeft.getCols();
			_rlen = mcLeft.getRows();
		}
		
		@Override
		public Iterator<Tuple2<Long, FrameBlock>> call(Tuple2<Long, FrameBlock> kv) 
			throws Exception 
		{
			ArrayList<Pair<Long,FrameBlock>> out = new ArrayList<>();

			IndexRange curBlockRange = new IndexRange(_ixrange.rowStart, _ixrange.rowEnd, _ixrange.colStart, _ixrange.colEnd);
			
			// Global index of row (1-based)
			long lGblStartRow = ((kv._1.longValue()-1)/_blen)*_blen+1;
			FrameBlock zeroBlk = null;
			int iMaxRowsToCopy = 0;
			
			// Starting local location (0-based) of target block where to start copy. 
			int iRowStartDest = UtilFunctions.computeCellInBlock(kv._1, _blen);
			for(int iRowStartSrc = 0; iRowStartSrc<kv._2.getNumRows(); iRowStartSrc += iMaxRowsToCopy, lGblStartRow += _blen) {
				IndexRange range = UtilFunctions.getSelectedRangeForZeroOut(
					new Pair<>(kv._1, kv._2), _blen, curBlockRange, lGblStartRow-1, lGblStartRow);
				if(range.rowStart == -1 && range.rowEnd == -1 && range.colStart == -1 && range.colEnd == -1) {
					throw new Exception("Error while getting range for zero-out");
				}
				//Maximum range of rows in target block 
				int iMaxRows=(int) Math.min(_blen, _rlen-lGblStartRow+1);
				
				// Maximum number of rows to be copied from source block to target.
				iMaxRowsToCopy = Math.min(iMaxRows, kv._2.getNumRows()-iRowStartSrc);
				iMaxRowsToCopy = Math.min(iMaxRowsToCopy, iMaxRows-iRowStartDest);
				
				// Zero out the applicable range in this block
				zeroBlk = kv._2.zeroOutOperations(new FrameBlock(), range, _complement, iRowStartSrc, iRowStartDest, iMaxRows, iMaxRowsToCopy);
				out.add(new Pair<>(lGblStartRow, zeroBlk));
				curBlockRange.rowStart =  lGblStartRow + _blen;
				iRowStartDest = UtilFunctions.computeCellInBlock(iRowStartDest+iMaxRowsToCopy+1, _blen);
			}
			return SparkUtils.fromIndexedFrameBlock(out).iterator();
		}
	}

	private static class LeftIndexPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<Long,FrameBlock>>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = -911940376947364915L;

		private PartitionedBroadcast<FrameBlock> _binput;
		private IndexRange _ixrange = null;
		
		public LeftIndexPartitionFunction(PartitionedBroadcast<FrameBlock> binput, IndexRange ixrange, DataCharacteristics mc)
		{
			_binput = binput;
			_ixrange = ixrange;
		}

		@Override
		public LazyIterableIterator<Tuple2<Long, FrameBlock>> call(Iterator<Tuple2<Long, FrameBlock>> arg0)
			throws Exception 
		{
			return new LeftIndexPartitionIterator(arg0);
		}

		private class LeftIndexPartitionIterator extends LazyIterableIterator<Tuple2<Long, FrameBlock>>
		{
			public LeftIndexPartitionIterator(Iterator<Tuple2<Long, FrameBlock>> in) {
				super(in);
			}
			
			@Override
			protected Tuple2<Long, FrameBlock> computeNext(Tuple2<Long, FrameBlock> arg) 
				throws Exception 
			{
				int iNumRowsInBlock = arg._2.getNumRows();
				int iNumCols = arg._2.getNumColumns();
				if(!UtilFunctions.isInFrameBlockRange(arg._1(), iNumRowsInBlock, _ixrange)) {
					return arg;
				}
				
				// Calculate global index of left hand side block
				long lhs_rl = Math.max(_ixrange.rowStart, arg._1); //Math.max(_ixrange.rowStart, (arg._1-1)*iNumRowsInBlock + 1);
				long lhs_ru = Math.min(_ixrange.rowEnd, arg._1+iNumRowsInBlock-1);
				long lhs_cl = Math.max(_ixrange.colStart, 1);
				long lhs_cu = Math.min(_ixrange.colEnd, iNumCols);
				
				// Calculate global index of right hand side block
				long rhs_rl = lhs_rl - _ixrange.rowStart + 1;
				long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
				long rhs_cl = lhs_cl - _ixrange.colStart + 1;
				long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
				
				// Provide local zero-based index to leftIndexingOperations
				int lhs_lrl = (int)(lhs_rl- arg._1);
				int lhs_lru = (int)(lhs_ru- arg._1);
				int lhs_lcl = (int)lhs_cl-1;
				int lhs_lcu = (int)lhs_cu-1;

				FrameBlock ret = arg._2;
				int blen = OptimizerUtils.DEFAULT_BLOCKSIZE;
				long rhs_rl_pb = rhs_rl;
				long rhs_ru_pb = Math.min(rhs_ru, (((rhs_rl-1)/blen)+1)*blen); 
				while(rhs_rl_pb <= rhs_ru_pb) {
					// Provide global zero-based index to sliceOperations, but only for one RHS partition block at a time.
					FrameBlock slicedRHSMatBlock = _binput.slice(rhs_rl_pb, rhs_ru_pb, rhs_cl, rhs_cu, new FrameBlock());
					
					// Provide local zero-based index to leftIndexingOperations
					int lhs_lrl_pb = (int) (lhs_lrl + (rhs_rl_pb - rhs_rl));
					int lhs_lru_pb = (int) (lhs_lru + (rhs_ru_pb - rhs_ru));
					ret = ret.leftIndexingOperations(slicedRHSMatBlock, lhs_lrl_pb, lhs_lru_pb, lhs_lcl, lhs_lcu, new FrameBlock());
					rhs_rl_pb = rhs_ru_pb + 1;
					rhs_ru_pb = Math.min(rhs_ru, rhs_ru_pb+blen);
				}
				
				return new Tuple2<>(arg._1, ret);
			}
		}
	}

	private static class SliceBlock implements PairFunction<Tuple2<Long, FrameBlock>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = -5270171193018691692L;
		
		private IndexRange _ixrange;
		
		public SliceBlock(IndexRange ixrange, DataCharacteristics mcOut) {
			_ixrange = ixrange;
		}

		@Override
		public Tuple2<Long, FrameBlock> call(Tuple2<Long, FrameBlock> kv) 
			throws Exception 
		{	
			long rowindex = kv._1();
			FrameBlock in = kv._2();
			
			//prepare local index range (block guaranteed to be in range)
			int rl = (int) ((rowindex > _ixrange.rowStart) ? 0 : _ixrange.rowStart-rowindex);
			int ru = (int) ((_ixrange.rowEnd-rowindex >= in.getNumRows()) ? 
					in.getNumRows()-1 : _ixrange.rowEnd-rowindex);
			
			//slice out the block
			FrameBlock out = in.slice(rl, ru, (int)(_ixrange.colStart-1), 
					(int)(_ixrange.colEnd-1), new FrameBlock());
			
			//return block with shifted row index
			long rowindex2 = (rowindex > _ixrange.rowStart) ? rowindex-_ixrange.rowStart+1 : 1; 
			return new Tuple2<>(rowindex2, out);
		}		
	}

	private static class SliceBlockPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<Long, FrameBlock>>, Long, FrameBlock> 
	{
		private static final long serialVersionUID = -1655390518299307588L;
		
		private IndexRange _ixrange;
		
		public SliceBlockPartitionFunction(IndexRange ixrange, DataCharacteristics mcOut) {
			_ixrange = ixrange;
		}

		@Override
		public LazyIterableIterator<Tuple2<Long, FrameBlock>> call(Iterator<Tuple2<Long, FrameBlock>> arg0)
			throws Exception 
		{
			return new SliceBlockPartitionIterator(arg0);
		}	
		
		/**
		 * NOTE: this function is only applied for slicing columns (which preserved all rows
		 * and hence the existing partitioning). 
		 */
		private class SliceBlockPartitionIterator extends LazyIterableIterator<Tuple2<Long, FrameBlock>>
		{
			public SliceBlockPartitionIterator(Iterator<Tuple2<Long, FrameBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<Long, FrameBlock> computeNext(Tuple2<Long, FrameBlock> arg)
				throws Exception
			{
				long rowindex = arg._1();
				FrameBlock in = arg._2();
				
				//slice out the block
				FrameBlock out = in.slice(0, in.getNumRows()-1, 
						(int)_ixrange.colStart-1, (int)_ixrange.colEnd-1, new FrameBlock());
				
				//return block with shifted row index
				return new Tuple2<>(rowindex, out);
			}			
		}
	}
}

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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.spark.Partitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.rdd.PartitionPruningRDD;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LeftIndex;
import org.apache.sysds.lops.LeftIndex.LixCacheType;
import org.apache.sysds.lops.RightIndex;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.instructions.spark.data.LazyIterableIterator;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.instructions.spark.functions.IsBlockInRange;
import org.apache.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import scala.Function1;
import scala.Tuple2;
import scala.reflect.ClassManifestFactory;
import scala.runtime.AbstractFunction1;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

/**
 * This class implements the matrix indexing functionality inside CP.
 */
public class MatrixIndexingSPInstruction extends IndexingSPInstruction {
	private final LixCacheType _type;

	protected MatrixIndexingSPInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, SparkAggType aggtype, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, aggtype, opcode, istr);
		_type = LixCacheType.NONE;
	}

	protected MatrixIndexingSPInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl,
			CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, LixCacheType type, String opcode, String istr) {
		super(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, istr);
		_type = type;
	}

	public LixCacheType getLixType() {
		return _type;
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
		
		//check bounds
		DataCharacteristics mcIn = sec.getDataCharacteristics(input1.getName());
		if( mcIn.dimsKnown() && (ru>mcIn.getRows() || cu>mcIn.getCols()) )
			throw new DMLRuntimeException("Index range out of bounds: "+ixrange+" "+mcIn);
		
		//right indexing
		if( opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) )
		{
			//update and check output dimensions
			DataCharacteristics mcOut = output.isScalar() ? 
				new MatrixCharacteristics(1,1) :
				ec.getDataCharacteristics(output.getName());
			mcOut.set(ru-rl+1, cu-cl+1, mcIn.getBlocksize(), mcIn.getBlocksize());
			mcOut.setNonZerosBound(Math.min(mcOut.getLength(), mcIn.getNonZerosBound()));
			checkValidOutputDimensions(mcOut);
			
			//execute right indexing operation (partitioning-preserving if possible)
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		
			if( output.isScalar() ) { //SCALAR output
				MatrixBlock ret = singleBlockIndexing(in1, mcIn, mcOut, ixrange);
				sec.setScalarOutput(output.getName(), new DoubleObject(ret.get(0, 0)));
			}
			else { //MATRIX output
				
				if( isSingleBlockLookup(mcIn, ixrange) ) {
					sec.setMatrixOutput(output.getName(), singleBlockIndexing(in1, mcIn, mcOut, ixrange));
				}
				else if( isMultiBlockLookup(in1, mcIn, mcOut, ixrange) ) {
					sec.setMatrixOutput(output.getName(), multiBlockIndexing(in1, mcIn, mcOut, ixrange));
				}
				else { //rdd output for general case
					JavaPairRDD<MatrixIndexes,MatrixBlock> out = generalCaseRightIndexing(in1, mcIn, mcOut, ixrange, _aggType);
					
					//put output RDD handle into symbol table
					sec.setRDDHandleForVariable(output.getName(), out);
					sec.addLineageRDD(output.getName(), input1.getName());
				}
			}
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPLEFTINDEX.toString()))
		{
			String rddVar = (_type==LixCacheType.LEFT) ? input2.getName() : input1.getName();
			String bcVar = (_type==LixCacheType.LEFT) ? input1.getName() : input2.getName();
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable( rddVar );
			PartitionedBroadcast<MatrixBlock> broadcastIn2 = null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			
			//update and check output dimensions
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
			DataCharacteristics mcLeft = mcIn;
			mcOut.set(mcLeft.getRows(), mcLeft.getCols(), mcLeft.getBlocksize(), mcLeft.getBlocksize());
			checkValidOutputDimensions(mcOut);
			
			//note: always matrix rhs, scalars are preprocessed via cast to 1x1 matrix
			DataCharacteristics mcRight = ec.getDataCharacteristics(input2.getName());
				
			//sanity check matching index range and rhs dimensions
			if(!mcRight.dimsKnown()) {
				throw new DMLRuntimeException("The right input matrix dimensions are not specified for MatrixIndexingSPInstruction");
			}
			if(!(ru-rl+1 == mcRight.getRows() && cu-cl+1 == mcRight.getCols())) {
				throw new DMLRuntimeException("Invalid index range of leftindexing: ["+rl+":"+ru+","+cl+":"+cu+"] vs ["+mcRight.getRows()+"x"+mcRight.getCols()+"]." );
			}
			
			if(opcode.equalsIgnoreCase(Opcodes.MAPLEFTINDEX.toString()))
			{
				broadcastIn2 = sec.getBroadcastForVariable( bcVar );
				
				//partitioning-preserving mappartitions (key access required for broadcast loopkup)
				out = in1.mapPartitionsToPair(
						new LeftIndexPartitionFunction(broadcastIn2, ixrange, _type, mcOut), true);
			}
			else { //general case
				// zero-out lhs
				in1 = in1.mapToPair(new ZeroOutLHS(ixrange, mcLeft));
				
				// slice rhs, shift and merge with lhs
				in2 = sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() )
					    .flatMapToPair(new SliceRHSForLeftIndexing(ixrange, mcLeft));
				out = RDDAggregateUtils.mergeByKey(in1.union(in2));
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), rddVar);
			if( broadcastIn2 != null)
				sec.addLineageBroadcast(output.getName(), bcVar);
			if(in2 != null) 
				sec.addLineageRDD(output.getName(), input2.getName());
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in MatrixIndexingSPInstruction.");
	}


	public static MatrixBlock inmemoryIndexing(JavaPairRDD<MatrixIndexes,MatrixBlock> in1,
		DataCharacteristics mcIn, DataCharacteristics mcOut, IndexRange ixrange)
	{
		if( isSingleBlockLookup(mcIn, ixrange) ) {
			return singleBlockIndexing(in1, mcIn, mcOut, ixrange);
		}
		else if( isMultiBlockLookup(in1, mcIn, mcOut, ixrange) ) {
			return multiBlockIndexing(in1, mcIn, mcOut, ixrange);
		}
		else
			throw new DMLRuntimeException("Incorrect usage of inmemoryIndexing");
	}
	
	private static MatrixBlock multiBlockIndexing(JavaPairRDD<MatrixIndexes,MatrixBlock> in1,
	                                              DataCharacteristics mcIn, DataCharacteristics mcOut, IndexRange ixrange) {
		//create list of all required matrix indexes
		List<MatrixIndexes> filter = new ArrayList<>();
		long rlix = UtilFunctions.computeBlockIndex(ixrange.rowStart, mcIn.getBlocksize());
		long ruix = UtilFunctions.computeBlockIndex(ixrange.rowEnd, mcIn.getBlocksize());
		long clix = UtilFunctions.computeBlockIndex(ixrange.colStart, mcIn.getBlocksize());
		long cuix = UtilFunctions.computeBlockIndex(ixrange.colEnd, mcIn.getBlocksize());
		for( long r=rlix; r<=ruix; r++ )
			for( long c=clix; c<=cuix; c++ )
				filter.add( new MatrixIndexes(r,c) );
		
		//wrap PartitionPruningRDD around input to exploit pruning for out-of-core datasets
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = createPartitionPruningRDD(in1, filter);
		out = out.filter(new IsBlockInRange(ixrange.rowStart, ixrange.rowEnd, ixrange.colStart, ixrange.colEnd, mcOut)) //filter unnecessary blocks 
				 .mapToPair(new SliceBlock2(ixrange, mcOut));       //slice relevant blocks
		
		//collect output without shuffle to avoid side-effects with custom PartitionPruningRDD
		MatrixBlock mbout = SparkExecutionContext.toMatrixBlock(out, (int)mcOut.getRows(), 
				(int)mcOut.getCols(), mcOut.getBlocksize(), -1);
		return mbout;
	}
	
	private static MatrixBlock singleBlockIndexing(JavaPairRDD<MatrixIndexes,MatrixBlock> in1,
	                                               DataCharacteristics mcIn, DataCharacteristics mcOut, IndexRange ixrange) {
		//single block output via lookup (on partitioned inputs, this allows for single partition
		//access to avoid a full scan of the input; note that this is especially important for 
		//out-of-core datasets as entire partitions are read, not just keys as in the in-memory setting.
		long rix = UtilFunctions.computeBlockIndex(ixrange.rowStart, mcIn.getBlocksize());
		long cix = UtilFunctions.computeBlockIndex(ixrange.colStart, mcIn.getBlocksize());
		List<MatrixBlock> list = in1.lookup(new MatrixIndexes(rix, cix));
		if( list.size() != 1 )
			throw new DMLRuntimeException("Block lookup returned "+list.size()+" blocks (expected 1).");
		
		MatrixBlock tmp = list.get(0);
		MatrixBlock mbout = (tmp.getNumRows()==mcOut.getRows() && tmp.getNumColumns()==mcOut.getCols()) ? 
				tmp : tmp.slice( //reference full block or slice out sub-block
				UtilFunctions.computeCellInBlock(ixrange.rowStart, mcIn.getBlocksize()), 
				UtilFunctions.computeCellInBlock(ixrange.rowEnd, mcIn.getBlocksize()), 
				UtilFunctions.computeCellInBlock(ixrange.colStart, mcIn.getBlocksize()), 
				UtilFunctions.computeCellInBlock(ixrange.colEnd, mcIn.getBlocksize()), new MatrixBlock());
		mbout.examSparsity();
		return mbout;
	}
	
	private static JavaPairRDD<MatrixIndexes,MatrixBlock> generalCaseRightIndexing(JavaPairRDD<MatrixIndexes,MatrixBlock> in1,
		DataCharacteristics mcIn, DataCharacteristics mcOut, IndexRange ixrange, SparkAggType aggType)
	{
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		
		if( isPartitioningPreservingRightIndexing(mcIn, ixrange) ) {
			out = in1.mapPartitionsToPair(
					new SliceBlockPartitionFunction(ixrange, mcOut), true);
		}
		else if( aggType == SparkAggType.NONE
			|| OptimizerUtils.isIndexingRangeBlockAligned(ixrange, mcIn) ) {
			out = in1.filter(new IsBlockInRange(ixrange.rowStart, ixrange.rowEnd, ixrange.colStart, ixrange.colEnd, mcOut))
		             .mapToPair(new SliceSingleBlock(ixrange, mcOut));
			int prefNoutPart = SparkUtils.getNumPreferredPartitions(mcOut);
			//determine the need for coalesce
			boolean coalesce = 1.4*prefNoutPart < in1.getNumPartitions() && !SparkUtils.isHashPartitioned(in1);
			if (coalesce) //merge partitions without shuffle
				out = out.coalesce(prefNoutPart);
		}
		else {
			out = in1.filter(new IsBlockInRange(ixrange.rowStart, ixrange.rowEnd, ixrange.colStart, ixrange.colEnd, mcOut))
		             .flatMapToPair(new SliceMultipleBlocks(ixrange, mcOut));
			out = RDDAggregateUtils.mergeByKey(out);
		}
		return out;
	}
	
	private static void checkValidOutputDimensions(DataCharacteristics mcOut) {
		if(!mcOut.dimsKnown()) {
			throw new DMLRuntimeException("MatrixIndexingSPInstruction: The updated output dimensions are invalid: " + mcOut);
		}
	}

	private static boolean isPartitioningPreservingRightIndexing(DataCharacteristics mcIn, IndexRange ixrange) {
		return ( mcIn.dimsKnown() &&
				(ixrange.rowStart==1 && ixrange.rowEnd==mcIn.getRows() && mcIn.getCols()<=mcIn.getBlocksize() )   //1-1 column block indexing
			  ||(ixrange.colStart==1 && ixrange.colEnd==mcIn.getCols() && mcIn.getRows()<=mcIn.getBlocksize() )); //1-1 row block indexing
	}
	
	/**
	 * Indicates if the given index range only covers a single blocks of the inputs matrix.
	 * In this case, we perform a key lookup which is very efficient in case of existing
	 * partitioner, especially for out-of-core datasets.
	 * 
	 * @param mcIn matrix characteristics
	 * @param ixrange index range
	 * @return true if index range covers a single block of the input matrix
	 */
	public static boolean isSingleBlockLookup(DataCharacteristics mcIn, IndexRange ixrange) {
		return UtilFunctions.computeBlockIndex(ixrange.rowStart, mcIn.getBlocksize())
			== UtilFunctions.computeBlockIndex(ixrange.rowEnd, mcIn.getBlocksize())
			&& UtilFunctions.computeBlockIndex(ixrange.colStart, mcIn.getBlocksize())
			== UtilFunctions.computeBlockIndex(ixrange.colEnd, mcIn.getBlocksize());
	}
	
	/**
	 * Indicates if the given index range and input matrix exhibit the following properties:
	 * (1) existing hash partitioner, (2) out-of-core input matrix (larger than aggregated memory), 
	 * (3) aligned indexing range (which does not required aggregation), and (4) the output fits 
	 * twice in memory (in order to collect the result). 
	 * 
	 * @param in input matrix
	 * @param mcIn input matrix characteristics
	 * @param mcOut output matrix characteristics
	 * @param ixrange index range
	 * @return true if index range requires a multi-block lookup
	 */
	public static boolean isMultiBlockLookup(JavaPairRDD<?,?> in, DataCharacteristics mcIn, DataCharacteristics mcOut, IndexRange ixrange) {
		return SparkUtils.isHashPartitioned(in)                          //existing partitioner
			&& OptimizerUtils.estimatePartitionedSizeExactSparsity(mcIn) //out-of-core dataset
			   > SparkExecutionContext.getDataMemoryBudget(true, true)
			&& OptimizerUtils.isIndexingRangeBlockAligned(ixrange, mcIn) //no block aggregation
			&& OptimizerUtils.estimateSize(mcOut) < OptimizerUtils.getLocalMemBudget()/2; //outputs fits in memory
	}

	private static class SliceRHSForLeftIndexing implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
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
			_blen = mcLeft.getBlocksize();
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> rightKV) 
			throws Exception 
		{
			IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(rightKV);
			ArrayList<IndexedMatrixValue> out = new ArrayList<>();
			OperationsOnMatrixValues.performShift(in, _ixrange, _blen, _rlen, _clen, out);
			return SparkUtils.fromIndexedMatrixBlock(out).iterator();
		}		
	}

	private static class ZeroOutLHS implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -3581795160948484261L;
		
		private IndexRange _ixrange = null;
		private int _blen = -1;
		
		public ZeroOutLHS(IndexRange range, DataCharacteristics mcLeft) {
			_ixrange = range;
			_blen = mcLeft.getBlocksize();
			_blen = mcLeft.getBlocksize();
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{
			if( !UtilFunctions.isInBlockRange(kv._1(), _blen, _ixrange) ) {
				return kv;
			}
			
			IndexRange range = UtilFunctions.getSelectedRangeForZeroOut(new IndexedMatrixValue(kv._1, kv._2), _blen, _ixrange);
			if(range.rowStart == -1 && range.rowEnd == -1 && range.colStart == -1 && range.colEnd == -1) {
				throw new Exception("Error while getting range for zero-out");
			}
			
			MatrixBlock zeroBlk = kv._2.zeroOutOperations(new MatrixBlock(), range);
			return new Tuple2<>(kv._1, zeroBlk);
		}
	}

	private static class LeftIndexPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 1757075506076838258L;
		
		private final PartitionedBroadcast<MatrixBlock> _binput;
		private final IndexRange _ixrange;
		private final LixCacheType _type;
		private final int _blen;
		
		public LeftIndexPartitionFunction(PartitionedBroadcast<MatrixBlock> binput, IndexRange ixrange, LixCacheType type, DataCharacteristics mc)
		{
			_binput = binput;
			_ixrange = ixrange;
			_type = type;
			_blen = mc.getBlocksize();
		}

		@Override
		public LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			return new LeftIndexPartitionIterator(arg0);
		}

		private class LeftIndexPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public LeftIndexPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}
			
			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg) 
				throws Exception 
			{
				if(_type==LixCacheType.RIGHT && !UtilFunctions.isInBlockRange(arg._1(), _blen, _ixrange)) {
					return arg;
				}
				
				if( _type == LixCacheType.LEFT ) 
				{
					// LixCacheType.LEFT guarantees aligned blocks, so for each rhs inputs block
					// the the corresponding left block and perform blockwise left indexing
					MatrixIndexes ix = arg._1();
					MatrixBlock right = arg._2();

					int rl = UtilFunctions.computeCellInBlock(_ixrange.rowStart, _blen);
					int ru = (int)Math.min(_ixrange.rowEnd, rl+right.getNumRows())-1;
					int cl = UtilFunctions.computeCellInBlock(_ixrange.colStart, _blen);
					int cu = (int)Math.min(_ixrange.colEnd, cl+right.getNumColumns())-1;
					
					MatrixBlock left = _binput.getBlock((int)ix.getRowIndex(), (int)ix.getColumnIndex());
					MatrixBlock tmp = left.leftIndexingOperations(right, 
							rl, ru, cl, cu, new MatrixBlock(), UpdateType.COPY);
					
					return new Tuple2<>(ix, tmp);
				}
				else //LixCacheType.RIGHT
				{
					// Calculate global index of left hand side block
					long lhs_rl = Math.max(_ixrange.rowStart, (arg._1.getRowIndex()-1)*_blen + 1);
					long lhs_ru = Math.min(_ixrange.rowEnd, arg._1.getRowIndex()*_blen);
					long lhs_cl = Math.max(_ixrange.colStart, (arg._1.getColumnIndex()-1)*_blen + 1);
					long lhs_cu = Math.min(_ixrange.colEnd, arg._1.getColumnIndex()*_blen);
					
					// Calculate global index of right hand side block
					long rhs_rl = lhs_rl - _ixrange.rowStart + 1;
					long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
					long rhs_cl = lhs_cl - _ixrange.colStart + 1;
					long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
					
					// Provide global zero-based index to sliceOperations
					MatrixBlock slicedRHSMatBlock = _binput.slice(rhs_rl, rhs_ru, rhs_cl, rhs_cu, new MatrixBlock());
					
					// Provide local zero-based index to leftIndexingOperations
					int lhs_lrl = UtilFunctions.computeCellInBlock(lhs_rl, _blen);
					int lhs_lru = UtilFunctions.computeCellInBlock(lhs_ru, _blen);
					int lhs_lcl = UtilFunctions.computeCellInBlock(lhs_cl, _blen);
					int lhs_lcu = UtilFunctions.computeCellInBlock(lhs_cu, _blen);
					MatrixBlock ret = arg._2.leftIndexingOperations(slicedRHSMatBlock, lhs_lrl, lhs_lru, lhs_lcl, lhs_lcu, new MatrixBlock(), UpdateType.COPY);
					return new Tuple2<>(arg._1, ret);
				}
			}
		}
	}

	private static class SliceSingleBlock implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -6724027136506200924L;
		
		private final IndexRange _ixrange;
		private final int _blen; 
		
		public SliceSingleBlock(IndexRange ixrange, DataCharacteristics mcOut) {
			_ixrange = ixrange;
			_blen = mcOut.getBlocksize();
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{	
			//get inputs (guaranteed to fall into indexing range)
			MatrixIndexes ix = kv._1();
			MatrixBlock block = kv._2();
			
			//compute local index range 
			long grix = UtilFunctions.computeCellIndex(ix.getRowIndex(), _blen, 0);
			long gcix = UtilFunctions.computeCellIndex(ix.getColumnIndex(), _blen, 0);
			int lrl = (int)((_ixrange.rowStart<grix) ? 0 : _ixrange.rowStart - grix);
			int lcl = (int)((_ixrange.colStart<gcix) ? 0 : _ixrange.colStart - gcix);
			int lru = (int)Math.min(block.getNumRows()-1, _ixrange.rowEnd - grix);
			int lcu = (int)Math.min(block.getNumColumns()-1, _ixrange.colEnd - gcix);
			
			//compute output index
			MatrixIndexes ixOut = new MatrixIndexes(
				ix.getRowIndex() - (_ixrange.rowStart-1)/_blen, 
				ix.getColumnIndex() - (_ixrange.colStart-1)/_blen);
			
			//create return matrix block (via shallow copy or slice)
			if( lrl == 0 && lru == block.getNumRows()-1
				&& lcl == 0 && lcu == block.getNumColumns()-1 ) {
				return new Tuple2<>(ixOut, block);
			}
			else {
				return new Tuple2<>(ixOut, 
					block.slice(lrl, lru, lcl, lcu, new MatrixBlock()));
			}
		}		
	}
	
	private static class SliceMultipleBlocks implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 5733886476413136826L;
		
		private final IndexRange _ixrange;
		private final int _blen; 
		
		public SliceMultipleBlocks(IndexRange ixrange, DataCharacteristics mcOut) {
			_ixrange = ixrange;
			_blen = mcOut.getBlocksize();
		}

		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{	
			IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(kv);
			ArrayList<IndexedMatrixValue> outlist = OperationsOnMatrixValues.performSlice(in, _ixrange, _blen);
			return SparkUtils.fromIndexedMatrixBlock(outlist).iterator();
		}		
	}

	/**
	 * Equivalent to SliceBlock except a different function signature.
	 */
	private static class SliceBlock2 implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 7481889252529447770L;
		
		private IndexRange _ixrange;
		private int _blen; 
		
		public SliceBlock2(IndexRange ixrange, DataCharacteristics mcOut) {
			_ixrange = ixrange;
			_blen = mcOut.getBlocksize();
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) 
			throws Exception 
		{	
			IndexedMatrixValue in = new IndexedMatrixValue(kv._1(), kv._2());
			ArrayList<IndexedMatrixValue> outlist = OperationsOnMatrixValues.performSlice(in, _ixrange, _blen);
			return SparkUtils.fromIndexedMatrixBlock(outlist.get(0));
		}		
	}

	private static class SliceBlockPartitionFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixBlock>>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = -8111291718258309968L;
		
		private IndexRange _ixrange;
		private int _blen; 
		
		public SliceBlockPartitionFunction(IndexRange ixrange, DataCharacteristics mcOut) {
			_ixrange = ixrange;
			_blen = mcOut.getBlocksize();
		}

		@Override
		public LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			return new SliceBlockPartitionIterator(arg0);
		}	
		
		private class SliceBlockPartitionIterator extends LazyIterableIterator<Tuple2<MatrixIndexes, MatrixBlock>>
		{
			public SliceBlockPartitionIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				super(in);
			}

			@Override
			protected Tuple2<MatrixIndexes, MatrixBlock> computeNext(Tuple2<MatrixIndexes, MatrixBlock> arg)
				throws Exception
			{
				IndexedMatrixValue in = SparkUtils.toIndexedMatrixBlock(arg);
				
				ArrayList<IndexedMatrixValue> outlist = OperationsOnMatrixValues.performSlice(in, _ixrange, _blen);
				
				assert(outlist.size() == 1); //1-1 row/column block indexing
				return SparkUtils.fromIndexedMatrixBlock(outlist.get(0));
			}
		}
	}
	
	/**
	 * Wraps the input RDD into a PartitionPruningRDD, which acts as a filter
	 * of required partitions. The distinct set of required partitions is determined
	 * via the partitioner of the input RDD.
	 * 
	 * @param in input matrix as {@code JavaPairRDD<MatrixIndexes,MatrixBlock>}
	 * @param filter partition filter
	 * @return matrix as {@code JavaPairRDD<MatrixIndexes,MatrixBlock>}
	 */
	private static JavaPairRDD<MatrixIndexes,MatrixBlock> createPartitionPruningRDD( 
			JavaPairRDD<MatrixIndexes,MatrixBlock> in, List<MatrixIndexes> filter )
	{
		//build hashset of required partition ids
		HashSet<Integer> flags = new HashSet<>();
		Partitioner partitioner = in.rdd().partitioner().get();
		for( MatrixIndexes key : filter )
			flags.add(partitioner.getPartition(key));

		//create partition pruning rdd
		Function1<Object,Object> f = new PartitionPruningFunction(flags);
		PartitionPruningRDD<Tuple2<MatrixIndexes, MatrixBlock>> ppRDD = 
				PartitionPruningRDD.create(in.rdd(), f);

		//wrap output into java pair rdd
		return new JavaPairRDD<>(ppRDD, 
				ClassManifestFactory.fromClass(MatrixIndexes.class), 
				ClassManifestFactory.fromClass(MatrixBlock.class));
	}
	
	/**
	 * Filter function required to create a PartitionPruningRDD.
	 */
	private static class PartitionPruningFunction extends AbstractFunction1<Object,Object> implements Serializable
	{
		private static final long serialVersionUID = -9114299718258329951L;

		private HashSet<Integer> _filterFlags = null;

		public PartitionPruningFunction(HashSet<Integer> flags) {
			_filterFlags = flags;
		}

		@Override
		public Boolean apply(Object partIndex) {
			return _filterFlags.contains(partIndex);
		}
	}
	
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		return Pair.of(output.getName(), new LineageItem(getOpcode(),
			LineageItemUtils.getLineage(ec, input1,input2,input3,rowLower,rowUpper,colLower,colUpper)));
	}
}

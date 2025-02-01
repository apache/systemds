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

package org.apache.sysds.runtime.instructions.spark.utils;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.data.TensorIndexes;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.instructions.spark.AggregateUnarySPInstruction.RDDUAggFunction2;
import org.apache.sysds.runtime.instructions.spark.data.CorrMatrixBlock;
import org.apache.sysds.runtime.instructions.spark.data.RowMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;

/**
 * Collection of utility methods for aggregating binary block rdds. As a general
 * policy always call stable algorithms which maintain corrections over blocks
 * per key. The performance overhead over a simple reducebykey is roughly 7-10% 
 * and with that acceptable. 
 * 
 */
public class RDDAggregateUtils
{	
	//internal configuration to use tree aggregation (treeReduce w/ depth=2),
	//this is currently disabled because it was 2x slower than a simple
	//single-block reduce due to additional overhead for shuffling 
	private static final boolean TREE_AGGREGATION = false; 

	public static MatrixBlock sumStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in ) {
		return sumStable( in.values() );
	}

	public static MatrixBlock sumStable( JavaRDD<MatrixBlock> in )
	{
		//stable sum of all blocks with correction block per function instance
		if( TREE_AGGREGATION ) {
			return in.treeReduce( 
					new SumSingleBlockFunction(true) );	
		}
		else { //DEFAULT
			//reduce-all aggregate via fold instead of reduce to allow 
			//for update in-place w/o deep copy of left-hand-side blocks
			return in.fold(
					new MatrixBlock(), 
					new SumSingleBlockFunction(false));
		}
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sumByKeyStable(JavaPairRDD<MatrixIndexes, MatrixBlock> in) {
		return sumByKeyStable(in, in.getNumPartitions(), true);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sumByKeyStable(JavaPairRDD<MatrixIndexes, MatrixBlock> in,
			boolean deepCopyCombiner) {
		return sumByKeyStable(in, in.getNumPartitions(), deepCopyCombiner);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sumByKeyStable(JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
			int numPartitions, boolean deepCopyCombiner)
	{
		//stable sum of blocks per key, by passing correction blocks along with aggregates
		JavaPairRDD<MatrixIndexes, CorrMatrixBlock> tmp = 
			in.combineByKey( new CreateCorrBlockCombinerFunction(deepCopyCombiner), 
				new MergeSumBlockValueFunction(deepCopyCombiner),
				new MergeSumBlockCombinerFunction(deepCopyCombiner), numPartitions );
		
		//strip-off correction blocks from
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =
			tmp.mapValues( new ExtractMatrixBlock() );
		
		//return the aggregate rdd
		return out;
	}

	
	public static JavaPairRDD<MatrixIndexes, Double> sumCellsByKeyStable( JavaPairRDD<MatrixIndexes, Double> in ) {
		return sumCellsByKeyStable(in, in.getNumPartitions());
	}
	
	public static JavaPairRDD<MatrixIndexes, Double> sumCellsByKeyStable( JavaPairRDD<MatrixIndexes, Double> in, int numParts )
	{
		//stable sum of blocks per key, by passing correction blocks along with aggregates
		JavaPairRDD<MatrixIndexes, KahanObject> tmp =
				in.combineByKey( new CreateCellCombinerFunction(),
					new MergeSumCellValueFunction(), 
					new MergeSumCellCombinerFunction(), numParts);
		
		//strip-off correction blocks from
		JavaPairRDD<MatrixIndexes, Double> out =
				tmp.mapValues( new ExtractDoubleCell() );
		
		//return the aggregate rdd
		return out;
	}
	
	/**
	 * Single block aggregation over pair rdds with corrections for numerical stability.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @param aop aggregate operator
	 * @return matrix block
	 */
	public static MatrixBlock aggStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in, AggregateOperator aop ) {
		return aggStable( in.values(), aop );
	}
	
	/**
	 * Single block aggregation over rdds with corrections for numerical stability.
	 * 
	 * @param in matrix as {@code JavaRDD<MatrixBlock>}
	 * @param aop aggregate operator
	 * @return matrix block
	 */
	public static MatrixBlock aggStable( JavaRDD<MatrixBlock> in, AggregateOperator aop )
	{
		//stable aggregate of all blocks with correction block per function instance
		
		//reduce-all aggregate via fold instead of reduce to allow 
		//for update in-place w/o deep copy of left-hand-side blocks
		return in.fold(
				new MatrixBlock(),
				new AggregateSingleBlockFunction(aop) );
	}

	/**
	 * Single block aggregation over pair rdds with corrections for numerical stability.
	 *
	 * @param in tensor as {@code JavaPairRDD<TensorIndexes, TensorBlock>}
	 * @param aop aggregate operator
	 * @return tensor block
	 */
	public static TensorBlock aggStableTensor(JavaPairRDD<TensorIndexes, TensorBlock> in, AggregateOperator aop) {
		return aggStableTensor(in.values(), aop);
	}

	/**
	 * Single block aggregation over rdds with corrections for numerical stability.
	 *
	 * @param in tensor as {@code JavaRDD<TensorBlock>}
	 * @param aop aggregate operator
	 * @return tensor block
	 */
	public static TensorBlock aggStableTensor(JavaRDD<TensorBlock> in, AggregateOperator aop )
	{
		//stable aggregate of all blocks with correction block per function instance

		//reduce-all aggregate via fold instead of reduce to allow
		//for update in-place w/o deep copy of left-hand-side blocks
		return in.fold(
				new TensorBlock(),
				new AggregateSingleTensorBlockFunction(aop) );
	}
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> aggByKeyStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in,
			AggregateOperator aop) {
		return aggByKeyStable(in, aop, in.getNumPartitions(), true);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> aggByKeyStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
			AggregateOperator aop, boolean deepCopyCombiner ) {
		return aggByKeyStable(in, aop, in.getNumPartitions(), deepCopyCombiner);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> aggByKeyStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
			AggregateOperator aop, int numPartitions, boolean deepCopyCombiner )
	{
		//stable sum of blocks per key, by passing correction blocks along with aggregates
		JavaPairRDD<MatrixIndexes, CorrMatrixBlock> tmp = 
				in.combineByKey( new CreateCorrBlockCombinerFunction(deepCopyCombiner),
							     new MergeAggBlockValueFunction(aop), 
							     new MergeAggBlockCombinerFunction(aop), numPartitions );
		
		//strip-off correction blocks from
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =  
				tmp.mapValues( new ExtractMatrixBlock() );
		
		//return the aggregate rdd
		return out;
	}
	
	public static double max(JavaPairRDD<MatrixIndexes, MatrixBlock> in) {
		AggregateUnaryOperator auop = InstructionUtils.parseBasicAggregateUnaryOperator(Opcodes.UAMAX.toString());
		MatrixBlock tmp = aggStable(in.map(new RDDUAggFunction2(auop, -1)), auop.aggOp);
		return tmp.get(0, 0);
	}
	
	/**
	 * Merges disjoint data of all blocks per key.
	 * 
	 * Note: The behavior of this method is undefined for both sparse and dense data if the 
	 * assumption of disjoint data is violated.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @return matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> mergeByKey( JavaPairRDD<MatrixIndexes, MatrixBlock> in ) {
		return mergeByKey(in, in.getNumPartitions(), true);
	}
	
	/**
	 * Merges disjoint data of all blocks per key.
	 * 
	 * Note: The behavior of this method is undefined for both sparse and dense data if the 
	 * assumption of disjoint data is violated.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @param deepCopyCombiner indicator if the createCombiner functions needs to deep copy the input block
	 * @return matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> mergeByKey( JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
		boolean deepCopyCombiner ) {
		return mergeByKey(in, in.getNumPartitions(), deepCopyCombiner); 
	}
	
	/**
	 * Merges disjoint data of all blocks per key.
	 * 
	 * Note: The behavior of this method is undefined for both sparse and dense data if the 
	 * assumption of disjoint data is violated.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @param numPartitions number of output partitions
	 * @param deepCopyCombiner indicator if the createCombiner functions needs to deep copy the input block
	 * @return matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> mergeByKey( JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
			int numPartitions, boolean deepCopyCombiner )
	{
		//use combine by key to avoid unnecessary deep block copies, i.e.
		//create combiner block once and merge remaining blocks in-place.
 		return in.combineByKey( 
 				new CreateBlockCombinerFunction(deepCopyCombiner), 
			    new MergeBlocksFunction(false), 
			    new MergeBlocksFunction(false), numPartitions );
	}
	
	/**
	 * Merges disjoint data of all blocks per key.
	 * 
	 * Note: The behavior of this method is undefined for both sparse and dense data if the 
	 * assumption of disjoint data is violated.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes, RowMatrixBlock>}
	 * @return matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> mergeRowsByKey( JavaPairRDD<MatrixIndexes, RowMatrixBlock> in )
	{
		return in.combineByKey( new CreateRowBlockCombinerFunction(), 
							    new MergeRowBlockValueFunction(), 
							    new MergeBlocksFunction(false) );
	}

	private static class CreateCorrBlockCombinerFunction implements Function<MatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = -3666451526776017343L;

		private final boolean _deep;

		public CreateCorrBlockCombinerFunction(boolean deep) {
			_deep = deep;
		}
		
		@Override
		public CorrMatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			//deep copy to allow update in-place
			return new CorrMatrixBlock(
				_deep ? new MatrixBlock(arg0) : arg0);
		}	
	}

	private static class MergeSumBlockValueFunction implements Function2<CorrMatrixBlock, MatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 3703543699467085539L;
		
		private AggregateOperator _op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.NONE);
		
		private final boolean _deep;

		public MergeSumBlockValueFunction(boolean deep) {
			_deep = deep;
		}
		
		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, MatrixBlock arg1) 
			throws Exception 
		{
			if( arg1.isEmptyBlock(false) )
				return arg0;
			
			//get current block and correction
			MatrixBlock value = arg0.getValue();
			MatrixBlock corr = arg0.getCorrection();
			
			//correction block allocation on demand
			if( corr == null && !arg1.isEmptyBlock(false) )
				corr = new MatrixBlock(value.getNumRows(), value.getNumColumns(), false);
			
			//aggregate other input and maintain corrections 
			//(existing value and corr are used in place)
			OperationsOnMatrixValues.incrementalAggregation(value, corr, arg1, _op, false, _deep);
			return arg0.set(value, corr);
		}
	}

	private static class MergeSumBlockCombinerFunction implements Function2<CorrMatrixBlock, CorrMatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 7664941774566119853L;
		
		private AggregateOperator _op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.NONE);
		private final boolean _deep;

		public MergeSumBlockCombinerFunction(boolean deep) {
			_deep = deep;
		}
		
		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, CorrMatrixBlock arg1) 
			throws Exception 
		{
			//get current block and correction
			MatrixBlock value1 = arg0.getValue();
			MatrixBlock value2 = arg1.getValue();
			MatrixBlock corr = arg0.getCorrection();
			
			//correction block allocation on demand (but use second if exists)
			if( corr == null ) {
				corr = (arg1.getCorrection()!=null) ? arg1.getCorrection() :
					value2.isEmptyBlock(false) || (!_deep && value1.isEmptyBlock(false)) ? null :
					new MatrixBlock(value1.getNumRows(), value1.getNumColumns(), false);
			}
			
			//aggregate other input and maintain corrections
			//(existing value and corr are used in place)
			OperationsOnMatrixValues.incrementalAggregation(value1, corr, value2, _op, false, _deep);
			return arg0.set(value1, corr);
		}
	}

	private static class CreateBlockCombinerFunction implements Function<MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 1987501624176848292L;
		
		private final boolean _deep;
		
		public CreateBlockCombinerFunction(boolean deep) {
			_deep = deep;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			//create deep copy of given block
			return _deep ? new MatrixBlock(arg0) : arg0;
		}	
	}

	private static class CreateRowBlockCombinerFunction implements Function<RowMatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 2866598914232118425L;

		@Override
		public MatrixBlock call(RowMatrixBlock arg0) 
			throws Exception 
		{
			//create new target block and copy row into it
			MatrixBlock row = arg0.getValue();
			MatrixBlock out = new MatrixBlock(arg0.getLen(), row.getNumColumns(), true);
			out.copy(arg0.getRow(), arg0.getRow(), 0, row.getNumColumns()-1, row, false);
			out.setNonZeros(row.getNonZeros());
			out.examSparsity();
			
			return out;
		}	
	}

	private static class MergeRowBlockValueFunction implements Function2<MatrixBlock, RowMatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = -803689998683298516L;

		@Override
		public MatrixBlock call(MatrixBlock arg0, RowMatrixBlock arg1) 
			throws Exception 
		{
			//copy row into existing target block
			MatrixBlock row = arg1.getValue();
			MatrixBlock out = arg0; //in-place update
			out.copy(arg1.getRow(), arg1.getRow(), 0, row.getNumColumns()-1, row, true);
			out.examSparsity();
			
			return out;
		}	
	}

	private static class CreateCellCombinerFunction implements Function<Double, KahanObject> 
	{
		private static final long serialVersionUID = 3697505233057172994L;

		@Override
		public KahanObject call(Double arg0) 
			throws Exception 
		{
			return new KahanObject(arg0, 0.0);
		}	
	}

	private static class MergeSumCellValueFunction implements Function2<KahanObject, Double, KahanObject> 
	{
		private static final long serialVersionUID = 468335171573184825L;

		@Override
		public KahanObject call(KahanObject arg0, Double arg1) 
			throws Exception 
		{
			//get reused kahan plus object
			KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
			
			//compute kahan plus and keep correction
			kplus.execute2(arg0, arg1);	
			
			return arg0;
		}	
	}

	private static class MergeSumCellCombinerFunction implements Function2<KahanObject, KahanObject, KahanObject> 
	{
		private static final long serialVersionUID = 8726716909849119657L;

		@Override
		public KahanObject call(KahanObject arg0, KahanObject arg1) 
			throws Exception 
		{
			//get reused kahan plus object
			KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
			
			//compute kahan plus and keep correction
			kplus.execute2(arg0, arg1._sum);
			
			return arg0;
		}	
	}

	private static class MergeAggBlockValueFunction implements Function2<CorrMatrixBlock, MatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 389422125491172011L;
		
		private AggregateOperator _op = null;	
		
		public MergeAggBlockValueFunction(AggregateOperator aop)
		{
			_op = aop;
		}
		
		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, MatrixBlock arg1) 
			throws Exception 
		{
			//get current block and correction
			MatrixBlock value = arg0.getValue();
			MatrixBlock corr = arg0.getCorrection();
			
			//correction block allocation on demand
			if( corr == null && _op.existsCorrection() ){
				corr = new MatrixBlock(value.getNumRows(), value.getNumColumns(), false);
			}
			
			//aggregate other input and maintain corrections 
			//(existing value and corr are used in place)
			if(_op.existsCorrection())
				OperationsOnMatrixValues.incrementalAggregation(value, corr, arg1, _op, true);
			else
				OperationsOnMatrixValues.incrementalAggregation(value, null, arg1, _op, true);
			return new CorrMatrixBlock(value, corr);
		}	
	}

	private static class MergeAggBlockCombinerFunction implements Function2<CorrMatrixBlock, CorrMatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 4803711632648880797L;
		
		private AggregateOperator _op = null;
		
		public MergeAggBlockCombinerFunction(AggregateOperator aop)
		{
			_op = aop;
		}
		
		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, CorrMatrixBlock arg1) 
			throws Exception 
		{
			//get current block and correction
			MatrixBlock value1 = arg0.getValue();
			MatrixBlock value2 = arg1.getValue();
			MatrixBlock corr = arg0.getCorrection();
			
			//correction block allocation on demand (but use second if exists)
			if( corr == null && _op.existsCorrection()) {
				corr = (arg1.getCorrection()!=null)?arg1.getCorrection():
					new MatrixBlock(value1.getNumRows(), value1.getNumColumns(), false);
			}
			
			//aggregate other input and maintain corrections
			//(existing value and corr are used in place)
			if(_op.existsCorrection())
				OperationsOnMatrixValues.incrementalAggregation(value1, corr, value2, _op, true);
			else
				OperationsOnMatrixValues.incrementalAggregation(value1, null, value2, _op, true);
			return new CorrMatrixBlock(value1, corr);
		}	
	}

	private static class ExtractMatrixBlock implements Function<CorrMatrixBlock, MatrixBlock> {
		private static final long serialVersionUID = 5242158678070843495L;
		@Override
		public MatrixBlock call(CorrMatrixBlock arg0) throws Exception {
			arg0.getValue().examSparsity();
			return arg0.getValue();
		}
	}

	private static class ExtractDoubleCell implements Function<KahanObject, Double> {
		private static final long serialVersionUID = -2873241816558275742L;
		@Override
		public Double call(KahanObject arg0) throws Exception {
			return arg0._sum;
		}
	}

	/**
	 * This aggregate function uses kahan+ with corrections to aggregate input blocks; it is meant for 
	 * reduce all operations where we can reuse the same correction block independent of the input
	 * block indexes. Note that this aggregation function does not apply to embedded corrections.
	 * 
	 */
	private static class SumSingleBlockFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock>
	{
		private static final long serialVersionUID = 1737038715965862222L;
		
		private AggregateOperator _op = null;
		private MatrixBlock _corr = null;
		private boolean _deep = false;
		
		public SumSingleBlockFunction(boolean deep) {
			_op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), CorrectionLocationType.NONE);
			_deep = deep;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//prepare combiner block
			if( arg0.getNumRows() <= 0 || arg0.getNumColumns() <= 0 ) {
				arg0.copy(arg1);
				return arg0;
			}
			else if( arg1.getNumRows() <= 0 || arg1.getNumColumns() <= 0 ) {
				return arg0;
			}
			
			//create correction block (on demand)
			if( _corr == null ) {
				_corr = new MatrixBlock(arg0.getNumRows(), arg0.getNumColumns(), false);
			}
			
			//aggregate other input (in-place if possible)
			MatrixBlock out = _deep ? new MatrixBlock(arg0) : arg0;
			OperationsOnMatrixValues.incrementalAggregation(
					out, _corr, arg1, _op, false);
			
			return out;
		}
	}

	/**
	 * Note: currently we always include the correction and use a subsequent maptopair to
	 * drop them at the end because during aggregation we dont know if we produce an
	 * intermediate or the final aggregate. 
	 */
	private static class AggregateSingleBlockFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = -3672377410407066396L;

		private AggregateOperator _op = null;
		private MatrixBlock _corr = null;
		
		public AggregateSingleBlockFunction( AggregateOperator op ) {
			_op = op;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//prepare combiner block
			if( arg0.getNumRows() == 0 && arg0.getNumColumns() == 0) {
				arg0.copy(arg1);
				return arg0;
			}
			else if( arg1.getNumRows() == 0 && arg1.getNumColumns() == 0 ) {
				return arg0;
			}
			
			//early-abort (without dense correction allocation)
			if( _op.sparseSafe && (arg0.isEmpty() | arg1.isEmpty()) )
				return arg1.isEmpty() ? arg0 : arg1;
			
			//create correction block (on demand)
			if( _op.existsCorrection() && _corr == null ) {
				_corr = new MatrixBlock(arg0.getNumRows(), arg0.getNumColumns(), false);
			}
			
			//aggregate second input (in-place)
			OperationsOnMatrixValues.incrementalAggregation(
				arg0, _op.existsCorrection() ? _corr : null, arg1, _op, true);
			
			return arg0;
		}
	}

	/**
	 * Note: currently we always include the correction and use a subsequent maptopair to
	 * drop them at the end because during aggregation we dont know if we produce an
	 * intermediate or the final aggregate.
	 */
	private static class AggregateSingleTensorBlockFunction implements Function2<TensorBlock, TensorBlock, TensorBlock>
	{
		private static final long serialVersionUID = 5665180309149919945L;

		private AggregateOperator _op = null;

		public AggregateSingleTensorBlockFunction( AggregateOperator op ) {
			_op = op;
		}

		@Override
		public TensorBlock call(TensorBlock arg0, TensorBlock arg1)
				throws Exception
		{
			//prepare combiner block
			if( arg0.isEmpty()) {
				return arg1;
			}
			else if( arg1.isEmpty() ) {
				return arg0;
			}

			// TODO remove once KahanPlus is completely replaced by plus
			if (_op.increOp.fn instanceof KahanPlus) {
				_op = new AggregateOperator(0, Plus.getPlusFnObject());
			}

			//aggregate second input (in-place)
			// TODO support DataTensor
			arg0.getBasicTensor().incrementalAggregate(_op, arg1.getBasicTensor());

			return arg0;
		}
	}

	private static class MergeBlocksFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock>
	{		
		private static final long serialVersionUID = -8881019027250258850L;
		private boolean _deep = false;
		
		@SuppressWarnings("unused")
		public MergeBlocksFunction() {
			//by default deep copy first argument
			this(true); 
		}
		
		public MergeBlocksFunction(boolean deep) {
			_deep = deep;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock b1, MatrixBlock b2) throws Exception {

			
			// sanity check input dimensions
			if(b1.getNumRows() != b2.getNumRows() || b1.getNumColumns() != b2.getNumColumns()) {
				throw new DMLRuntimeException("Mismatched block sizes for: " + b1.getNumRows() + " " + b1.getNumColumns()
				+ " " + b2.getNumRows() + " " + b2.getNumColumns());
			}
			
			// execute merge (never pass by reference)
			MatrixBlock ret = _deep ? new MatrixBlock(b1) : b1;
			ret = ret.merge(b2, false, false, _deep);
			ret.examSparsity();

			return ret;
		}
	}
}

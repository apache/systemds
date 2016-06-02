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

package org.apache.sysml.runtime.instructions.spark.utils;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.sysml.lops.PartialAggregate.CorrectionLocationType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.instructions.spark.data.CorrMatrixBlock;
import org.apache.sysml.runtime.instructions.spark.data.RowMatrixBlock;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;

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
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static MatrixBlock sumStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in )
	{
		//stable sum of all blocks with correction block per function instance
		if( TREE_AGGREGATION ) {
			return in.values().treeReduce( 
					new SumSingleBlockFunction() );	
		}
		else { //DEFAULT
			return in.values().reduce( 
					new SumSingleBlockFunction() );
		}
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sumByKey( JavaPairRDD<MatrixIndexes, MatrixBlock> in )
	{
		//sum of blocks per key, w/o exploitation of correction blocks
		return in.reduceByKey(
				new SumMultiBlockFunction());
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, Double> sumCellsByKey( JavaPairRDD<MatrixIndexes, Double> in )
	{
		//sum of blocks per key, w/o exploitation of corrections
		return in.reduceByKey(
				new SumDoubleCellsFunction());
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sumByKeyStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in )
	{
		//stable sum of blocks per key, by passing correction blocks along with aggregates 		
		JavaPairRDD<MatrixIndexes, CorrMatrixBlock> tmp = 
				in.combineByKey( new CreateBlockCombinerFunction(), 
							     new MergeSumBlockValueFunction(), 
							     new MergeSumBlockCombinerFunction() );
		
		//strip-off correction blocks from 					     
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =  
				tmp.mapValues( new ExtractMatrixBlock() );
		
		//return the aggregate rdd
		return out;
	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
//	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sumByKeyStable( MatrixCharacteristics mc, JavaPairRDD<MatrixIndexes, MatrixBlock> in )
//	{
//		//stable sum of blocks per key, by passing correction blocks along with aggregates 		
//		JavaPairRDD<MatrixIndexes, CorrMatrixBlock> tmp = 
//				in.combineByKey( new CreateBlockCombinerFunction(), 
//							     new MergeSumBlockValueFunction(), 
//							     new MergeSumBlockCombinerFunction(),
//							     new BlockPartitioner(mc, in.partitions().size()));
//		
//		//strip-off correction blocks from 					     
//		JavaPairRDD<MatrixIndexes, MatrixBlock> out =  
//				tmp.mapValues( new ExtractMatrixBlock() );
//		
//		//return the aggregate rdd
//		return out;
//	}
	
	/**
	 * 
	 * @param in
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, Double> sumCellsByKeyStable( JavaPairRDD<MatrixIndexes, Double> in )
	{
		//stable sum of blocks per key, by passing correction blocks along with aggregates 		
		JavaPairRDD<MatrixIndexes, KahanObject> tmp = 
				in.combineByKey( new CreateCellCombinerFunction(), 
							     new MergeSumCellValueFunction(), 
							     new MergeSumCellCombinerFunction() );
		
		//strip-off correction blocks from 					     
		JavaPairRDD<MatrixIndexes, Double> out =  
				tmp.mapValues( new ExtractDoubleCell() );
		
		//return the aggregate rdd
		return out;
	}
	
	/**
	 * 
	 * @param in
	 * @param aop
	 * @return
	 */
	public static MatrixBlock aggStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in, AggregateOperator aop )
	{
		//stable aggregate of all blocks with correction block per function instance
		return in.values().reduce( 
				new AggregateSingleBlockFunction(aop) );
	}
	
	/**
	 * 
	 * @param in
	 * @param aop
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> aggByKey( JavaPairRDD<MatrixIndexes, MatrixBlock> in, AggregateOperator aop )
	{
		//aggregate of blocks per key, w/o exploitation of correction blocks
		return in.reduceByKey(
				new AggregateMultiBlockFunction(aop));
	}
	
	/**
	 * 
	 * @param in
	 * @param aop
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> aggByKeyStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in, AggregateOperator aop )
	{
		//stable sum of blocks per key, by passing correction blocks along with aggregates 		
		JavaPairRDD<MatrixIndexes, CorrMatrixBlock> tmp = 
				in.combineByKey( new CreateBlockCombinerFunction(), 
							     new MergeAggBlockValueFunction(aop), 
							     new MergeAggBlockCombinerFunction(aop) );
		
		//strip-off correction blocks from 					     
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =  
				tmp.mapValues( new ExtractMatrixBlock() );
		
		//return the aggregate rdd
		return out;
	}
	
	/**
	 * 
	 * @param mc
	 * @param in
	 * @param aop
	 * @return
	 */
//	public static JavaPairRDD<MatrixIndexes, MatrixBlock> aggByKeyStable( MatrixCharacteristics mc, JavaPairRDD<MatrixIndexes, MatrixBlock> in, AggregateOperator aop )
//	{
//		//stable sum of blocks per key, by passing correction blocks along with aggregates 		
//		JavaPairRDD<MatrixIndexes, CorrMatrixBlock> tmp = 
//				in.combineByKey( new CreateBlockCombinerFunction(), 
//							     new MergeAggBlockValueFunction(aop), 
//							     new MergeAggBlockCombinerFunction(aop),
//							     new BlockPartitioner(mc, in.partitions().size()));
//		
//		//strip-off correction blocks from 					     
//		JavaPairRDD<MatrixIndexes, MatrixBlock> out =  
//				tmp.mapValues( new ExtractMatrixBlock() );
//		
//		//return the aggregate rdd
//		return out;
//	}
	
	/**
	 * Merges disjoint data of all blocks per key.
	 * 
	 * Note: The behavior of this method is undefined for both sparse and dense data if the 
	 * assumption of disjoint data is violated.
	 * 
	 * @param in
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> mergeByKey( JavaPairRDD<MatrixIndexes, MatrixBlock> in )
	{
		return in.reduceByKey(
				new MergeBlocksFunction());
	}
	
	/**
	 * 
	 * @param mc
	 * @param in
	 * @return
	 */
//	public static JavaPairRDD<MatrixIndexes, MatrixBlock> mergeByKey( MatrixCharacteristics mc, JavaPairRDD<MatrixIndexes, MatrixBlock> in )
//	{
//		return in.reduceByKey(
//				new BlockPartitioner(mc, in.partitions().size()),
//				new MergeBlocksFunction());
//	}
	
	/**
	 * Merges disjoint data of all blocks per key.
	 * 
	 * Note: The behavior of this method is undefined for both sparse and dense data if the 
	 * assumption of disjoint data is violated.
	 * 
	 * @param in
	 * @return
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> mergeRowsByKey( JavaPairRDD<MatrixIndexes, RowMatrixBlock> in )
	{
		return in.combineByKey( new CreateRowBlockCombinerFunction(), 
							    new MergeRowBlockValueFunction(), 
							    new MergeRowBlockCombinerFunction() );
	}
	
	/**
	 * 
	 */
	private static class CreateBlockCombinerFunction implements Function<MatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = -3666451526776017343L;

		@Override
		public CorrMatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			return new CorrMatrixBlock(arg0);
		}	
	}
	
	/**
	 * 
	 */
	private static class MergeSumBlockValueFunction implements Function2<CorrMatrixBlock, MatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 3703543699467085539L;
		
		private AggregateOperator _op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.NONE);	
		
		@Override
		public CorrMatrixBlock call(CorrMatrixBlock arg0, MatrixBlock arg1) 
			throws Exception 
		{
			//get current block and correction
			MatrixBlock value = arg0.getValue();
			MatrixBlock corr = arg0.getCorrection();
			
			//correction block allocation on demand
			if( corr == null ){
				corr = new MatrixBlock(value.getNumRows(), value.getNumColumns(), false);
			}
			
			//aggregate other input and maintain corrections 
			//(existing value and corr are used in place)
			OperationsOnMatrixValues.incrementalAggregation(value, corr, arg1, _op, false);
			return new CorrMatrixBlock(value, corr);
		}	
	}
	
	/**
	 * 
	 */
	private static class MergeSumBlockCombinerFunction implements Function2<CorrMatrixBlock, CorrMatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 7664941774566119853L;
		
		private AggregateOperator _op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.NONE);	
		
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
				corr = (arg1.getCorrection()!=null)?arg1.getCorrection():
					new MatrixBlock(value1.getNumRows(), value1.getNumColumns(), false);
			}
			
			//aggregate other input and maintain corrections
			//(existing value and corr are used in place)
			OperationsOnMatrixValues.incrementalAggregation(value1, corr, value2, _op, false);
			return new CorrMatrixBlock(value1, corr);
		}	
	}

	/**
	 *
	 */
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
	
	/**
	 * 
	 */
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
	
	/**
	 * 
	 */
	private static class MergeRowBlockCombinerFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 5142967296705548000L;

		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1) 
			throws Exception 
		{
			//merge second matrix block into first
			MatrixBlock out = arg0; //in-place update
			out.merge(arg1, false);
			out.examSparsity();
			
			return out;
		}	
	}
	
	/**
	 * 
	 */
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
	
	/**
	 * 
	 */
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
	
	/**
	 * 
	 */
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
	
	/**
	 * 
	 */
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
			if( corr == null && _op.correctionExists ){
				corr = new MatrixBlock(value.getNumRows(), value.getNumColumns(), false);
			}
			
			//aggregate other input and maintain corrections 
			//(existing value and corr are used in place)
			if(_op.correctionExists)
				OperationsOnMatrixValues.incrementalAggregation(value, corr, arg1, _op, true);
			else
				OperationsOnMatrixValues.incrementalAggregation(value, null, arg1, _op, true);
			return new CorrMatrixBlock(value, corr);
		}	
	}
	
	/**
	 * 
	 */
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
			if( corr == null && _op.correctionExists) {
				corr = (arg1.getCorrection()!=null)?arg1.getCorrection():
					new MatrixBlock(value1.getNumRows(), value1.getNumColumns(), false);
			}
			
			//aggregate other input and maintain corrections
			//(existing value and corr are used in place)
			if(_op.correctionExists)
				OperationsOnMatrixValues.incrementalAggregation(value1, corr, value2, _op, true);
			else
				OperationsOnMatrixValues.incrementalAggregation(value1, null, value2, _op, true);
			return new CorrMatrixBlock(value1, corr);
		}	
	}
	
	/**
	 * 
	 */
	private static class ExtractMatrixBlock implements Function<CorrMatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = 5242158678070843495L;

		@Override
		public MatrixBlock call(CorrMatrixBlock arg0) 
			throws Exception 
		{
			return arg0.getValue();
		}	
	}
	
	/**
	 * 
	 */
	private static class ExtractDoubleCell implements Function<KahanObject, Double> 
	{
		private static final long serialVersionUID = -2873241816558275742L;

		@Override
		public Double call(KahanObject arg0) 
			throws Exception 
		{
			//return sum and drop correction
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
		
		public SumSingleBlockFunction()
		{
			_op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.NONE);	
			_corr = null;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//create correction block (on demand)
			if( _corr == null ){
				_corr = new MatrixBlock(arg0.getNumRows(), arg0.getNumColumns(), false);
			}
			
			//copy one input to output
			MatrixBlock out = new MatrixBlock(arg0);
			
			//aggregate other input
			OperationsOnMatrixValues.incrementalAggregation(out, _corr, arg1, _op, false);
			
			return out;
		}
	}
	
	/**
	 * This aggregate function uses kahan+ with corrections to aggregate input blocks; it is meant for 
	 * reducebykey operations where we CANNOT reuse the same correction block independent of the input
	 * block indexes. Note that this aggregation function does not apply to embedded corrections.
	 * 
	 */
	private static class SumMultiBlockFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = -4015979658416853324L;

		private AggregateOperator _op = null;
		private MatrixBlock _corr = null;
		
		public SumMultiBlockFunction()
		{
			_op = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), true, CorrectionLocationType.NONE);	
			_corr = new MatrixBlock();
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//copy one input to output
			MatrixBlock out = new MatrixBlock(arg0);
			
			//aggregate other input
			_corr.reset(out.getNumRows(), out.getNumColumns());
			OperationsOnMatrixValues.incrementalAggregation(out, _corr, arg1, _op, false);
			
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
		
		public AggregateSingleBlockFunction( AggregateOperator op )
		{
			_op = op;	
			_corr = null;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//copy one first input
			MatrixBlock out = new MatrixBlock(arg0); 
			
			//create correction block (on demand)
			if( _corr == null ){
				_corr = new MatrixBlock(arg0.getNumRows(), arg0.getNumColumns(), false);
			}
			
			//aggregate second input
			if(_op.correctionExists) {
				OperationsOnMatrixValues.incrementalAggregation(
						out, _corr, arg1, _op, true);
			}
			else {
				OperationsOnMatrixValues.incrementalAggregation(
						out, null, arg1, _op, true);
			}
			
			return out;
		}
	}
	
	/**
	 * Note: currently we always include the correction and use a subsequent maptopair to
	 * drop them at the end because during aggregation we dont know if we produce an
	 * intermediate or the final aggregate. 
	 */
	private static class AggregateMultiBlockFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{
		private static final long serialVersionUID = -3672377410407066396L;

		private AggregateOperator _op = null;
		private MatrixBlock _corr = null;
		
		public AggregateMultiBlockFunction( AggregateOperator op )
		{
			_op = op;	
			_corr = new MatrixBlock();
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0, MatrixBlock arg1)
			throws Exception 
		{
			//copy one first input
			MatrixBlock out = new MatrixBlock(arg0); 
			
			//aggregate second input
			_corr.reset(out.getNumRows(), out.getNumColumns());
			if(_op.correctionExists) {
				OperationsOnMatrixValues.incrementalAggregation(
						out, _corr, arg1, _op, true);
			}
			else {
				OperationsOnMatrixValues.incrementalAggregation(
						out, null, arg1, _op, true);
			}
			
			return out;
		}
	}
	
	/**
	 * 
	 */
	private static class MergeBlocksFunction implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> 
	{		
		private static final long serialVersionUID = -8881019027250258850L;

		@Override
		public MatrixBlock call(MatrixBlock b1, MatrixBlock b2) 
			throws Exception 
		{
			// sanity check input dimensions
			if (b1.getNumRows() != b2.getNumRows() || b1.getNumColumns() != b2.getNumColumns()) {
				throw new DMLRuntimeException("Mismatched block sizes for: "
						+ b1.getNumRows() + " " + b1.getNumColumns() + " "
						+ b2.getNumRows() + " " + b2.getNumColumns());
			}

			// execute merge (never pass by reference)
			MatrixBlock ret = new MatrixBlock(b1);
			ret.merge(b2, false);
			ret.examSparsity();
			
			// sanity check output number of non-zeros
			if (ret.getNonZeros() != b1.getNonZeros() + b2.getNonZeros()) {
				throw new DMLRuntimeException("Number of non-zeros does not match: "
						+ ret.getNonZeros() + " != " + b1.getNonZeros() + " + " + b2.getNonZeros());
			}

			return ret;
		}

	}
	
	/**
	 * 
	 */
	private static class SumDoubleCellsFunction implements Function2<Double, Double, Double> 
	{
		private static final long serialVersionUID = -8167625566734873796L;

		@Override
		public Double call(Double v1, Double v2) throws Exception {
			return v1 + v2;
		}	
	}
	
	/**
	 * @param: in
	 * @return: 
	 */
	public static JavaPairRDD<?, FrameBlock> mergeByFrameKey( JavaPairRDD<?, FrameBlock> in )
	{
		return in.reduceByKey(
				new MergeFrameBlocksFunction());
	}
	
	/**
	 * 
	 */
	private static class MergeFrameBlocksFunction implements Function2<FrameBlock, FrameBlock, FrameBlock> 
	{		
		private static final long serialVersionUID = -8881019027250258850L;

		@Override
		public FrameBlock call(FrameBlock b1, FrameBlock b2) 
			throws Exception 
		{
			// sanity check input dimensions
			if (b1.getNumRows() != b2.getNumRows() || b1.getNumColumns() != b2.getNumColumns()) {
				throw new DMLRuntimeException("Mismatched frame block sizes for: "
						+ b1.getNumRows() + " " + b1.getNumColumns() + " "
						+ b2.getNumRows() + " " + b2.getNumColumns());
			}

			// execute merge (never pass by reference)
			FrameBlock ret = new FrameBlock(b1);
			ret.merge(b2);
			
			return ret;
		}

	}
	

}

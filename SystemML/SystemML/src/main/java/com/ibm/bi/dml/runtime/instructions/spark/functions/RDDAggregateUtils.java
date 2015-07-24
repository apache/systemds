/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.lops.PartialAggregate.CorrectionLocationType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.functionobjects.KahanPlus;
import com.ibm.bi.dml.runtime.instructions.spark.data.CorrMatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.operators.AggregateOperator;

/**
 * Collection of utility methods for aggregating binary block rdds. As a general
 * policy always call stable algorithms which maintain corrections over blocks
 * per key. The performance overhead over a simple reducebykey is roughly 7-10% 
 * and with that acceptable. 
 * 
 */
public class RDDAggregateUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sumByKeyStable( JavaPairRDD<MatrixIndexes, MatrixBlock> in )
	{
		//stable sum of blocks per key, by passing correction blocks along with aggregates 		
		JavaPairRDD<MatrixIndexes, CorrMatrixBlock> tmp = 
				in.combineByKey( new CreateCombinerFunction(), 
							     new MergeSumValueFunction(), 
							     new MergeSumCombinerFunction() );
		
		//strip-off correction blocks from 					     
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =  
				tmp.mapValues( new ExtractMatrixBlock() );
		
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
				in.combineByKey( new CreateCombinerFunction(), 
							     new MergeAggValueFunction(aop), 
							     new MergeAggCombinerFunction(aop) );
		
		//strip-off correction blocks from 					     
		JavaPairRDD<MatrixIndexes, MatrixBlock> out =  
				tmp.mapValues( new ExtractMatrixBlock() );
		
		//return the aggregate rdd
		return out;
	}
	
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
	 */
	private static class CreateCombinerFunction implements Function<MatrixBlock, CorrMatrixBlock> 
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
	private static class MergeSumValueFunction implements Function2<CorrMatrixBlock, MatrixBlock, CorrMatrixBlock> 
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
	private static class MergeSumCombinerFunction implements Function2<CorrMatrixBlock, CorrMatrixBlock, CorrMatrixBlock> 
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
	private static class MergeAggValueFunction implements Function2<CorrMatrixBlock, MatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 389422125491172011L;
		
		private AggregateOperator _op = null;	
		
		public MergeAggValueFunction(AggregateOperator aop)
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
	private static class MergeAggCombinerFunction implements Function2<CorrMatrixBlock, CorrMatrixBlock, CorrMatrixBlock> 
	{
		private static final long serialVersionUID = 4803711632648880797L;
		
		private AggregateOperator _op = null;
		
		public MergeAggCombinerFunction(AggregateOperator aop)
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
				throw new DMLRuntimeException("Mismatched block sizes: "
						+ b1.getNumRows() + " " + b1.getNumColumns() + " "
						+ b2.getNumRows() + " " + b2.getNumColumns());
			}

			// execute merge (never pass by reference)
			MatrixBlock ret = new MatrixBlock(b1);
			ret.merge(b2, false);

			// sanity check output number of non-zeros
			if (ret.getNonZeros() != b1.getNonZeros() + b2.getNonZeros()) {
				throw new DMLRuntimeException("Number of non-zeros does not match: "
						+ ret.getNonZeros() + " != " + b1.getNonZeros() + " + " + b2.getNonZeros());
			}

			return ret;
		}

	}
}

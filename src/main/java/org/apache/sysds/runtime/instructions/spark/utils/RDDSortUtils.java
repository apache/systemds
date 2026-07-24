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
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.SortIndex;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBlock;
import org.apache.sysds.runtime.instructions.spark.data.RowMatrixBlock;
import org.apache.sysds.runtime.instructions.spark.functions.ReplicateVectorFunction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.SortUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;

public class RDDSortUtils 
{

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortByVal( JavaPairRDD<MatrixIndexes, MatrixBlock> in, long rlen, int blen )
	{
		//create value-index rdd from inputs
		JavaRDD<Double> dvals = in.values()
				.flatMap(new ExtractDoubleValuesFunction());
	
		//sort (creates sorted range per partition)
		long hdfsBlocksize = InfrastructureAnalyzer.getHDFSBlockSize();
		int numPartitions = (int)Math.ceil(((double)rlen*8)/hdfsBlocksize);
		JavaRDD<Double> sdvals = dvals
				.sortBy(new CreateDoubleKeyFunction(), true, numPartitions);
		
		//create binary block output
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
				.zipWithIndex()
				.mapPartitionsToPair(new ConvertToBinaryBlockFunction(rlen, blen));
		ret = RDDAggregateUtils.mergeByKey(ret, false);
		
		return ret;
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortByVal( JavaPairRDD<MatrixIndexes, MatrixBlock> in, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> in2, long rlen, int blen )
	{
		//create value-index rdd from inputs
		JavaRDD<DoublePair> dvals = in.join(in2).values()
			.flatMap(new ExtractDoubleValuesFunction2());
	
		//sort (creates sorted range per partition)
		long hdfsBlocksize = InfrastructureAnalyzer.getHDFSBlockSize();
		int numPartitions = (int)Math.ceil(((double)rlen*8)/hdfsBlocksize);
		JavaRDD<DoublePair> sdvals = dvals
			.sortBy(new CreateDoubleKeyFunction2(), true, numPartitions);

		//create binary block output
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
			.zipWithIndex()
			.mapPartitionsToPair(new ConvertToBinaryBlockFunction2(rlen, blen));
		ret = RDDAggregateUtils.mergeByKey(ret, false);		
		
		return ret;
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortByVals(
		JavaPairRDD<MatrixIndexes, MatrixBlock> in, long rlen, long clen, int blen )
	{
		//create value-index rdd from inputs
		JavaRDD<MatrixBlock> dvals = in.values()
			.flatMap(new ExtractRowsFunction());
		
		//sort (creates sorted range per partition)
		int numPartitions = SparkUtils.getNumPreferredPartitions(
			new MatrixCharacteristics(rlen, clen, blen, blen), in);
		JavaRDD<MatrixBlock> sdvals = dvals
			.sortBy(new CreateDoubleKeysFunction(), true, numPartitions);
		
		//create binary block output
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
			.zipWithIndex()
			.mapPartitionsToPair(new ConvertToBinaryBlockFunction5(rlen, blen));
		ret = RDDAggregateUtils.mergeByKey(ret, false);
		
		return ret;
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortIndexesByVal( JavaPairRDD<MatrixIndexes, MatrixBlock> val, 
			boolean asc, long rlen, int blen )
	{
		//create value-index rdd from inputs
		JavaPairRDD<ValueIndexPair, Double> dvals = val
			.flatMapToPair(new ExtractDoubleValuesWithIndexFunction(blen));
		
		//sort (creates sorted range per partition)
		long hdfsBlocksize = InfrastructureAnalyzer.getHDFSBlockSize();
		int numPartitions = (int)Math.ceil(((double)rlen*16)/hdfsBlocksize);
		JavaRDD<ValueIndexPair> sdvals = dvals
			.sortByKey(new IndexComparator(asc), true, numPartitions)
			.keys(); //workaround for index comparator
		
		//create binary block output
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
			.zipWithIndex()
			.mapPartitionsToPair(new ConvertToBinaryBlockFunction3(rlen, blen));
		ret = RDDAggregateUtils.mergeByKey(ret, false);
		
		return ret;
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortIndexesByVals( JavaPairRDD<MatrixIndexes, MatrixBlock> in,
			boolean asc, long rlen, long clen, int blen )
	{
		//create value-index rdd from inputs
		JavaPairRDD<ValuesIndexPair, double[]> dvals = in
			.flatMapToPair(new ExtractDoubleValuesWithIndexFunction2(blen));
		
		//sort (creates sorted range per partition)
		int numPartitions = SparkUtils.getNumPreferredPartitions(
			new MatrixCharacteristics(rlen, clen+1, blen, blen));
		JavaRDD<ValuesIndexPair> sdvals = dvals
			.sortByKey(new IndexComparator2(asc), true, numPartitions)
			.keys(); //workaround for index comparator
		
		//create binary block output
		JavaPairRDD<MatrixIndexes, MatrixBlock> ret = sdvals
			.zipWithIndex()
			.mapPartitionsToPair(new ConvertToBinaryBlockFunction6(rlen, blen));
		ret = RDDAggregateUtils.mergeByKey(ret, false);
		
		return ret;
	}

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortDataByVal( JavaPairRDD<MatrixIndexes, MatrixBlock> val, 
		JavaPairRDD<MatrixIndexes, MatrixBlock> data, boolean asc, long rlen, long clen, int blen )
	{
		//create value-index rdd from inputs
		JavaPairRDD<ValueIndexPair, Double> dvals = val
			.flatMapToPair(new ExtractDoubleValuesWithIndexFunction(blen));
		
		//sort (creates sorted range per partition)
		long hdfsBlocksize = InfrastructureAnalyzer.getHDFSBlockSize();
		int numPartitions = (int)Math.ceil(((double)rlen*16)/hdfsBlocksize);
		JavaRDD<ValueIndexPair> sdvals = dvals
			.sortByKey(new IndexComparator(asc), true, numPartitions)
			.keys(); //workaround for index comparator
		
		//create target indexes by original index
		JavaPairRDD<MatrixIndexes, MatrixBlock> ixmap = sdvals
			.zipWithIndex()
			.mapToPair(new ExtractIndexFunction())
			.sortByKey()
			.mapPartitionsToPair(new ConvertToBinaryBlockFunction4(rlen, blen));
		ixmap = RDDAggregateUtils.mergeByKey(ixmap, false);
		
		//actual data sort
		return sortDataByIx(data, ixmap, rlen, clen, blen);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortDataByVals( JavaPairRDD<MatrixIndexes, MatrixBlock> val, 
		JavaPairRDD<MatrixIndexes, MatrixBlock> data, boolean asc, long rlen, long clen, long clen2, int blen )
	{
		//create value-index rdd from inputs
		JavaPairRDD<ValuesIndexPair, double[]> dvals = val
			.flatMapToPair(new ExtractDoubleValuesWithIndexFunction2(blen));
		
		//sort (creates sorted range per partition)
		int numPartitions = SparkUtils.getNumPreferredPartitions(
			new MatrixCharacteristics(rlen, clen2+1, blen, blen));
		JavaRDD<ValuesIndexPair> sdvals = dvals
			.sortByKey(new IndexComparator2(asc), true, numPartitions)
			.keys(); //workaround for index comparator
		
		//create target indexes by original index
		JavaPairRDD<MatrixIndexes, MatrixBlock> ixmap = sdvals
			.zipWithIndex()
			.mapToPair(new ExtractIndexFunction2())
			.sortByKey()
			.mapPartitionsToPair(new ConvertToBinaryBlockFunction4(rlen, blen));
		ixmap = RDDAggregateUtils.mergeByKey(ixmap, false);
		
		//actual data sort
		return sortDataByIx(data, ixmap, rlen, clen, blen);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortDataByIx(JavaPairRDD<MatrixIndexes,MatrixBlock> data,
		JavaPairRDD<MatrixIndexes,MatrixBlock> ixmap, long rlen, long clen, int blen) {
		//replicate indexes for all column blocks
		long numRep = (long)Math.ceil((double)clen/blen);
		JavaPairRDD<MatrixIndexes, MatrixBlock> rixmap = ixmap
			.flatMapToPair(new ReplicateVectorFunction(false, numRep));
		
		//create binary block output
		JavaPairRDD<MatrixIndexes, RowMatrixBlock> ret = data
			.join(rixmap)
			.mapPartitionsToPair(new ShuffleMatrixBlockRowsFunction(rlen, blen));
		return RDDAggregateUtils.mergeRowsByKey(ret);
	}
	
	/**
	 * This function collects and sorts value column in memory and then broadcasts it. 
	 * 
	 * @param val value as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @param data data as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @param asc if true, sort ascending
	 * @param rlen number of rows
	 * @param clen number of columns
	 * @param blen block length
	 * @param sec spark execution context
	 * @param r_op reorg operator
	 * @return data as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> sortDataByValMemSort( JavaPairRDD<MatrixIndexes, MatrixBlock> val, 
			JavaPairRDD<MatrixIndexes, MatrixBlock> data, boolean asc, long rlen, long clen, int blen, 
			SparkExecutionContext sec, ReorgOperator r_op) 
	{
		//collect orderby column for in-memory sorting
		MatrixBlock inMatBlock = SparkExecutionContext
			.toMatrixBlock(val, (int)rlen, 1, blen, -1);

		//in-memory sort operation (w/ index return: source index in target position)
		ReorgOperator lrop = new ReorgOperator(new SortIndex(1, !asc, true));
		MatrixBlock sortedIx = inMatBlock.reorgOperations(lrop, new MatrixBlock(), -1, -1, -1);
		
		//flip sort indices from <source ix in target pos> to <target ix in source pos>
		MatrixBlock sortedIxSrc = new MatrixBlock(sortedIx.getNumRows(), 1, false); 
		for (int i=0; i < sortedIx.getNumRows(); i++) 
			sortedIxSrc.set((int)sortedIx.get(i,0)-1, 0, i+1);

		//broadcast index vector
		PartitionedBlock<MatrixBlock> pmb = new PartitionedBlock<>(sortedIxSrc, blen);
		Broadcast<PartitionedBlock<MatrixBlock>> pmb2 = sec.getSparkContext().broadcast(pmb);

		//sort data with broadcast index vector
		JavaPairRDD<MatrixIndexes, RowMatrixBlock> ret = data
				.mapPartitionsToPair(new ShuffleMatrixBlockRowsInMemFunction(rlen, blen, pmb2));
		return RDDAggregateUtils.mergeRowsByKey(ret);
	}

	private static class ExtractDoubleValuesFunction implements FlatMapFunction<MatrixBlock,Double> 
	{
		private static final long serialVersionUID = 6888003502286282876L;

		@Override
		public Iterator<Double> call(MatrixBlock arg0) 
			throws Exception {
			return DataConverter.convertToDoubleList(arg0).iterator();
		}
	}
	
	private static class ExtractRowsFunction implements FlatMapFunction<MatrixBlock,MatrixBlock> 
	{
		private static final long serialVersionUID = -2786968469468554974L;

		@Override
		public Iterator<MatrixBlock> call(MatrixBlock arg0) 
			throws Exception {
			ArrayList<MatrixBlock> rows = new ArrayList<>();
			for(int i=0; i<arg0.getNumRows(); i++)
				rows.add(arg0.slice(i, i));
			return rows.iterator();
		}
	}

	private static class ExtractDoubleValuesFunction2 implements FlatMapFunction<Tuple2<MatrixBlock,MatrixBlock>,DoublePair> 
	{
		private static final long serialVersionUID = 2132672563825289022L;

		@Override
		public Iterator<DoublePair> call(Tuple2<MatrixBlock,MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<DoublePair> ret = new ArrayList<>(); 
			MatrixBlock mb1 = arg0._1();
			MatrixBlock mb2 = arg0._2();
			
			for( int i=0; i<mb1.getNumRows(); i++) {
				ret.add(new DoublePair(
						mb1.get(i, 0),
						mb2.get(i, 0)));
			}
			
			return ret.iterator();
		}		
	}
	
	private static class ExtractDoubleValuesWithIndexFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,ValueIndexPair,Double> 
	{
		private static final long serialVersionUID = -3976735381580482118L;
		
		private int _blen = -1;
		
		public ExtractDoubleValuesWithIndexFunction(int blen)
		{
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<ValueIndexPair,Double>> call(Tuple2<MatrixIndexes,MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<ValueIndexPair,Double>> ret = new ArrayList<>(); 
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			
			long ixoffset = (ix.getRowIndex()-1)*_blen;
			for( int i=0; i<mb.getNumRows(); i++) {
				double val = mb.get(i, 0);
				ret.add(new Tuple2<>(new ValueIndexPair(val,ixoffset+i+1), val));
			}
			
			return ret.iterator();
		}
	}
	
	private static class ExtractDoubleValuesWithIndexFunction2 implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>,ValuesIndexPair,double[]> 
	{
		private static final long serialVersionUID = 8358254634903633283L;
		
		private final int _blen;
		
		public ExtractDoubleValuesWithIndexFunction2(int blen) {
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<ValuesIndexPair,double[]>> call(Tuple2<MatrixIndexes,MatrixBlock> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<ValuesIndexPair,double[]>> ret = new ArrayList<>(); 
			MatrixIndexes ix = arg0._1();
			MatrixBlock mb = arg0._2();
			
			long ixoffset = (ix.getRowIndex()-1)*_blen;
			for( int i=0; i<mb.getNumRows(); i++) {
				double[] vals = DataConverter.convertToDoubleVector(
					mb.slice(i, i));
				ret.add(new Tuple2<>(new ValuesIndexPair(vals,ixoffset+i+1), vals));
			}
			
			return ret.iterator();
		}
	}

	private static class CreateDoubleKeyFunction implements Function<Double,Double> 
	{
		private static final long serialVersionUID = 2021786334763247835L;

		@Override
		public Double call(Double arg0) 
			throws Exception 
		{
			return arg0;
		}
	}

	private static class CreateDoubleKeyFunction2 implements Function<DoublePair,Double> 
	{
		private static final long serialVersionUID = -7954819651274239592L;

		@Override
		public Double call(DoublePair arg0) 
			throws Exception 
		{
			return arg0.val1;
		}
	}

	private static class CreateDoubleKeysFunction implements Function<MatrixBlock,double[]> 
	{
		private static final long serialVersionUID = 4316858496746520340L;

		@Override
		public double[] call(MatrixBlock row) throws Exception {
			return DataConverter.convertToDoubleVector(row);
		}
	}
	
	private static class ExtractIndexFunction implements PairFunction<Tuple2<ValueIndexPair,Long>,Long,Long> {
		private static final long serialVersionUID = -4553468724131249535L;

		@Override
		public Tuple2<Long, Long> call(Tuple2<ValueIndexPair,Long> arg0) throws Exception {
			return new Tuple2<>(arg0._1().ix, arg0._2());
		}
	}
	
	private static class ExtractIndexFunction2 implements PairFunction<Tuple2<ValuesIndexPair,Long>,Long,Long> {
		private static final long serialVersionUID = -1366455446597907270L;

		@Override
		public Tuple2<Long, Long> call(Tuple2<ValuesIndexPair,Long> arg0) throws Exception {
			return new Tuple2<>(arg0._1().ix, arg0._2());
		}
	}

	private static class ConvertToBinaryBlockFunction implements PairFlatMapFunction<Iterator<Tuple2<Double,Long>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 5000298196472931653L;
		
		private long _rlen = -1;
		private int _blen = -1;
		
		public ConvertToBinaryBlockFunction(long rlen, int blen) {
			_rlen = rlen;
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Double,Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			MatrixIndexes ix = null;
			MatrixBlock mb = null;
			
			while( arg0.hasNext() ) 
			{
				Tuple2<Double,Long> val = arg0.next();
				long valix = val._2 + 1;
				long rix = UtilFunctions.computeBlockIndex(valix, _blen);
				int pos = UtilFunctions.computeCellInBlock(valix, _blen);
				
				if( ix == null || ix.getRowIndex() != rix )
				{
					if( ix !=null )
						ret.add(new Tuple2<>(ix,mb));
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 1, false);	
				}
				
				mb.set(pos, 0, val._1);
			}
			
			//flush last block
			if( mb!=null && mb.getNonZeros() != 0 )
				ret.add(new Tuple2<>(ix,mb));
			return ret.iterator();
		}
	}

	private static class ConvertToBinaryBlockFunction2 implements PairFlatMapFunction<Iterator<Tuple2<DoublePair,Long>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = -8638434373377180192L;
		
		private long _rlen = -1;
		private int _blen = -1;
		
		public ConvertToBinaryBlockFunction2(long rlen, int blen) {
			_rlen = rlen;
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<DoublePair,Long>> arg0) 
			throws Exception
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			
			MatrixIndexes ix = null;
			MatrixBlock mb = null;
			
			while( arg0.hasNext() ) 
			{
				Tuple2<DoublePair,Long> val = arg0.next();
				long valix = val._2 + 1;
				long rix = UtilFunctions.computeBlockIndex(valix, _blen);
				int pos = UtilFunctions.computeCellInBlock(valix, _blen);
				
				if( ix == null || ix.getRowIndex() != rix )
				{
					if( ix !=null )
						ret.add(new Tuple2<>(ix,mb));
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 2, false);
				}
				
				mb.set(pos, 0, val._1.val1);
				mb.set(pos, 1, val._1.val2);
			}
			
			//flush last block
			if( mb!=null && mb.getNonZeros() != 0 )
				ret.add(new Tuple2<>(ix,mb));
			
			return ret.iterator();
		}
	}

	private static class ConvertToBinaryBlockFunction3 implements PairFlatMapFunction<Iterator<Tuple2<ValueIndexPair,Long>>,MatrixIndexes,MatrixBlock> 
	{		
		private static final long serialVersionUID = 9113122668214965797L;
		
		private long _rlen = -1;
		private int _blen = -1;
		
		public ConvertToBinaryBlockFunction3(long rlen, int blen)
		{
			_rlen = rlen;
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<ValueIndexPair,Long>> arg0) 
			throws Exception
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			
			MatrixIndexes ix = null;
			MatrixBlock mb = null;
			
			while( arg0.hasNext() ) 
			{
				Tuple2<ValueIndexPair,Long> val = arg0.next();
				long valix = val._2 + 1;
				long rix = UtilFunctions.computeBlockIndex(valix, _blen);
				int pos = UtilFunctions.computeCellInBlock(valix, _blen);
				
				if( ix == null || ix.getRowIndex() != rix )
				{
					if( ix !=null )
						ret.add(new Tuple2<>(ix,mb));
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 1, false);	
				}
				
				mb.set(pos, 0, val._1.ix);
			}
			
			//flush last block
			if( mb!=null && mb.getNonZeros() != 0 )
				ret.add(new Tuple2<>(ix,mb));
			
			return ret.iterator();
		}
	}

	private static class ConvertToBinaryBlockFunction4 implements PairFlatMapFunction<Iterator<Tuple2<Long,Long>>,MatrixIndexes,MatrixBlock> 
	{	
		private static final long serialVersionUID = 9113122668214965797L;
		
		private long _rlen = -1;
		private int _blen = -1;
		
		public ConvertToBinaryBlockFunction4(long rlen, int blen)
		{
			_rlen = rlen;
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<Long,Long>> arg0) 
			throws Exception
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			
			MatrixIndexes ix = null;
			MatrixBlock mb = null;
			
			while( arg0.hasNext() ) 
			{
				Tuple2<Long,Long> val = arg0.next();
				long valix = val._1;
				long rix = UtilFunctions.computeBlockIndex(valix, _blen);
				int pos = UtilFunctions.computeCellInBlock(valix, _blen);
				
				if( ix == null || ix.getRowIndex() != rix )
				{
					if( ix !=null )
						ret.add(new Tuple2<>(ix,mb));
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 1, false);	
				}
				
				mb.set(pos, 0, val._2+1);
			}
			
			//flush last block
			if( mb!=null && mb.getNonZeros() != 0 )
				ret.add(new Tuple2<>(ix,mb));
			
			return ret.iterator();
		}
	}
	
	private static class ConvertToBinaryBlockFunction5 implements PairFlatMapFunction<Iterator<Tuple2<MatrixBlock,Long>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 6357994683868091724L;
		
		private long _rlen = -1;
		private int _blen = -1;
		
		public ConvertToBinaryBlockFunction5(long rlen, int blen)
		{
			_rlen = rlen;
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<MatrixBlock,Long>> arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			MatrixIndexes ix = null;
			MatrixBlock mb = null;
			
			while( arg0.hasNext() ) 
			{
				Tuple2<MatrixBlock,Long> val = arg0.next();
				long valix = val._2 + 1;
				long rix = UtilFunctions.computeBlockIndex(valix, _blen);
				int pos = UtilFunctions.computeCellInBlock(valix, _blen);
				
				if( ix == null || ix.getRowIndex() != rix ) {
					if( ix !=null )
						ret.add(new Tuple2<>(ix,mb));
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, val._1.getNumColumns(), false);
				}
				
				mb.leftIndexingOperations(val._1, pos, pos, 0, val._1.getNumColumns()-1, mb, UpdateType.INPLACE);
			}
			
			//flush last block
			if( mb!=null && mb.getNonZeros() != 0 )
				ret.add(new Tuple2<>(ix,mb));
			return ret.iterator();
		}
	}
	
	private static class ConvertToBinaryBlockFunction6 implements PairFlatMapFunction<Iterator<Tuple2<ValuesIndexPair,Long>>,MatrixIndexes,MatrixBlock> 
	{
		private static final long serialVersionUID = 5351649694631911694L;
		
		private long _rlen = -1;
		private int _blen = -1;
		
		public ConvertToBinaryBlockFunction6(long rlen, int blen)
		{
			_rlen = rlen;
			_blen = blen;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Iterator<Tuple2<ValuesIndexPair,Long>> arg0) 
			throws Exception
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<>();
			
			MatrixIndexes ix = null;
			MatrixBlock mb = null;
			
			while( arg0.hasNext() ) 
			{
				Tuple2<ValuesIndexPair,Long> val = arg0.next();
				long valix = val._2 + 1;
				long rix = UtilFunctions.computeBlockIndex(valix, _blen);
				int pos = UtilFunctions.computeCellInBlock(valix, _blen);
				
				if( ix == null || ix.getRowIndex() != rix ) {
					if( ix !=null )
						ret.add(new Tuple2<>(ix,mb));
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					ix = new MatrixIndexes(rix,1);
					mb = new MatrixBlock((int)len, 1, false);
				}
				
				mb.set(pos, 0, val._1.ix);
			}
			
			//flush last block
			if( mb!=null && mb.getNonZeros() != 0 )
				ret.add(new Tuple2<>(ix,mb));
			
			return ret.iterator();
		}
	}
	
	private static class ShuffleMatrixBlockRowsFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,Tuple2<MatrixBlock,MatrixBlock>>>,MatrixIndexes,RowMatrixBlock> 
	{	
		private static final long serialVersionUID = 6885207719329119646L;
		
		private long _rlen = -1;
		private int _blen = -1;
		
		public ShuffleMatrixBlockRowsFunction(long rlen, int blen)
		{
			_rlen = rlen;
			_blen = blen;
		}

		@Override
		public ShuffleMatrixIterator call(Iterator<Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>>> arg0)
			throws Exception 
		{
			return new ShuffleMatrixIterator(arg0);
		}
		
		/**
		 * Lazy iterator to prevent blk output for better resource efficiency; 
		 * This also lowered garbage collection overhead.
		 */
		private class ShuffleMatrixIterator implements Iterable<Tuple2<MatrixIndexes, RowMatrixBlock>>, Iterator<Tuple2<MatrixIndexes, RowMatrixBlock>>
		{
			private Iterator<Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>>> _inIter = null;
			private Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>> _currBlk = null;
			private int _currPos = -1;
			
			public ShuffleMatrixIterator(Iterator<Tuple2<MatrixIndexes, Tuple2<MatrixBlock, MatrixBlock>>> in) {
				_inIter = in;
			}

			@Override
			public Iterator<Tuple2<MatrixIndexes, RowMatrixBlock>> iterator() {
				return this;
			}

			@Override
			public boolean hasNext() {
				return _currBlk != null || _inIter.hasNext();
			}
			
			@Override
			public Tuple2<MatrixIndexes, RowMatrixBlock> next() 
			{
				//pull next input block (if required)
				if( _currBlk == null ){
					_currBlk = _inIter.next();
					_currPos = 0;
				}
				
				try
				{
					//produce next output tuple
					MatrixIndexes ixmap = _currBlk._1();
					MatrixBlock data = _currBlk._2()._1();
					MatrixBlock mbTargetIndex = _currBlk._2()._2();
					
					long valix = (long) mbTargetIndex.get(_currPos, 0);
					long rix = UtilFunctions.computeBlockIndex(valix, _blen);
					int pos = UtilFunctions.computeCellInBlock(valix, _blen);
					int len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					MatrixIndexes lix = new MatrixIndexes(rix,ixmap.getColumnIndex());
					MatrixBlock tmp = data.slice(_currPos, _currPos);
					_currPos++;
					
					//handle end of block situations
					if( _currPos == data.getNumRows() ){
						_currBlk = null;
					}
					
					return new Tuple2<>(lix, new RowMatrixBlock(len, pos, tmp));
				}
				catch(Exception ex) {
					throw new RuntimeException(ex);
				}
			}

			@Override
			public void remove() {
				throw new RuntimeException("Unsupported remove operation.");
			}
		}
	}
	
	private static class ShuffleMatrixBlockRowsInMemFunction implements PairFlatMapFunction<Iterator<Tuple2<MatrixIndexes,MatrixBlock>>,MatrixIndexes,RowMatrixBlock> 
	{	
		private static final long serialVersionUID = 6885207719329119646L; 
		
		private long _rlen = -1;
		private int _blen = -1;

		private Broadcast<PartitionedBlock<MatrixBlock>> _pmb = null;
		
		public ShuffleMatrixBlockRowsInMemFunction(long rlen, int blen, Broadcast<PartitionedBlock<MatrixBlock>> pmb)
		{
			_rlen = rlen;
			_blen = blen;
			_pmb = pmb;
		}

		@Override
		public ShuffleMatrixIterator call(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> arg0)
			throws Exception 
		{
			return new ShuffleMatrixIterator(arg0);
		}
		
		/**
		 * Lazy iterator to prevent blk output for better resource efficiency; 
		 * This also lowered garbage collection overhead.
		 */
		private class ShuffleMatrixIterator implements Iterable<Tuple2<MatrixIndexes, RowMatrixBlock>>, Iterator<Tuple2<MatrixIndexes, RowMatrixBlock>>
		{
			private Iterator<Tuple2<MatrixIndexes, MatrixBlock>> _inIter = null;
			private Tuple2<MatrixIndexes, MatrixBlock> _currBlk = null;
			private int _currPos = -1;
			
			public ShuffleMatrixIterator(Iterator<Tuple2<MatrixIndexes, MatrixBlock>> in) {
				_inIter = in;
			}

			@Override
			public Iterator<Tuple2<MatrixIndexes, RowMatrixBlock>> iterator() {
				return this;
			}

			@Override
			public boolean hasNext() {
				return _currBlk != null || _inIter.hasNext();
			}
			
			@Override
			public Tuple2<MatrixIndexes, RowMatrixBlock> next() 
			{
				//pull next input block (if required)
				if( _currBlk == null ){
					_currBlk = _inIter.next();
					_currPos = 0;
				}
				
				try
				{
					//produce next output tuple
					MatrixIndexes ixmap = _currBlk._1();
					MatrixBlock data = _currBlk._2();
					MatrixBlock mbTargetIndex = _pmb.value().getBlock((int)ixmap.getRowIndex(), 1);
					
					long valix = (long) mbTargetIndex.get(_currPos, 0);
					long rix = UtilFunctions.computeBlockIndex(valix, _blen);
					int pos = UtilFunctions.computeCellInBlock(valix, _blen);
					int len = UtilFunctions.computeBlockSize(_rlen, rix, _blen);
					MatrixIndexes lix = new MatrixIndexes(rix,ixmap.getColumnIndex());
					MatrixBlock tmp = data.slice(_currPos, _currPos);
					_currPos++;
					
					//handle end of block situations
					if( _currPos == data.getNumRows() ){
						_currBlk = null;
					}
					
					return new Tuple2<>(lix, new RowMatrixBlock(len, pos, tmp));
				}
				catch(Exception ex) {
					throw new RuntimeException(ex);
				}
			}

			@Override
			public void remove() {
				throw new RuntimeException("Unsupported remove operation.");
			}
		}
	}
	
	/**
	 * More memory-efficient representation than {@code Tuple2<Double,Double>} which requires
	 * three instead of one object per cell.
	 */
	private static class DoublePair implements Serializable
	{
		private static final long serialVersionUID = 4373356163734559009L;
		
		public double val1;
		public double val2;
		
		public DoublePair(double d1, double d2) {
			val1 = d1;
			val2 = d2;
		}
	}

	private static class ValueIndexPair implements Serializable 
	{
		private static final long serialVersionUID = -3273385845538526829L;
		
		public double val; 
		public long ix; 

		public ValueIndexPair(double dval, long lix) {
			val = dval;
			ix = lix;
		}
	}
	
	private static class ValuesIndexPair implements Serializable 
	{
		private static final long serialVersionUID = 4297433409147784971L;
		
		public double[] vals;
		public long ix; 

		public ValuesIndexPair(double[] dvals, long lix) {
			vals = dvals;
			ix = lix;
		}
	}
	
	public static class IndexComparator implements Comparator<ValueIndexPair>, Serializable 
	{
		private static final long serialVersionUID = 5154839870549241343L;
		
		private boolean _asc;
		public IndexComparator(boolean asc) {
			_asc = asc;
		}
			
		@Override
		public int compare(ValueIndexPair o1, ValueIndexPair o2) {
			int retVal = Double.compare(o1.val, o2.val);
			if(retVal != 0)
				return (_asc ? retVal : -1*retVal);
			else //for stable sort
				return Long.compare(o1.ix, o2.ix);
		}
	}
	
	public static class IndexComparator2 implements Comparator<ValuesIndexPair>, Serializable 
	{
		private static final long serialVersionUID = 5531987863790922691L;
		
		private boolean _asc;
		public IndexComparator2(boolean asc) {
			_asc = asc;
		}
		
		@Override
		public int compare(ValuesIndexPair o1, ValuesIndexPair o2) 
		{
			int retVal = SortUtils.compare(o1.vals, o2.vals);
			if(retVal != 0)
				return (_asc ? retVal : -1*retVal);
			else //for stable sort
				return Long.compare(o1.ix, o2.ix);
		}
		
	}
}

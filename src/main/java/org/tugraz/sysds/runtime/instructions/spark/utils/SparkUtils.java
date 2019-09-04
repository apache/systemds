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


package org.tugraz.sysds.runtime.instructions.spark.utils;

import org.apache.spark.HashPartitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.storage.StorageLevel;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.Checkpoint;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.data.IndexedTensorBlock;
import org.tugraz.sysds.runtime.data.BasicTensorBlock;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyBinaryCellFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyMatrixBlockFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyMatrixBlockPairFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyTensorBlockFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyTensorBlockPairFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.FilterNonEmptyBlocksFunction;
import org.tugraz.sysds.runtime.instructions.spark.functions.RecomputeNnzFunction;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixCell;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.Pair;
import org.tugraz.sysds.runtime.matrix.mapred.IndexedMatrixValue;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import scala.Tuple2;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class SparkUtils 
{	
	//internal configuration
	public static final StorageLevel DEFAULT_TMP = Checkpoint.DEFAULT_STORAGE_LEVEL;

	public static IndexedMatrixValue toIndexedMatrixBlock( Tuple2<MatrixIndexes,MatrixBlock> in ) {
		return new IndexedMatrixValue(in._1(), in._2());
	}

	public static IndexedMatrixValue toIndexedMatrixBlock( MatrixIndexes ix, MatrixBlock mb ) {
		return new IndexedMatrixValue(ix, mb);
	}

	public static IndexedTensorBlock toIndexedTensorBlock( Tuple2<TensorIndexes, TensorBlock> in ) {
		return new IndexedTensorBlock(in._1(), in._2());
	}

	public static IndexedTensorBlock toIndexedTensorBlock(TensorIndexes ix, TensorBlock mb ) {
		return new IndexedTensorBlock(ix, mb);
	}

	public static Tuple2<MatrixIndexes,MatrixBlock> fromIndexedMatrixBlock( IndexedMatrixValue in ){
		return new Tuple2<>(in.getIndexes(), (MatrixBlock)in.getValue());
	}

	public static List<Tuple2<MatrixIndexes,MatrixBlock>> fromIndexedMatrixBlock( List<IndexedMatrixValue> in ) {
		return in.stream().map(imv -> fromIndexedMatrixBlock(imv)).collect(Collectors.toList());
	}

	public static Pair<MatrixIndexes,MatrixBlock> fromIndexedMatrixBlockToPair( IndexedMatrixValue in ){
		return new Pair<>(in.getIndexes(), (MatrixBlock)in.getValue());
	}

	public static List<Pair<MatrixIndexes,MatrixBlock>> fromIndexedMatrixBlockToPair( List<IndexedMatrixValue> in ) {
		return in.stream().map(imv -> fromIndexedMatrixBlockToPair(imv)).collect(Collectors.toList());
	}

	public static Tuple2<Long,FrameBlock> fromIndexedFrameBlock( Pair<Long, FrameBlock> in ){
		return new Tuple2<>(in.getKey(), in.getValue());
	}

	public static List<Tuple2<Long,FrameBlock>> fromIndexedFrameBlock(List<Pair<Long, FrameBlock>> in) {
		return in.stream().map(ifv -> fromIndexedFrameBlock(ifv)).collect(Collectors.toList());
	}

	public static List<Pair<Long,Long>> toIndexedLong( List<Tuple2<Long, Long>> in ) {
		return in.stream().map(e -> new Pair<>(e._1(), e._2())).collect(Collectors.toList());
	}

	public static Pair<Long,FrameBlock> toIndexedFrameBlock( Tuple2<Long,FrameBlock> in ) {
		return new Pair<>(in._1(), in._2());
	}

	/**
	 * Indicates if the input RDD is hash partitioned, i.e., it has a partitioner
	 * of type {@code org.apache.spark.HashPartitioner}.
	 * 
	 * @param in input JavaPairRDD
	 * @return true if input is hash partitioned
	 */
	public static boolean isHashPartitioned(JavaPairRDD<?,?> in) {
		return !in.rdd().partitioner().isEmpty()
			&& in.rdd().partitioner().get() instanceof HashPartitioner;
	}
	
	public static int getNumPreferredPartitions(DataCharacteristics dc, JavaPairRDD<?,?> in) {
		if( !dc.dimsKnown(true) && in != null )
			return in.getNumPartitions();
		return getNumPreferredPartitions(dc);
	}
	
	public static int getNumPreferredPartitions(DataCharacteristics dc) {
		if( !dc.dimsKnown() )
			return SparkExecutionContext.getDefaultParallelism(true);
		double hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
		double matrixPSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(dc);
		return (int) Math.max(Math.ceil(matrixPSize/hdfsBlockSize), 1);
	}
	
	/**
	 * Creates a partitioning-preserving deep copy of the input matrix RDD, where 
	 * the indexes and values are copied.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes,MatrixBlock>}
	 * @return matrix as {@code JavaPairRDD<MatrixIndexes,MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes,MatrixBlock> copyBinaryBlockMatrix(
			JavaPairRDD<MatrixIndexes,MatrixBlock> in) {
		return copyBinaryBlockMatrix(in, true);
	}
	
	/**
	 * Creates a partitioning-preserving copy of the input matrix RDD. If a deep copy is 
	 * requested, indexes and values are copied, otherwise they are simply passed through.
	 * 
	 * @param in matrix as {@code JavaPairRDD<MatrixIndexes,MatrixBlock>}
	 * @param deep if true, perform deep copy
	 * @return matrix as {@code JavaPairRDD<MatrixIndexes,MatrixBlock>}
	 */
	public static JavaPairRDD<MatrixIndexes,MatrixBlock> copyBinaryBlockMatrix(
			JavaPairRDD<MatrixIndexes,MatrixBlock> in, boolean deep) 
	{
		if( !deep ) //pass through of indexes and blocks
			return in.mapValues(new CopyMatrixBlockFunction(false));
		else //requires key access, so use mappartitions
			return in.mapPartitionsToPair(new CopyMatrixBlockPairFunction(deep), true);
	}

	/**
	 * Creates a partitioning-preserving deep copy of the input tensor RDD, where
	 * the indexes and values are copied.
	 *
	 * @param in tensor as {@code JavaPairRDD<TensorIndexes,HomogTensor>}
	 * @return tensor as {@code JavaPairRDD<TensorIndexes,HomogTensor>}
	 */
	public static JavaPairRDD<TensorIndexes, BasicTensorBlock> copyBinaryBlockTensor(
			JavaPairRDD<TensorIndexes, BasicTensorBlock> in) {
		return copyBinaryBlockTensor(in, true);
	}

	/**
	 * Creates a partitioning-preserving copy of the input tensor RDD. If a deep copy is
	 * requested, indexes and values are copied, otherwise they are simply passed through.
	 *
	 * @param in   tensor as {@code JavaPairRDD<TensorIndexes,HomogTensor>}
	 * @param deep if true, perform deep copy
	 * @return tensor as {@code JavaPairRDD<TensorIndexes,HomogTensor>}
	 */
	public static JavaPairRDD<TensorIndexes, BasicTensorBlock> copyBinaryBlockTensor(
			JavaPairRDD<TensorIndexes, BasicTensorBlock> in, boolean deep) {
		if (!deep) //pass through of indexes and blocks
			return in.mapValues(new CopyTensorBlockFunction(false));
		else //requires key access, so use mappartitions
			return in.mapPartitionsToPair(new CopyTensorBlockPairFunction(deep), true);
	}

	// This returns RDD with identifier as well as location
	public static String getStartLineFromSparkDebugInfo(String line) {
		// To remove: (2)  -- Assumption: At max, 9 RDDs as input to transformation/action
		String withoutPrefix = line.substring(4, line.length());
		// To remove: [Disk Memory Deserialized 1x Replicated]
		return  withoutPrefix.split(":")[0]; // Return 'MapPartitionsRDD[51] at mapToPair at ReorgSPInstruction.java'
	}
	
	public static String getPrefixFromSparkDebugInfo(String line) {
		String [] lines = line.split("\\||\\+-");
		String retVal = lines[0];
		for(int i = 1; i < lines.length-1; i++) {
			retVal += "|" + lines[i];
		}
		String twoSpaces = "  ";
		if(line.contains("+-"))
			return retVal + "+- ";
		else
			return retVal + "|" + twoSpaces;
	}

	/**
	 * Creates an RDD of empty blocks according to the given matrix characteristics. This is
	 * done in a scalable manner by parallelizing block ranges and generating empty blocks
	 * in a distributed manner, under awareness of preferred output partition sizes.
	 * 
	 * @param sc spark context
	 * @param mc matrix characteristics
	 * @return pair rdd of empty matrix blocks 
	 */
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> getEmptyBlockRDD( JavaSparkContext sc, DataCharacteristics mc )
	{
		//compute degree of parallelism and block ranges
		long size = mc.getNumBlocks() * OptimizerUtils.estimateSizeEmptyBlock(Math.min(
				Math.max(mc.getRows(),1), mc.getBlocksize()), Math.min(Math.max(mc.getCols(),1), mc.getBlocksize()));
		int par = (int) Math.min(4*Math.max(SparkExecutionContext.getDefaultParallelism(true),
				Math.ceil(size/InfrastructureAnalyzer.getHDFSBlockSize())), mc.getNumBlocks());
		long pNumBlocks = (long)Math.ceil((double)mc.getNumBlocks()/par);
		
		//generate block offsets per partition
		List<Long> offsets = LongStream.iterate(0, n -> n+pNumBlocks)
				.limit(par).boxed().collect(Collectors.toList());
		
		//parallelize offsets and generate all empty blocks
		return (JavaPairRDD<MatrixIndexes,MatrixBlock>) sc.parallelize(offsets, par)
				.flatMapToPair(new GenerateEmptyBlocks(mc, pNumBlocks));
	}

	public static JavaPairRDD<MatrixIndexes, MatrixCell> cacheBinaryCellRDD(JavaPairRDD<MatrixIndexes, MatrixCell> input) {
		return !input.getStorageLevel().equals(DEFAULT_TMP) ? 
			input.mapToPair(new CopyBinaryCellFunction()).persist(DEFAULT_TMP) : input;
	}

	/**
	 * Utility to compute dimensions and non-zeros in a given RDD of binary cells.
	 * 
	 * @param input matrix as {@code JavaPairRDD<MatrixIndexes, MatrixCell>}
	 * @return matrix characteristics
	 */
	public static DataCharacteristics computeDataCharacteristics(JavaPairRDD<MatrixIndexes, MatrixCell> input)
	{
		// compute dimensions and nnz in single pass
		DataCharacteristics ret = input
				.map(new AnalyzeCellDataCharacteristics())
				.reduce(new AggregateDataCharacteristics());
		
		return ret;
	}
	
	public static long getNonZeros(JavaPairRDD<MatrixIndexes, MatrixBlock> input) {
		//note: avoid direct lambda expression due reduce unnecessary GC overhead
		return input.filter(new FilterNonEmptyBlocksFunction())
			.values().mapPartitions(new RecomputeNnzFunction()).reduce((a,b)->a+b);
	}

	private static class AnalyzeCellDataCharacteristics implements Function<Tuple2<MatrixIndexes,MatrixCell>, DataCharacteristics>
	{
		private static final long serialVersionUID = 8899395272683723008L;

		@Override
		public DataCharacteristics call(Tuple2<MatrixIndexes, MatrixCell> arg0) throws Exception {
			long rix = arg0._1().getRowIndex();
			long cix = arg0._1().getColumnIndex();
			long nnz = (arg0._2().getValue()!=0) ? 1 : 0;
			return new MatrixCharacteristics(rix, cix, 0, nnz);
		}
	}

	private static class AggregateDataCharacteristics implements Function2<DataCharacteristics, DataCharacteristics, DataCharacteristics>
	{
		private static final long serialVersionUID = 4263886749699779994L;

		@Override
		public DataCharacteristics call(DataCharacteristics arg0, DataCharacteristics arg1) throws Exception {
			return new MatrixCharacteristics(
				Math.max(arg0.getRows(), arg1.getRows()),  //max
				Math.max(arg0.getCols(), arg1.getCols()),  //max
				arg0.getBlocksize(), 
				arg0.getNonZeros() + arg1.getNonZeros() ); //sum
		}
	}
	
	private static class GenerateEmptyBlocks implements PairFlatMapFunction<Long, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 630129586089106855L;

		private final DataCharacteristics _mc;
		private final long _pNumBlocks;
		
		public GenerateEmptyBlocks(DataCharacteristics mc, long pNumBlocks) {
			_mc = mc;
			_pNumBlocks = pNumBlocks;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Long arg0) throws Exception {
			//NOTE: for cases of a very large number of empty blocks per partition
			//(e.g., >3M for 128MB partitions), it is important for low GC overhead 
			//not to materialized these objects but return a lazy iterator instead.
			long ncblks = _mc.getNumColBlocks();
			long nblocksU = Math.min(arg0+_pNumBlocks, _mc.getNumBlocks());
			return LongStream.range(arg0, nblocksU).mapToObj(i -> {
				long rix = 1 + i / ncblks;
				long cix = 1 + i % ncblks;
				int lrlen = UtilFunctions.computeBlockSize(_mc.getRows(), rix, _mc.getBlocksize());
				int lclen = UtilFunctions.computeBlockSize(_mc.getCols(), cix, _mc.getBlocksize());
				return new Tuple2<>(new MatrixIndexes(rix,cix), new MatrixBlock(lrlen, lclen, true));
			}).iterator();
		}
	}
}

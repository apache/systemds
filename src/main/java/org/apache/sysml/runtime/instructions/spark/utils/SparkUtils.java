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

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import org.apache.spark.HashPartitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.storage.StorageLevel;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.Checkpoint;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBinaryCellFunction;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBlockFunction;
import org.apache.sysml.runtime.instructions.spark.functions.CopyBlockPairFunction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.util.UtilFunctions;

import scala.Tuple2;

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

	public static Tuple2<MatrixIndexes,MatrixBlock> fromIndexedMatrixBlock( IndexedMatrixValue in ){
		return new Tuple2<MatrixIndexes,MatrixBlock>(in.getIndexes(), (MatrixBlock)in.getValue());
	}

	public static ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> fromIndexedMatrixBlock( ArrayList<IndexedMatrixValue> in ) {
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		for( IndexedMatrixValue imv : in )
			ret.add(fromIndexedMatrixBlock(imv));
		return ret;
	}

	public static Pair<MatrixIndexes,MatrixBlock> fromIndexedMatrixBlockToPair( IndexedMatrixValue in ){
		return new Pair<MatrixIndexes,MatrixBlock>(in.getIndexes(), (MatrixBlock)in.getValue());
	}

	public static ArrayList<Pair<MatrixIndexes,MatrixBlock>> fromIndexedMatrixBlockToPair( ArrayList<IndexedMatrixValue> in ) {
		ArrayList<Pair<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Pair<MatrixIndexes,MatrixBlock>>();
		for( IndexedMatrixValue imv : in )
			ret.add(fromIndexedMatrixBlockToPair(imv));
		return ret;
	}

	public static Tuple2<Long,FrameBlock> fromIndexedFrameBlock( Pair<Long, FrameBlock> in ){
		return new Tuple2<Long, FrameBlock>(in.getKey(), in.getValue());
	}

	public static ArrayList<Tuple2<Long,FrameBlock>> fromIndexedFrameBlock( ArrayList<Pair<Long, FrameBlock>> in ) {
		ArrayList<Tuple2<Long, FrameBlock>> ret = new ArrayList<Tuple2<Long, FrameBlock>>();
		for( Pair<Long, FrameBlock> ifv : in )
			ret.add(fromIndexedFrameBlock(ifv));
		return ret;
	}

	public static ArrayList<Pair<Long,Long>> toIndexedLong( List<Tuple2<Long, Long>> in ) {
		ArrayList<Pair<Long, Long>> ret = new ArrayList<Pair<Long, Long>>();
		for( Tuple2<Long, Long> e : in )
			ret.add(new Pair<Long,Long>(e._1(), e._2()));
		return ret;
	}

	public static Pair<Long,FrameBlock> toIndexedFrameBlock( Tuple2<Long,FrameBlock> in ) {
		return new Pair<Long,FrameBlock>(in._1(), in._2());
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
	
	public static int getNumPreferredPartitions(MatrixCharacteristics mc, JavaPairRDD<?,?> in) {
		if( !mc.dimsKnown(true) && in != null )
			return in.getNumPartitions();
		return getNumPreferredPartitions(mc);
	}
	
	public static int getNumPreferredPartitions(MatrixCharacteristics mc) {
		if( !mc.dimsKnown() )
			return SparkExecutionContext.getDefaultParallelism(true);
		double hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
		double matrixPSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(mc);
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
			return in.mapValues(new CopyBlockFunction(false));
		else //requires key access, so use mappartitions
			return in.mapPartitionsToPair(new CopyBlockPairFunction(deep), true);
	}

	// This returns RDD with identifier as well as location
	public static String getStartLineFromSparkDebugInfo(String line) throws DMLRuntimeException {
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
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> getEmptyBlockRDD( JavaSparkContext sc, MatrixCharacteristics mc )
	{
		//compute degree of parallelism and block ranges
		long size = mc.getNumBlocks() * OptimizerUtils.estimateSizeEmptyBlock(Math.max(
				mc.getRows(), mc.getRowsPerBlock()), Math.max(mc.getCols(), mc.getColsPerBlock()));
		int par = (int) Math.min(Math.max(SparkExecutionContext.getDefaultParallelism(true),
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
	public static MatrixCharacteristics computeMatrixCharacteristics(JavaPairRDD<MatrixIndexes, MatrixCell> input) 
	{
		// compute dimensions and nnz in single pass
		MatrixCharacteristics ret = input
				.map(new AnalyzeCellMatrixCharacteristics())
				.reduce(new AggregateMatrixCharacteristics());
		
		return ret;
	}

	private static class AnalyzeCellMatrixCharacteristics implements Function<Tuple2<MatrixIndexes,MatrixCell>, MatrixCharacteristics> 
	{
		private static final long serialVersionUID = 8899395272683723008L;

		@Override
		public MatrixCharacteristics call(Tuple2<MatrixIndexes, MatrixCell> arg0) 
			throws Exception 
		{
			long rix = arg0._1().getRowIndex();
			long cix = arg0._1().getColumnIndex();
			long nnz = (arg0._2().getValue()!=0) ? 1 : 0;
			return new MatrixCharacteristics(rix, cix, 0, 0, nnz);
		}
	}

	private static class AggregateMatrixCharacteristics implements Function2<MatrixCharacteristics, MatrixCharacteristics, MatrixCharacteristics> 
	{
		private static final long serialVersionUID = 4263886749699779994L;

		@Override
		public MatrixCharacteristics call(MatrixCharacteristics arg0, MatrixCharacteristics arg1) 
			throws Exception 
		{
			return new MatrixCharacteristics(
					Math.max(arg0.getRows(), arg1.getRows()),  //max
					Math.max(arg0.getCols(), arg1.getCols()),  //max
					arg0.getRowsPerBlock(), 
					arg0.getColsPerBlock(),
					arg0.getNonZeros() + arg1.getNonZeros() ); //sum
		}	
	}
	
	private static class GenerateEmptyBlocks implements PairFlatMapFunction<Long, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 630129586089106855L;

		private final MatrixCharacteristics _mc;
		private final long _pNumBlocks;
		
		public GenerateEmptyBlocks(MatrixCharacteristics mc, long pNumBlocks) {
			_mc = mc;
			_pNumBlocks = pNumBlocks;
		}
		
		@Override
		public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> call(Long arg0) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> list = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			long ncblks = _mc.getNumColBlocks();
			long nblocksU = Math.min(arg0+_pNumBlocks, _mc.getNumBlocks());
			for( long i=arg0; i<nblocksU; i++ ) {
				long rix = 1 + i / ncblks;
				long cix = 1 + i % ncblks;
				int lrlen = UtilFunctions.computeBlockSize(_mc.getRows(), rix, _mc.getRowsPerBlock());
				int lclen = UtilFunctions.computeBlockSize(_mc.getCols(), cix, _mc.getColsPerBlock());
				list.add(new Tuple2<MatrixIndexes,MatrixBlock>(
						new MatrixIndexes(rix,cix), 
						new MatrixBlock(lrlen, lclen, true)));
			}
			return list.iterator();
		}
	}
}

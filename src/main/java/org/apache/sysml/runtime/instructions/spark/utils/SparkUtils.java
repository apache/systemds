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
import java.util.List;

import org.apache.spark.HashPartitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.storage.StorageLevel;
import org.apache.sysml.lops.Checkpoint;
import org.apache.sysml.runtime.DMLRuntimeException;
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

	public static ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> fromIndexedMatrixBlock( ArrayList<IndexedMatrixValue> in )
	{
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		for( IndexedMatrixValue imv : in )
			ret.add(fromIndexedMatrixBlock(imv));
		
		return ret;
	}

	public static Pair<MatrixIndexes,MatrixBlock> fromIndexedMatrixBlockToPair( IndexedMatrixValue in ){
		return new Pair<MatrixIndexes,MatrixBlock>(in.getIndexes(), (MatrixBlock)in.getValue());
	}

	public static ArrayList<Pair<MatrixIndexes,MatrixBlock>> fromIndexedMatrixBlockToPair( ArrayList<IndexedMatrixValue> in )
	{
		ArrayList<Pair<MatrixIndexes,MatrixBlock>> ret = new ArrayList<Pair<MatrixIndexes,MatrixBlock>>();
		for( IndexedMatrixValue imv : in )
			ret.add(fromIndexedMatrixBlockToPair(imv));
		
		return ret;
	}

	public static Tuple2<Long,FrameBlock> fromIndexedFrameBlock( Pair<Long, FrameBlock> in ){
		return new Tuple2<Long, FrameBlock>(in.getKey(), in.getValue());
	}

	public static ArrayList<Tuple2<Long,FrameBlock>> fromIndexedFrameBlock( ArrayList<Pair<Long, FrameBlock>> in )
	{
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

	public static JavaPairRDD<MatrixIndexes, MatrixBlock> getEmptyBlockRDD( JavaSparkContext sc, MatrixCharacteristics mc )
	{
		//create all empty blocks
		ArrayList<Tuple2<MatrixIndexes,MatrixBlock>> list = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
		int nrblks = (int)Math.ceil((double)mc.getRows()/mc.getRowsPerBlock());
		int ncblks = (int)Math.ceil((double)mc.getCols()/mc.getColsPerBlock());
		for(long r=1; r<=nrblks; r++)
			for(long c=1; c<=ncblks; c++)
			{
				int lrlen = UtilFunctions.computeBlockSize(mc.getRows(), r, mc.getRowsPerBlock());
				int lclen = UtilFunctions.computeBlockSize(mc.getCols(), c, mc.getColsPerBlock());
				MatrixIndexes ix = new MatrixIndexes(r, c);
				MatrixBlock mb = new MatrixBlock(lrlen, lclen, true);
				list.add(new Tuple2<MatrixIndexes,MatrixBlock>(ix,mb));
			}
		
		//create rdd of in-memory list
		return sc.parallelizePairs(list);
	}

	public static JavaPairRDD<MatrixIndexes, MatrixCell> cacheBinaryCellRDD(JavaPairRDD<MatrixIndexes, MatrixCell> input)
	{
		JavaPairRDD<MatrixIndexes, MatrixCell> ret = null;
		
		if( !input.getStorageLevel().equals(DEFAULT_TMP) ) {
			ret = input.mapToPair(new CopyBinaryCellFunction())
					   .persist(DEFAULT_TMP);
		}
		
		return ret;
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

	/**
	 * Utility to compute number of non-zeros from the given RDD of MatrixBlocks
	 * 
	 * @param rdd matrix as {@code JavaPairRDD<MatrixIndexes, MatrixBlock>}
	 * @return number of non-zeros
	 */
	public static long computeNNZFromBlocks(JavaPairRDD<MatrixIndexes, MatrixBlock> rdd) {
		long nnz = rdd.values().aggregate(	0L, 
						new Function2<Long,MatrixBlock,Long>() {
							private static final long serialVersionUID = 4907645080949985267L;
							@Override
							public Long call(Long v1, MatrixBlock v2) throws Exception {
								return (v1 + v2.getNonZeros());
							} 
						}, 
						new Function2<Long,Long,Long>() {
							private static final long serialVersionUID = 333028431986883739L;
							@Override
							public Long call(Long v1, Long v2) throws Exception {
								return v1+v2;
							}
						} );
		return nnz;
	}
}

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

import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.PrimitiveIterator;
import java.util.Random;
import java.util.stream.LongStream;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.random.Well1024a;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.random.SamplingUtils;

import scala.Tuple2;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.DataGen;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.utils.Statistics;

public class RandSPInstruction extends UnarySPInstruction {
	// internal configuration
	private static final long INMEMORY_NUMBLOCKS_THRESHOLD = 1024 * 1024;

	private DataGenMethod method = DataGenMethod.INVALID;
	private final CPOperand rows, cols;
	private final int rowsInBlock, colsInBlock;
	private final double minValue, maxValue;
	private final double sparsity;
	private final String pdf, pdfParams;
	private long seed = 0;
	private final String dir;
	private final CPOperand seq_from, seq_to, seq_incr;

	// sample specific attributes
	private final boolean replace;

	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, 
			CPOperand rows, CPOperand cols, int rpb, int cpb, double minValue, double maxValue, double sparsity, long seed,
			String dir, String probabilityDensityFunction, String pdfParams,
			CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, boolean replace, String opcode, String istr) {
		super(op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.minValue = minValue;
		this.maxValue = maxValue;
		this.sparsity = sparsity;
		this.seed = seed;
		this.dir = dir;
		this.pdf = probabilityDensityFunction;
		this.pdfParams = pdfParams;
		this.seq_from = seqFrom;
		this.seq_to = seqTo;
		this.seq_incr = seqIncr;
		this.replace = replace;
	}
	
	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			int rpb, int cpb, double minValue, double maxValue, double sparsity, long seed, String dir,
			String probabilityDensityFunction, String pdfParams, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity,
			seed, dir, probabilityDensityFunction, pdfParams, null, null, null, false, opcode, istr);
	}

	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			int rpb, int cpb, CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, rpb, cpb, -1, -1, -1,
			-1, null, null, null, seqFrom, seqTo, seqIncr, false, opcode, istr);
	}

	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
			int rpb, int cpb, double maxValue, boolean replace, long seed, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, rpb, cpb, -1, maxValue, -1,
			seed, null, null, null, null, null, null, replace, opcode, istr);
	}

	public long getRows() {
		return rows.isLiteral() ? Long.parseLong(rows.getName()) : -1;
	}

	public long getCols() {
		return cols.isLiteral() ? Long.parseLong(cols.getName()) : -1;
	}

	public int getRowsInBlock() {
		return rowsInBlock;
	}

	public int getColsInBlock() {
		return colsInBlock;
	}

	public double getMinValue() {
		return minValue;
	}

	public double getMaxValue() {
		return maxValue;
	}

	public double getSparsity() {
		return sparsity;
	}

	public static RandSPInstruction parseInstruction(String str) 
		throws DMLRuntimeException 
	{
		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = s[0];
		
		DataGenMethod method = DataGenMethod.INVALID;
		if ( opcode.equalsIgnoreCase(DataGen.RAND_OPCODE) ) {
			method = DataGenMethod.RAND;
			InstructionUtils.checkNumFields ( str, 12 );
		}
		else if ( opcode.equalsIgnoreCase(DataGen.SEQ_OPCODE) ) {
			method = DataGenMethod.SEQ;
			// 8 operands: rows, cols, rpb, cpb, from, to, incr, outvar
			InstructionUtils.checkNumFields ( str, 8 ); 
		}
		else if ( opcode.equalsIgnoreCase(DataGen.SAMPLE_OPCODE) ) {
			method = DataGenMethod.SAMPLE;
			// 7 operands: range, size, replace, seed, rpb, cpb, outvar
			InstructionUtils.checkNumFields ( str, 7 ); 
		}
		
		Operator op = null;
		// output is specified by the last operand
		CPOperand out = new CPOperand(s[s.length-1]); 

		if ( method == DataGenMethod.RAND ) {
			CPOperand rows = new CPOperand(s[1]);
			CPOperand cols = new CPOperand(s[2]);
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			double minValue = !s[5].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[5]).doubleValue() : -1;
			double maxValue = !s[6].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[6]).doubleValue() : -1;
			double sparsity = !s[7].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[7]).doubleValue() : -1;
			long seed = !s[8].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Long.valueOf(s[8]).longValue() : -1;
			String dir = s[9];
			String pdf = s[10];
			String pdfParams = !s[11].contains( Lop.VARIABLE_NAME_PLACEHOLDER) ?
				s[11] : null;
			
			return new RandSPInstruction(op, method, null, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, dir, pdf, pdfParams, opcode, str);
		}
		else if ( method == DataGenMethod.SEQ) {
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			CPOperand from = new CPOperand(s[5]);
			CPOperand to = new CPOperand(s[6]);
			CPOperand incr = new CPOperand(s[7]);
			
			CPOperand in = null;
			return new RandSPInstruction(op, method, in, out, null, null, rpb, cpb, from, to, incr, opcode, str);
		}
		else if ( method == DataGenMethod.SAMPLE) 
		{
			double max = !s[1].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[1]) : 0;
			CPOperand rows = new CPOperand(s[2]);
			CPOperand cols = new CPOperand("1", ValueType.INT, DataType.SCALAR);
			boolean replace = (!s[3].contains(Lop.VARIABLE_NAME_PLACEHOLDER) 
				&& Boolean.valueOf(s[3]));
			
			long seed = Long.parseLong(s[4]);
			int rpb = Integer.parseInt(s[5]);
			int cpb = Integer.parseInt(s[6]);
			
			return new RandSPInstruction(op, method, null, out, rows, cols, rpb, cpb, max, replace, seed, opcode, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//process specific datagen operator
		switch( method ) {
			case RAND: generateRandData(sec); break;
			case SEQ: generateSequence(sec); break;
			case SAMPLE: generateSample(sec); break;
			default: 
				throw new DMLRuntimeException("Invalid datagen method: "+method); 
		}
	}

	private void generateRandData(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		long lrows = sec.getScalarInput(rows).getLongValue();
		long lcols = sec.getScalarInput(cols).getLongValue();
		
		//step 1: generate pseudo-random seed (because not specified) 
		long lSeed = seed; //seed per invocation
		if( lSeed == DataGenOp.UNSPECIFIED_SEED ) 
			lSeed = DataGenOp.generateRandomSeed();
		
		if( LOG.isTraceEnabled() )
			LOG.trace("Process RandSPInstruction rand with seed = "+lSeed+".");

		//step 2: potential in-memory rand operations if applicable
		if( isMemAvail(lrows, lcols, sparsity, minValue, maxValue) 
			&&  DMLScript.rtplatform != RUNTIME_PLATFORM.SPARK )
		{
			RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
				pdf, (int)lrows, (int)lcols, rowsInBlock, colsInBlock, 
				sparsity, minValue, maxValue, pdfParams);
			MatrixBlock mb = MatrixBlock.randOperations(rgen, lSeed);
			
			sec.setMatrixOutput(output.getName(), mb, getExtendedOpcode());
			Statistics.decrementNoOfExecutedSPInst();
			return;
		}
		
		//step 3: seed generation 
		JavaPairRDD<MatrixIndexes, Tuple2<Long, Long>> seedsRDD = null;
		Well1024a bigrand = LibMatrixDatagen.setupSeedsForRand(lSeed);
		LongStream nnz = LibMatrixDatagen.computeNNZperBlock(lrows, lcols, rowsInBlock, colsInBlock, sparsity);
		PrimitiveIterator.OfLong nnzIter = nnz.iterator();
		double totalSize = OptimizerUtils.estimatePartitionedSizeExactSparsity( lrows, lcols, rowsInBlock, 
			colsInBlock, sparsity); //overestimate for on disk, ensures hdfs block per partition
		double hdfsBlkSize = InfrastructureAnalyzer.getHDFSBlockSize();
		long numBlocks = new MatrixCharacteristics(lrows, lcols, rowsInBlock, colsInBlock).getNumBlocks();
		long numColBlocks = (long)Math.ceil((double)lcols/(double)colsInBlock);
		
		//a) in-memory seed rdd construction 
		if( numBlocks < INMEMORY_NUMBLOCKS_THRESHOLD )
		{
			ArrayList<Tuple2<MatrixIndexes, Tuple2<Long, Long>>> seeds = 
					new ArrayList<>();
			for( long i=0; i<numBlocks; i++ ) {
				long r = 1 + i/numColBlocks;
				long c = 1 + i%numColBlocks;
				MatrixIndexes indx = new MatrixIndexes(r, c);
				Long seedForBlock = bigrand.nextLong();
				seeds.add(new Tuple2<>(indx, new Tuple2<>(seedForBlock, nnzIter.nextLong())));
			}
			
			//for load balancing: degree of parallelism such that ~128MB per partition
			int numPartitions = (int) Math.max(Math.min(totalSize/hdfsBlkSize, numBlocks), 1);
				
			//create seeds rdd 
			seedsRDD = sec.getSparkContext().parallelizePairs(seeds, numPartitions);
		}
		//b) file-based seed rdd construction (for robustness wrt large number of blocks)
		else
		{
			Path path = new Path(LibMatrixDatagen.generateUniqueSeedPath(dir));
			PrintWriter pw = null;
			try
			{
				FileSystem fs = IOUtilFunctions.getFileSystem(path);
				pw = new PrintWriter(fs.create(path));
				StringBuilder sb = new StringBuilder();
				for( long i=0; i<numBlocks; i++ ) {
					sb.append(1 + i/numColBlocks);
					sb.append(',');
					sb.append(1 + i%numColBlocks);
					sb.append(',');
					sb.append(bigrand.nextLong());
					sb.append(',');
					sb.append(nnzIter.nextLong());
					pw.println(sb.toString());
					sb.setLength(0);
				}
			}
			catch( IOException ex ) {
				throw new DMLRuntimeException(ex);
			}
			finally {
				IOUtilFunctions.closeSilently(pw);
			}
			
			//for load balancing: degree of parallelism such that ~128MB per partition
			int numPartitions = (int) Math.max(Math.min(totalSize/hdfsBlkSize, numBlocks), 1);
			
			//create seeds rdd 
			seedsRDD = sec.getSparkContext()
					.textFile(path.toString(), numPartitions)
					.mapToPair(new ExtractSeedTuple());
		}
		
		//step 4: execute rand instruction over seed input
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = seedsRDD
				.mapToPair(new GenerateRandomBlock(lrows, lcols, rowsInBlock, colsInBlock, 
					sparsity, minValue, maxValue, pdf, pdfParams)); 
		
		//step 5: output handling
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown(true)) {
			//note: we cannot compute the nnz from sparsity because this would not reflect the 
			//actual number of non-zeros, except for extreme values of sparsity equals 0 or 1.
			long lnnz = (sparsity==0 || sparsity==1) ? (long) (sparsity*lrows*lcols) : -1;
			mcOut.set(lrows, lcols, rowsInBlock, colsInBlock, lnnz);
		}
		sec.setRDDHandleForVariable(output.getName(), out);
	}

	private void generateSequence(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		double lfrom = sec.getScalarInput(seq_from).getDoubleValue();
		double lto = sec.getScalarInput(seq_to).getDoubleValue();
		double lincr = sec.getScalarInput(seq_incr).getDoubleValue();
		
		//sanity check valid increment
		if( lincr == 0 ) {
			throw new DMLRuntimeException("ERROR: While performing seq(" + lfrom + "," + lto + "," + lincr + ")");
		}
		
		//handle default 1 to -1 for special case of from>to
		lincr = LibMatrixDatagen.updateSeqIncr(lfrom, lto, lincr);
		
		if( LOG.isTraceEnabled() )
			LOG.trace("Process RandSPInstruction seq with seqFrom="+lfrom+", seqTo="+lto+", seqIncr"+lincr);
		
		//step 1: offset generation 
		JavaRDD<Double> offsetsRDD = null;
		long nnz = UtilFunctions.getSeqLength(lfrom, lto, lincr);
		double totalSize = OptimizerUtils.estimatePartitionedSizeExactSparsity( nnz, 1, rowsInBlock, 
				colsInBlock, nnz); //overestimate for on disk, ensures hdfs block per partition
		double hdfsBlkSize = InfrastructureAnalyzer.getHDFSBlockSize();
		long numBlocks = (long)Math.ceil(((double)nnz)/rowsInBlock);
	
		//a) in-memory offset rdd construction 
		if( numBlocks < INMEMORY_NUMBLOCKS_THRESHOLD )
		{
			ArrayList<Double> offsets = new ArrayList<>();
			for( long i=0; i<numBlocks; i++ ) {
				double off = lfrom + lincr*i*rowsInBlock;
				offsets.add(off);
			}
			
			//for load balancing: degree of parallelism such that ~128MB per partition
			int numPartitions = (int) Math.max(Math.min(totalSize/hdfsBlkSize, numBlocks), 1);
			
			//create offset rdd
			offsetsRDD = sec.getSparkContext().parallelize(offsets, numPartitions);
		}
		//b) file-based offset rdd construction (for robustness wrt large number of blocks)
		else
		{
			Path path = new Path(LibMatrixDatagen.generateUniqueSeedPath(dir));
			
			PrintWriter pw = null;
			try {
				FileSystem fs = IOUtilFunctions.getFileSystem(path);
				pw = new PrintWriter(fs.create(path));
				for( long i=0; i<numBlocks; i++ ) {
					double off = lfrom + lincr*i*rowsInBlock;
					pw.println(off);
				}
			}
			catch( IOException ex ) {
				throw new DMLRuntimeException(ex);
			}
			finally {
				IOUtilFunctions.closeSilently(pw);
			}
			
			//for load balancing: degree of parallelism such that ~128MB per partition
			int numPartitions = (int) Math.max(Math.min(totalSize/hdfsBlkSize, numBlocks), 1);
			
			//create seeds rdd 
			offsetsRDD = sec.getSparkContext()
					.textFile(path.toString(), numPartitions)
					.map(new ExtractOffsetTuple());
		}
		
		//step 2: execute seq instruction over offset input
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = offsetsRDD
			.mapToPair(new GenerateSequenceBlock(rowsInBlock, lfrom, lto, lincr));

		//step 3: output handling
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			mcOut.set(nnz, 1, rowsInBlock, colsInBlock, nnz);
		}
		sec.setRDDHandleForVariable(output.getName(), out);
	}
	
	/**
	 * Helper function to construct a sample.
	 * 
	 * @param sec spark execution context
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	private void generateSample(SparkExecutionContext sec) 
		throws DMLRuntimeException 
	{
		long lrows = sec.getScalarInput(rows).getLongValue();
		if ( maxValue < lrows && !replace )
			throw new DMLRuntimeException("Sample (size=" + rows + ") larger than population (size=" + maxValue + ") can only be generated with replacement.");

		if( LOG.isTraceEnabled() )
			LOG.trace("Process RandSPInstruction sample with range="+ maxValue +", size="+ lrows +", replace="+ replace + ", seed=" + seed);
		
		// sampling rate that guarantees a sample of size >= sampleSizeLowerBound 99.99% of the time.
		double fraction = SamplingUtils.computeFractionForSampleSize((int)lrows, UtilFunctions.toLong(maxValue), replace);
		
		Well1024a bigrand = LibMatrixDatagen.setupSeedsForRand(seed);

		// divide the population range across numPartitions by creating SampleTasks
		double hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
		long outputSize = MatrixBlock.estimateSizeDenseInMemory(lrows,1);
		int numPartitions = (int) Math.ceil((double)outputSize/hdfsBlockSize);
		long partitionSize = (long) Math.ceil(maxValue/numPartitions);

		ArrayList<SampleTask> offsets = new ArrayList<>();
		long st = 1;
		while ( st <= maxValue ) {
			SampleTask s = new SampleTask();
			s.range_start = st;
			s.seed = bigrand.nextLong();
			offsets.add(s);
			st = st + partitionSize;
		}
		JavaRDD<SampleTask> offsetRDD = sec.getSparkContext().parallelize(offsets, numPartitions);
		
		// Construct the sample in a distributed manner
		JavaRDD<Double> rdd = offsetRDD.flatMap( (new GenerateSampleBlock(replace, fraction, (long)maxValue, partitionSize)) );
		
		// Randomize the sampled elements
		JavaRDD<Double> randomizedRDD = rdd.mapToPair(new AttachRandom()).sortByKey().values();
		
		// Trim the sampled list to required size & attach matrix indexes to randomized elements
		JavaPairRDD<MatrixIndexes, MatrixCell> miRDD = randomizedRDD
				.zipWithIndex()
				.filter( new TrimSample(lrows) )
				.mapToPair( new Double2MatrixCell() );
		
		MatrixCharacteristics mcOut = new MatrixCharacteristics(lrows, 1, rowsInBlock, colsInBlock, lrows);
		
		// Construct BinaryBlock representation
		JavaPairRDD<MatrixIndexes, MatrixBlock> mbRDD = 
				RDDConverterUtils.binaryCellToBinaryBlock(sec.getSparkContext(), miRDD, mcOut, true);
		
		sec.getMatrixCharacteristics(output.getName()).setNonZeros(lrows);
		sec.setRDDHandleForVariable(output.getName(), mbRDD);
	}
	
	/**
	 * Private class that defines a sampling task. 
	 * The task produces a portion of sample from range [range_start, range_start+partitionSize].
	 *
	 */
	private static class SampleTask implements Serializable 
	{
		private static final long serialVersionUID = -725284524434342939L;
		long seed;
		long range_start;
		@Override
		public String toString() { return "(" + seed + "," + range_start +")"; } 
	}
	
	/** 
	 * Main class to perform distributed sampling.
	 * 
	 * Each invocation of this FlatMapFunction produces a portion of sample 
	 * to be included in the final output. 
	 * 
	 * The input range from which the sample is constructed is given by 
	 * [range_start, range_start+partitionSize].
	 * 
	 * When replace=TRUE, the sample is produced by generating Poisson 
	 * distributed counts (denoting the number of occurrences) for each 
	 * element in the input range. 
	 * 
	 * When replace=FALSE, the sample is produced by comparing a generated 
	 * random number against the required sample fraction.
	 * 
	 * In the special case of fraction=1.0, the permutation of the input 
	 * range is computed, simply by creating RDD of elements from input range.
	 *
	 */
	private static class GenerateSampleBlock implements FlatMapFunction<SampleTask, Double> 
	{
		private static final long serialVersionUID = -8211490954143527232L;
		private double _frac;
		private boolean _replace;
		private long _maxValue, _partitionSize; 

		GenerateSampleBlock(boolean replace, double frac, long max, long psize)
		{
			_replace = replace;
			_frac = frac;
			_maxValue = max;
			_partitionSize = psize;
		}
		
		@Override
		public Iterator<Double> call(SampleTask t)
				throws Exception {

			long st = t.range_start;
			long end = Math.min(t.range_start+_partitionSize, _maxValue);
			ArrayList<Double> retList = new ArrayList<>();
			
			if ( _frac == 1.0 ) 
			{
				for(long i=st; i <= end; i++) 
					retList.add((double)i);
			}
			else 
			{
				if(_replace) 
				{
					PoissonDistribution pdist = new PoissonDistribution( (_frac > 0.0 ? _frac :1.0) );
					for(long i=st; i <= end; i++)
					{
						int count = pdist.sample();
						while(count > 0) {
							retList.add((double)i);
							count--;
						}
					}
				}
				else 
				{
					Random rnd = new Random(t.seed);
					for(long i=st; i <=end; i++) 
						if ( rnd.nextDouble() < _frac )
							retList.add((double) i);
				}
			}
			return retList.iterator();
		}
	}
	
	/**
	 * Function that filters the constructed sample contain to required number of elements.
	 *
	 */
	private static class TrimSample implements Function<Tuple2<Double,Long>, Boolean> {
		private static final long serialVersionUID = 6773370625013346530L;
		long _max;
		
		TrimSample(long max) {
			_max = max;
		}
		
		@Override
		public Boolean call(Tuple2<Double, Long> v1) throws Exception {
			return ( v1._2 < _max );
		}
		
	}
	
	/**
	 * Function to convert JavaRDD of Doubles to {@code JavaPairRDD<MatrixIndexes, MatrixCell>}
	 *
	 */
	private static class Double2MatrixCell implements PairFunction<Tuple2<Double, Long>, MatrixIndexes, MatrixCell>
	{
		private static final long serialVersionUID = -2125669746624320536L;
		
		@Override
		public Tuple2<MatrixIndexes, MatrixCell> call(Tuple2<Double, Long> t)
				throws Exception {
			long rowID = t._2()+1;
			MatrixIndexes mi = new MatrixIndexes(rowID, 1);
			MatrixCell mc = new MatrixCell(t._1());
			return new Tuple2<>(mi, mc);
		}
	}
	
	/**
	 * Pair function to attach a random number as a key to input JavaRDD.
	 * The produced JavaPairRDD is subsequently used to randomize the sampled elements. 
	 *
	 */
	private static class AttachRandom implements PairFunction<Double, Double, Double> {
		private static final long serialVersionUID = -7508858192367406554L;
		Random r = null;
		AttachRandom() {
			r = new Random();
		}
		@Override
		public Tuple2<Double, Double> call(Double t) throws Exception {
			return new Tuple2<>( r.nextDouble(), t );
		}
	}

	private static class ExtractSeedTuple implements PairFunction<String, MatrixIndexes, Tuple2<Long,Long>> {
		private static final long serialVersionUID = 3973794676854157101L;

		@Override
		public Tuple2<MatrixIndexes, Tuple2<Long, Long>> call(String arg)
				throws Exception 
		{
			String[] parts = IOUtilFunctions.split(arg, ",");
			MatrixIndexes ix = new MatrixIndexes(
					Long.parseLong(parts[0]), Long.parseLong(parts[1]));
			Tuple2<Long,Long> seed = new Tuple2<>(
					Long.parseLong(parts[2]), Long.parseLong(parts[3]));
			return new Tuple2<>(ix,seed);
		}
	}

	private static class ExtractOffsetTuple implements Function<String, Double> {
		private static final long serialVersionUID = -3980257526545002552L;

		@Override
		public Double call(String arg) throws Exception {
			return Double.parseDouble(arg);
		}
	}

	private static class GenerateRandomBlock implements PairFunction<Tuple2<MatrixIndexes, Tuple2<Long, Long> >, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 1616346120426470173L;
		
		private long _rlen; 
		private long _clen;
		private int _brlen; 
		private int _bclen; 
		private double _sparsity; 
		private double _min; 
		private double _max; 
		private String _pdf; 
		private String _pdfParams;
		
		public GenerateRandomBlock(long rlen, long clen, int brlen, int bclen, double sparsity, double min, double max, String pdf, String pdfParams) {
			_rlen = rlen;
			_clen = clen;
			_brlen = brlen;
			_bclen = bclen;
			_sparsity = sparsity;
			_min = min;
			_max = max;
			_pdf = pdf;
			_pdfParams = pdfParams;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Long, Long>> kv) 
			throws Exception 
		{
			//compute local block size: 
			MatrixIndexes ix = kv._1();
			long blockRowIndex = ix.getRowIndex();
			long blockColIndex = ix.getColumnIndex();
			int lrlen = UtilFunctions.computeBlockSize(_rlen, blockRowIndex, _brlen);
			int lclen = UtilFunctions.computeBlockSize(_clen, blockColIndex, _bclen);
			long seed = kv._2._1;
			long blockNNZ = kv._2._2;
			
			MatrixBlock blk = new MatrixBlock();
			RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
					_pdf, lrlen, lclen, lrlen, lclen,   
					_sparsity, _min, _max, _pdfParams );
			blk.randOperationsInPlace(rgen, LongStream.of(blockNNZ), null, seed);
			return new Tuple2<>(kv._1, blk);
		}
	}

	private static class GenerateSequenceBlock implements PairFunction<Double, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 5779681055705756965L;
		
		private final double _global_seq_start;
		private final double _global_seq_end;
		private final double _seq_incr;
		private final int _brlen;
		
		public GenerateSequenceBlock(int brlen, double global_seq_start, double global_seq_end, double seq_incr) {
			_global_seq_start = global_seq_start;
			_global_seq_end = global_seq_end;
			_seq_incr = seq_incr;
			_brlen = brlen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Double seq_from) 
			throws Exception 
		{
			double seq_to = (_seq_incr > 0) ?
				Math.min(_global_seq_end, seq_from + _seq_incr*(_brlen-1)) :
				Math.max(_global_seq_end, seq_from + _seq_incr*(_brlen+1));
			long globalRow = (long)Math.round((seq_from-_global_seq_start)/_seq_incr)+1;
			long rowIndex = UtilFunctions.computeBlockIndex(globalRow, _brlen);
			
			MatrixIndexes indx = new MatrixIndexes(rowIndex, 1);
			MatrixBlock blk = MatrixBlock.seqOperations(seq_from, seq_to, _seq_incr);
			return new Tuple2<>(indx, blk);
		}	
	}
	
	/**
	 * This will check if there is sufficient memory locally.
	 * 
	 * @param lRows number of rows
	 * @param lCols number of columns
	 * @param sparsity sparsity ratio
	 * @param min minimum value
	 * @param max maximum value
	 * @return
	 */
	private static boolean isMemAvail(long lrows, long lcols, double sparsity, double min, double max) {
		double size = (min == 0 && max == 0) ? OptimizerUtils.estimateSizeEmptyBlock(lrows, lcols):
			OptimizerUtils.estimateSizeExactSparsity(lrows, lcols, sparsity);
		return ( OptimizerUtils.isValidCPDimensions(lrows, lcols)
				 && OptimizerUtils.isValidCPMatrixSize(lrows, lcols, sparsity) 
				 && size < OptimizerUtils.getLocalMemBudget() );
	}	

}

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

package org.tugraz.sysds.runtime.instructions.spark;

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
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.hops.DataGenOp;
import org.tugraz.sysds.hops.Hop.DataGenMethod;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.DataGen;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.data.BasicTensor;
import org.tugraz.sysds.runtime.data.TensorIndexes;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixCell;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.meta.TensorCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import org.tugraz.sysds.utils.Statistics;
import scala.Array;
import scala.Tuple2;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

public class RandSPInstruction extends UnarySPInstruction {
	// internal configuration
	private static final long INMEMORY_NUMBLOCKS_THRESHOLD = 1024 * 1024;

	private DataGenMethod method = DataGenMethod.INVALID;
	private final CPOperand rows, cols, dims;
	private final int blocksize;
	//private boolean minMaxAreDoubles;
	private final double minValue, maxValue;
	private final String minValueStr, maxValueStr;
	private final double sparsity;
	private final String pdf, pdfParams;
	private long seed = 0;
	private final String dir;
	private final CPOperand seq_from, seq_to, seq_incr;

	// sample specific attributes
	private final boolean replace;

	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows,
		CPOperand cols, CPOperand dims, int blen, String minValue, String maxValue,
		double sparsity, long seed, String dir, String probabilityDensityFunction, String pdfParams,
		CPOperand seqFrom, CPOperand seqTo, CPOperand seqIncr, boolean replace, String opcode, String istr)
	{
		super(SPType.Rand, op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.dims = dims;
		this.blocksize = blen;
		this.minValueStr = minValue;
		this.maxValueStr = maxValue;
		double minDouble, maxDouble;
		try {
			minDouble = !minValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
					Double.valueOf(minValue) : -1;
			maxDouble = !maxValue.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
					Double.valueOf(maxValue) : -1;
			//minMaxAreDoubles = true;
		} catch (NumberFormatException e) {
			// Non double values
			if (!minValueStr.equals(maxValueStr)) {
				throw new DMLRuntimeException("Rand instruction does not support " +
						"non numeric Datatypes for range initializations.");
			}
			minDouble = -1;
			maxDouble = -1;
			//minMaxAreDoubles = false;
		}
		this.minValue = minDouble;
		this.maxValue = maxDouble;
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
	
	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows,
		CPOperand cols, CPOperand dims, int blen, String minValue, String maxValue, double sparsity, long seed,
		String dir, String probabilityDensityFunction, String pdfParams, String opcode, String istr)
	{
		this(op, mthd, in, out, rows, cols, dims, blen, minValue, maxValue, sparsity, seed, dir,
			probabilityDensityFunction, pdfParams, null, null, null, false, opcode, istr);
	}

	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows,
		CPOperand cols, CPOperand dims, int blen, CPOperand seqFrom, CPOperand seqTo,
		CPOperand seqIncr, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, "-1", "-1", -1, -1, null,
			null, null, seqFrom, seqTo, seqIncr, false, opcode, istr);
	}

	private RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, CPOperand rows, CPOperand cols,
		CPOperand dims, int blen, String maxValue, boolean replace, long seed, String opcode, String istr) {
		this(op, mthd, in, out, rows, cols, dims, blen, "-1", maxValue, -1, seed, null,
			null, null, null, null, null, replace, opcode, istr);
	}

	public long getRows() {
		return rows.isLiteral() ? Long.parseLong(rows.getName()) : -1;
	}

	public long getCols() {
		return cols.isLiteral() ? Long.parseLong(cols.getName()) : -1;
	}

	public int getBlocksize() {
		return blocksize;
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

	public static RandSPInstruction parseInstruction(String str) {
		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		String opcode = s[0];
		
		DataGenMethod method = DataGenMethod.INVALID;
		if ( opcode.equalsIgnoreCase(DataGen.RAND_OPCODE) ) {
			method = DataGenMethod.RAND;
			InstructionUtils.checkNumFields ( str, 13 );
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
			CPOperand dims = new CPOperand(s[3]);
			int blen = Integer.parseInt(s[4]);
			double sparsity = !s[8].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Double.valueOf(s[8]).doubleValue() : -1;
			long seed = !s[9].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				Long.valueOf(s[9]).longValue() : -1;
			String dir = s[10];
			String pdf = s[11];
			String pdfParams = !s[12].contains( Lop.VARIABLE_NAME_PLACEHOLDER) ?
				s[12] : null;
			
			return new RandSPInstruction(op, method, null, out, rows, cols, dims,
				blen, s[6], s[7], sparsity, seed, dir, pdf, pdfParams, opcode, str);
		}
		else if ( method == DataGenMethod.SEQ) {
			int blen = Integer.parseInt(s[3]);
			CPOperand from = new CPOperand(s[5]);
			CPOperand to = new CPOperand(s[6]);
			CPOperand incr = new CPOperand(s[7]);
			
			CPOperand in = null;
			return new RandSPInstruction(op, method, in, out, null,
				null, null, blen, from, to, incr, opcode, str);
		}
		else if ( method == DataGenMethod.SAMPLE) 
		{
			String max = !s[1].contains(Lop.VARIABLE_NAME_PLACEHOLDER) ?
				s[1] : "0";
			CPOperand rows = new CPOperand(s[2]);
			CPOperand cols = new CPOperand("1", ValueType.INT64, DataType.SCALAR);
			boolean replace = (!s[3].contains(Lop.VARIABLE_NAME_PLACEHOLDER) 
				&& Boolean.valueOf(s[3]));
			
			long seed = Long.parseLong(s[4]);
			int blen = Integer.parseInt(s[5]);
			
			return new RandSPInstruction(op, method, null, out, rows, cols,
				null, blen, max, replace, seed, opcode, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ){
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

	private void generateRandData(SparkExecutionContext sec) {
		if (output.getDataType() == DataType.MATRIX) {
			generateRandDataMatrix(sec);
		} else {
			generateRandDataTensor(sec);
		}
	}

	private void generateRandDataMatrix(SparkExecutionContext sec) {
		long lrows = sec.getScalarInput(rows).getLongValue();
		long lcols = sec.getScalarInput(cols).getLongValue();

		//step 1: generate pseudo-random seed (because not specified)
		long lSeed = seed; //seed per invocation
		if( lSeed == DataGenOp.UNSPECIFIED_SEED )
			lSeed = DataGenOp.generateRandomSeed();

		if( LOG.isTraceEnabled() )
			LOG.trace("Process RandSPInstruction rand with seed = "+lSeed+".");

		//step 2: potential in-memory rand operations if applicable
		if( ConfigurationManager.isDynamicRecompilation()
			&& isMemAvail(lrows, lcols, sparsity, minValue, maxValue)
			&& DMLScript.getGlobalExecMode() != ExecMode.SPARK )
		{
			RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(
				pdf, (int) lrows, (int) lcols, blocksize,
				sparsity, minValue, maxValue, pdfParams);
			MatrixBlock mb = MatrixBlock.randOperations(rgen, lSeed);

			sec.setMatrixOutput(output.getName(), mb);
			Statistics.decrementNoOfExecutedSPInst();
			return;
		}

		//step 3: seed generation
		JavaPairRDD<MatrixIndexes, Long> seedsRDD = null;
		Well1024a bigrand = LibMatrixDatagen.setupSeedsForRand(lSeed);
		double totalSize = OptimizerUtils.estimatePartitionedSizeExactSparsity( lrows, lcols, blocksize,
			sparsity); //overestimate for on disk, ensures hdfs block per partition
		double hdfsBlkSize = InfrastructureAnalyzer.getHDFSBlockSize();
		DataCharacteristics tmp = new MatrixCharacteristics(lrows, lcols, blocksize);
		long numBlocks = tmp.getNumBlocks();
		long numColBlocks = tmp.getNumColBlocks();

		//a) in-memory seed rdd construction
		if( numBlocks < INMEMORY_NUMBLOCKS_THRESHOLD )
		{
			ArrayList<Tuple2<MatrixIndexes, Long>> seeds = new ArrayList<>();
			for( long i=0; i<numBlocks; i++ ) {
				long r = 1 + i/numColBlocks;
				long c = 1 + i%numColBlocks;
				MatrixIndexes indx = new MatrixIndexes(r, c);
				Long seedForBlock = bigrand.nextLong();
				seeds.add(new Tuple2<>(indx, seedForBlock));
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
					.mapToPair(new ExtractMatrixSeedTuple());
		}

		//step 4: execute rand instruction over seed input
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = seedsRDD
				.mapToPair(new GenerateRandomBlock(lrows, lcols, blocksize,
					sparsity, minValue, maxValue, pdf, pdfParams));

		//step 5: output handling
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		if(!mcOut.dimsKnown(true)) {
			//note: we cannot compute the nnz from sparsity because this would not reflect the
			//actual number of non-zeros, except for extreme values of sparsity equals 0 or 1.
			long lnnz = (sparsity==0 || sparsity==1) ? (long) (sparsity*lrows*lcols) : -1;
			mcOut.set(lrows, lcols, blocksize, lnnz);
		}
		sec.setRDDHandleForVariable(output.getName(), out);
	}

	private void generateRandDataTensor(SparkExecutionContext sec) {
		int[] tDims = DataConverter.getTensorDimensions(sec, dims);

		//step 1: generate pseudo-random seed (because not specified)
		long lSeed = seed; //seed per invocation
		if( lSeed == DataGenOp.UNSPECIFIED_SEED )
			lSeed = DataGenOp.generateRandomSeed();

		if( LOG.isTraceEnabled() )
			LOG.trace("Process RandSPInstruction rand with seed = "+lSeed+".");

		//step 2: TODO potential in-memory rand operations if applicable

		//step 3: seed generation
		JavaPairRDD<TensorIndexes, Long> seedsRDD;
		Well1024a bigrand = LibMatrixDatagen.setupSeedsForRand(lSeed);
		// TODO calculate totalSize
		long[] longDims = new long[tDims.length];
		long totalSize = 1;
		long hdfsBlkSize = blocksize * tDims.length;
		for (int i = 0; i < tDims.length; i++) {
			longDims[i] = tDims[i];
			totalSize *= tDims[i];
		}
		TensorCharacteristics tmp = new TensorCharacteristics(longDims, blocksize, 0);
		long numBlocks = tmp.getNumBlocks();

		//a) in-memory seed rdd construction
		//for load balancing: degree of parallelism such that ~128MB per partition
		int numPartitions = (int) Math.max(Math.min(totalSize / hdfsBlkSize, numBlocks), 1);
		if( numBlocks < INMEMORY_NUMBLOCKS_THRESHOLD )
		{
			ArrayList<Tuple2<TensorIndexes, Long>> seeds = new ArrayList<>();
			for( long i=0; i<numBlocks; i++ ) {
				long blockIndex = i;
				long[] ix = new long[tmp.getNumDims()];
				for (int j = tmp.getNumDims() - 1; j >= 0; j--) {
					ix[j] = 1 + (blockIndex % tmp.getNumBlocks(j));
					blockIndex /= tmp.getNumBlocks(j);
				}
				TensorIndexes indx = new TensorIndexes(ix);
				Long seedForBlock = bigrand.nextLong();
				seeds.add(new Tuple2<>(indx, seedForBlock));
			}

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
					long blockIndex = i;
					for (int j = tmp.getNumDims() - 1; j >= 0; j--) {
						sb.append(1 + blockIndex % tmp.getNumBlocks(j)).append(',');
						blockIndex /= tmp.getNumBlocks(j);
					}
					sb.append(bigrand.nextLong());
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

			//create seeds rdd
			seedsRDD = sec.getSparkContext()
					.textFile(path.toString(), numPartitions)
					.mapToPair(new ExtractTensorSeedTuple());
		}

		//step 4: execute rand instruction over seed input
		// TODO getDimLengthPerBlock accurate for each dimension
		JavaPairRDD<TensorIndexes, BasicTensor> out = seedsRDD
				.mapToPair(new GenerateRandomTensorBlock(output.getValueType(), tDims, blocksize,
						sparsity, minValueStr, maxValueStr, pdf, pdfParams));

		//step 5: output handling
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		if(!mcOut.dimsKnown(true)) {
			mcOut.set(tmp);
		}
		sec.setRDDHandleForVariable(output.getName(), out);
	}

	private void generateSequence(SparkExecutionContext sec) {
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
		double totalSize = OptimizerUtils.estimatePartitionedSizeExactSparsity( nnz, 1, blocksize, 
			nnz); //overestimate for on disk, ensures hdfs block per partition
		double hdfsBlkSize = InfrastructureAnalyzer.getHDFSBlockSize();
		long numBlocks = (long)Math.ceil(((double)nnz)/blocksize);
	
		//a) in-memory offset rdd construction 
		if( numBlocks < INMEMORY_NUMBLOCKS_THRESHOLD )
		{
			ArrayList<Double> offsets = new ArrayList<>();
			for( long i=0; i<numBlocks; i++ ) {
				double off = lfrom + lincr*i*blocksize;
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
					double off = lfrom + lincr*i*blocksize;
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
			.mapToPair(new GenerateSequenceBlock(blocksize, lfrom, lto, lincr));

		//step 3: output handling
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			mcOut.set(nnz, 1, blocksize, nnz);
		}
		sec.setRDDHandleForVariable(output.getName(), out);
	}
	
	/**
	 * Helper function to construct a sample.
	 * 
	 * @param sec spark execution context
	 */
	private void generateSample(SparkExecutionContext sec) {
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
		long partitionSize = (long) Math.ceil(maxValue /numPartitions);

		ArrayList<SampleTask> offsets = new ArrayList<>();
		long st = 1;
		while (st <= maxValue) {
			SampleTask s = new SampleTask();
			s.range_start = st;
			s.seed = bigrand.nextLong();
			offsets.add(s);
			st = st + partitionSize;
		}
		JavaRDD<SampleTask> offsetRDD = sec.getSparkContext().parallelize(offsets, numPartitions);
		
		// Construct the sample in a distributed manner
		JavaRDD<Double> rdd = offsetRDD.flatMap( (new GenerateSampleBlock(replace, fraction, (long) maxValue, partitionSize)) );
		
		// Randomize the sampled elements
		JavaRDD<Double> randomizedRDD = rdd.mapToPair(new AttachRandom()).sortByKey().values();
		
		// Trim the sampled list to required size & attach matrix indexes to randomized elements
		JavaPairRDD<MatrixIndexes, MatrixCell> miRDD = randomizedRDD
				.zipWithIndex()
				.filter( new TrimSample(lrows) )
				.mapToPair( new Double2MatrixCell() );
		
		DataCharacteristics mcOut = new MatrixCharacteristics(lrows, 1, blocksize, lrows);
		
		// Construct BinaryBlock representation
		JavaPairRDD<MatrixIndexes, MatrixBlock> mbRDD = 
				RDDConverterUtils.binaryCellToBinaryBlock(sec.getSparkContext(), miRDD, mcOut, true);
		
		sec.getDataCharacteristics(output.getName()).setNonZeros(lrows);
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

	private static class ExtractMatrixSeedTuple implements PairFunction<String, MatrixIndexes, Long> {
		private static final long serialVersionUID = 3973794676854157101L;

		@Override
		public Tuple2<MatrixIndexes, Long> call(String arg)
				throws Exception 
		{
			String[] parts = IOUtilFunctions.split(arg, ",");
			MatrixIndexes ix = new MatrixIndexes(
				Long.parseLong(parts[0]), Long.parseLong(parts[1]));
			return new Tuple2<>(ix,Long.parseLong(parts[2]));
		}
	}

	private static class ExtractTensorSeedTuple implements PairFunction<String, TensorIndexes, Long> {
		private static final long serialVersionUID = 3973794676854157101L;

		@Override
		public Tuple2<TensorIndexes, Long> call(String arg)
				throws Exception
		{
			String[] parts = IOUtilFunctions.split(arg, ",");
			long[] ix = new long[parts.length - 1];
			for (int i = 0; i < parts.length - 1; i++)
				ix[i] = Long.parseLong(parts[i]);
			TensorIndexes to = new TensorIndexes(ix);
			return new Tuple2<>(to,Long.parseLong(parts[parts.length - 1]));
		}
	}

	private static class ExtractOffsetTuple implements Function<String, Double> {
		private static final long serialVersionUID = -3980257526545002552L;

		@Override
		public Double call(String arg) throws Exception {
			return Double.parseDouble(arg);
		}
	}

	private static class GenerateRandomBlock implements PairFunction<Tuple2<MatrixIndexes, Long>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 1616346120426470173L;
		
		private long _rlen; 
		private long _clen;
		private int _blen;
		private double _sparsity; 
		private double _min; 
		private double _max; 
		private String _pdf; 
		private String _pdfParams;
		
		public GenerateRandomBlock(long rlen, long clen, int blen, double sparsity, double min, double max, String pdf, String pdfParams) {
			_rlen = rlen;
			_clen = clen;
			_blen = blen;
			_sparsity = sparsity;
			_min = min;
			_max = max;
			_pdf = pdf;
			_pdfParams = pdfParams;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Long> kv) 
			throws Exception 
		{
			//compute local block size: 
			MatrixIndexes ix = kv._1();
			long blockRowIndex = ix.getRowIndex();
			long blockColIndex = ix.getColumnIndex();
			int lrlen = UtilFunctions.computeBlockSize(_rlen, blockRowIndex, _blen);
			int lclen = UtilFunctions.computeBlockSize(_clen, blockColIndex, _blen);
			long seed = kv._2;
			
			MatrixBlock blk = new MatrixBlock();
			RandomMatrixGenerator rgen = LibMatrixDatagen
				.createRandomMatrixGenerator(_pdf, lrlen, lclen,
					lrlen, _sparsity, _min, _max, _pdfParams);
			blk.randOperationsInPlace(rgen, null, seed);
			blk.examSparsity();
			return new Tuple2<>(kv._1, blk);
		}
	}

	private static class GenerateRandomTensorBlock implements PairFunction<Tuple2<TensorIndexes, Long>, TensorIndexes, BasicTensor>
	{
		private static final long serialVersionUID = -512119897654170462L;

		private ValueType _vt;
		private int[] _dims;
		private int _blen;
		private double _sparsity;
		private String _min;
		private String _max;
		private String _pdf;
		private String _pdfParams;

		public GenerateRandomTensorBlock(ValueType vt, int[] dims, int blen,
			double sparsity, String min, String max, String pdf, String pdfParams) {
			_vt = vt;
			_dims = new int[dims.length];
			Array.copy(dims, 0, _dims, 0, dims.length);
			_blen = blen;
			_sparsity = sparsity;
			_min = min;
			_max = max;
			_pdf = pdf;
			_pdfParams = pdfParams;
		}

		@Override
		public Tuple2<TensorIndexes, BasicTensor> call(Tuple2<TensorIndexes, Long> kv)
				throws Exception
		{
			//compute local block size:
			TensorIndexes ix = kv._1();
			// TODO: accurate block size computation
			int[] blockDims = new int[_dims.length];
			blockDims[0] = UtilFunctions.computeBlockSize(_dims[0], ix.getIndex(0), _blen);
			for (int i = 1; i < _dims.length; i++) {
				blockDims[i] = UtilFunctions.computeBlockSize(_dims[i], ix.getIndex(i), _blen);
			}
			int clen = (int) UtilFunctions.prod(blockDims, 1);
			long seed = kv._2;

			BasicTensor tb = new BasicTensor(_vt, blockDims);
			// TODO implement sparse support
			tb.allocateDenseBlock();
			if (!_min.equals(_max)) {
				if (_vt == ValueType.STRING) {
					throw new DMLRuntimeException("Random string data can not be generated for tensors.");
				}
				MatrixBlock blk = new MatrixBlock();
				RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(_pdf, blockDims[0], clen,
					_blen, _sparsity, Double.parseDouble(_min), Double.parseDouble(_max), _pdfParams);
				blk.randOperationsInPlace(rgen, null, seed);
				blk.examSparsity();
				tb.set(blk);
			}
			else {
				switch (_vt) {
					case STRING:
					case BOOLEAN:
						tb.set(_min);
						break;
					case INT64:
					case INT32:
						tb.set(Long.parseLong(_min));
						break;
					default:
						tb.set(Double.parseDouble(_min));
						break;
				}
			}

			return new Tuple2<>(kv._1, tb);
		}
	}

	private static class GenerateSequenceBlock implements PairFunction<Double, MatrixIndexes, MatrixBlock>
	{
		private static final long serialVersionUID = 5779681055705756965L;
		
		private final double _global_seq_start;
		private final double _global_seq_end;
		private final double _seq_incr;
		private final int _blen;
		
		public GenerateSequenceBlock(int blen, double global_seq_start, double global_seq_end, double seq_incr) {
			_global_seq_start = global_seq_start;
			_global_seq_end = global_seq_end;
			_seq_incr = seq_incr;
			_blen = blen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Double seq_from) 
			throws Exception 
		{
			double seq_to = (_seq_incr > 0) ?
				Math.min(_global_seq_end, seq_from + _seq_incr*(_blen-1)) :
				Math.max(_global_seq_end, seq_from + _seq_incr*(_blen+1));
			long globalRow = (long)Math.round((seq_from-_global_seq_start)/_seq_incr)+1;
			long rowIndex = UtilFunctions.computeBlockIndex(globalRow, _blen);
			
			MatrixIndexes indx = new MatrixIndexes(rowIndex, 1);
			MatrixBlock blk = MatrixBlock.seqOperations(seq_from, seq_to, _seq_incr);
			return new Tuple2<>(indx, blk);
		}	
	}
	
	/**
	 * This will check if there is sufficient memory locally.
	 * 
	 * @param lrows number of rows
	 * @param lcols number of columns
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

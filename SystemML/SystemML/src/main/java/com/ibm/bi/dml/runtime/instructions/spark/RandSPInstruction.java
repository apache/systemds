package com.ibm.bi.dml.runtime.instructions.spark;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.random.Well1024a;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixDatagen;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class RandSPInstruction extends UnarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private DataGenMethod method = DataGenMethod.INVALID;
	
	private long rows;
	private long cols;
	private int rowsInBlock;
	private int colsInBlock;
	private double minValue;
	private double maxValue;
	private double sparsity;
	private String pdf;
	private long seed=0;
	private double seq_from;
	private double seq_to; 
	private double seq_incr;
	
	
	public RandSPInstruction (Operator op, 
							  DataGenMethod mthd,
							  CPOperand in, 
							  CPOperand out, 
							  long rows, 
							  long cols,
							  int rpb, int cpb,
							  double minValue, 
							  double maxValue,
							  double sparsity, 
							  long seed,
							  String probabilityDensityFunction,
							  String opcode,
							  String istr) {
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
		this.pdf = probabilityDensityFunction;

	}

	public RandSPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out,
			long rows, long cols, int rpb, int cpb, double seqFrom,
			double seqTo, double seqIncr, String opcode, String istr) {
		super(op, in, out, opcode, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.seq_from = seqFrom;
		this.seq_to = seqTo;
		this.seq_incr = seqIncr;
	}

	public long getRows() {
		return rows;
	}

	public void setRows(long rows) {
		this.rows = rows;
	}

	public long getCols() {
		return cols;
	}

	public void setCols(long cols) {
		this.cols = cols;
	}

	public int getRowsInBlock() {
		return rowsInBlock;
	}

	public void setRowsInBlock(int rowsInBlock) {
		this.rowsInBlock = rowsInBlock;
	}

	public int getColsInBlock() {
		return colsInBlock;
	}

	public void setColsInBlock(int colsInBlock) {
		this.colsInBlock = colsInBlock;
	}

	public double getMinValue() {
		return minValue;
	}

	public void setMinValue(double minValue) {
		this.minValue = minValue;
	}

	public double getMaxValue() {
		return maxValue;
	}

	public void setMaxValue(double maxValue) {
		this.maxValue = maxValue;
	}

	public double getSparsity() {
		return sparsity;
	}

	public void setSparsity(double sparsity) {
		this.sparsity = sparsity;
	}

	public static Instruction parseInstruction(String str) throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		DataGenMethod method = DataGenMethod.INVALID;
		if ( opcode.equalsIgnoreCase(DataGen.RAND_OPCODE) ) {
			method = DataGenMethod.RAND;
			InstructionUtils.checkNumFields ( str, 10 );
		}
		else if ( opcode.equalsIgnoreCase(DataGen.SEQ_OPCODE) ) {
			method = DataGenMethod.SEQ;
			// 8 operands: rows, cols, rpb, cpb, from, to, incr, outvar
			InstructionUtils.checkNumFields ( str, 8 ); 
		}
		
		Operator op = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		out.split(s[s.length-1]); // ouput is specified by the last operand

		if ( method == DataGenMethod.RAND ) {
			long rows = -1, cols = -1;
	        if (!s[1].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
			   	rows = Double.valueOf(s[1]).longValue();
	        }
	        if (!s[2].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
	        	cols = Double.valueOf(s[2]).longValue();
	        }
			
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			
			double minValue = -1, maxValue = -1;
	        if (!s[5].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
			   	minValue = Double.valueOf(s[5]).doubleValue();
	        }
	        if (!s[6].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
	        	maxValue = Double.valueOf(s[6]).doubleValue();
	        }
	        
	        double sparsity = Double.parseDouble(s[7]);
			long seed = Long.parseLong(s[8]);
			String pdf = s[9];
			
			return new RandSPInstruction(op, method, null, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, opcode, str);
		}
		else if ( method == DataGenMethod.SEQ) {
			// Example Instruction: CP:seq:11:1:1000:1000:1:0:-0.1:scratch_space/_p7932_192.168.1.120//_t0/:mVar1
			long rows = Double.valueOf(s[1]).longValue();
			long cols = Double.valueOf(s[2]).longValue();
			int rpb = Integer.parseInt(s[3]);
			int cpb = Integer.parseInt(s[4]);
			
	        double from, to, incr;
	        from = to = incr = Double.NaN;
			if (!s[5].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
				from = Double.valueOf(s[5]);
	        }
			if (!s[6].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
				to   = Double.valueOf(s[6]);
	        }
			if (!s[7].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
				incr = Double.valueOf(s[7]);
	        }
			
			CPOperand in = null;
			return new RandSPInstruction(op, method, in, out, rows, cols, rpb, cpb, from, to, incr, opcode, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException
	{
		
		//check valid for integer dimensions (we cannot even represent empty blocks with larger dimensions)
		if( rows > Integer.MAX_VALUE || cols > Integer.MAX_VALUE )
			throw new DMLRuntimeException("RandCPInstruction does not support dimensions larger than integer: rows="+rows+", cols="+cols+".");
		
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		//process specific datagen operator
		if ( this.method == DataGenMethod.RAND ) {
			// The implementation is in same spirit as MapReduce
			// We generate seeds similar to com.ibm.bi.dml.runtime.matrix.DataGenMR
			// and then generate blocks similar to com.ibm.bi.dml.runtime.matrix.mapred.DataGenMapper
			
			//generate pseudo-random seed (because not specified) 
			long lSeed = seed; //seed per invocation
			if( lSeed == DataGenOp.UNSPECIFIED_SEED ) 
				lSeed = DataGenOp.generateRandomSeed();
			
			// seed generation: 
			// TODO:This will create issue for extremely large dataset, but will work for preview
			Well1024a bigrand = LibMatrixDatagen.setupSeedsForRand(lSeed);
			long[] nnz = LibMatrixDatagen.computeNNZperBlock(rows, cols, rowsInBlock, colsInBlock, sparsity);
			ArrayList<Tuple2<MatrixIndexes, Tuple2<Long, Long>>> seeds = new ArrayList<Tuple2<MatrixIndexes, Tuple2<Long, Long>>>();
			int iter = 0;
			double meanNNz = 0; 
			long numBlocks = 0;
			for(long r = 1; r <= (long)Math.ceil((double)rows/(double)rowsInBlock); r++) {
				for(long c = 1; c <= (long)Math.ceil((double)cols/(double)colsInBlock); c++) {
					MatrixIndexes indx = new MatrixIndexes(r, c);
					Long seedForBlock = bigrand.nextLong();
					seeds.add(new Tuple2<MatrixIndexes, Tuple2<Long, Long>>(indx, new Tuple2<Long, Long>(seedForBlock, nnz[iter])));
					meanNNz += nnz[iter]/nnz.length;
					iter++;
					numBlocks++;
				}
			}
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process RandCPInstruction rand with seed = "+lSeed+".");
			
			
			// For load balancing: degree of parallelism such that ~128MB per partition
			double avgNumBytesOfRandomBlock = meanNNz*8 + 16;
			int numPartitions = (int) Math.max(Math.min(avgNumBytesOfRandomBlock*numBlocks / (128 * 10^6), numBlocks), 1);
			
			JavaPairRDD<MatrixIndexes, Tuple2<Long, Long>> seedsRDD = JavaPairRDD.fromJavaRDD(sec.getSparkContext().parallelize(seeds, numPartitions));
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = seedsRDD.mapToPair(new GenerateRandomBlock(rows, cols, rowsInBlock, colsInBlock, sparsity, minValue, maxValue, pdf)); 
			
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			if(!mcOut.dimsKnown()) {
				mcOut.set(rows, cols, rowsInBlock, colsInBlock, (long) (meanNNz*nnz.length));
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
		}
		else if ( this.method == DataGenMethod.SEQ ) {
			
			if(seq_incr == 0) {
				throw new DMLRuntimeException("ERROR: While performing seq(" + seq_from + "," + seq_to + "," + seq_incr + ")");
			}
			
			// Generate start seq for each blocks
			// TODO: bad way: This will create issue for ultra sparse dataset, but will work for preview
			ArrayList<Double> startSequences = new ArrayList<Double>();
			double start = seq_from;
			long nnz = (long) Math.abs(Math.round((seq_to - seq_from)/seq_incr)) + 1;
			
			while((seq_incr > 0 && start <= seq_to) || (seq_incr < 0 && start >= seq_to)) {
				startSequences.add(start);
				
				// 1 entry at seq_incr
				// rowsInBlock entries at seq_incr*rowsInBlock
				start += seq_incr*rowsInBlock;
			}
			
			if(nnz != rows && rows != -1) {
				throw new DMLRuntimeException("Incorrect number of non-zeros: " + nnz + " != " + rows);
			}
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process RandCPInstruction seq with seqFrom="+seq_from+", seqTo="+seq_to+", seqIncr"+seq_incr);
			
			JavaRDD<Double> startSequencesRDD = sec.getSparkContext().parallelize(startSequences);
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = startSequencesRDD.mapToPair(new GenerateSequenceBlock(rowsInBlock, seq_from, seq_to, seq_incr));

			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			if(!mcOut.dimsKnown()) {
				mcOut.set(nnz, 1, rowsInBlock, colsInBlock, nnz);
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
		}
		
	}
	
	public static class GenerateSequenceBlock implements PairFunction<Double, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = 5779681055705756965L;
		int brlen; double global_seq_end; double seq_incr;
		double global_seq_start;
		
		public GenerateSequenceBlock(int brlen, double global_seq_start, double global_seq_end, double seq_incr) {
			this.brlen = brlen;
			this.global_seq_end = global_seq_end;
			this.global_seq_start = global_seq_start;
			this.seq_incr = seq_incr;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Double seq_from) throws Exception {
			double seq_to;
			if(seq_incr > 0) {
				seq_to = Math.min(global_seq_end, seq_from + seq_incr*(brlen-1));
			}
			else {
				seq_to = Math.max(global_seq_end, seq_from + seq_incr*(brlen+1));
			}
			long globalRow = (long) ((seq_from-global_seq_start)/seq_incr + 1);
			long rowIndex = (long) Math.ceil((double)globalRow/(double)brlen);
			
			MatrixIndexes indx = new MatrixIndexes(rowIndex, 1);
			MatrixBlock blk = MatrixBlock.seqOperations(seq_from, seq_to, seq_incr);
			return new Tuple2<MatrixIndexes, MatrixBlock>(indx, blk);
		}

		
	}
	
	public static class GenerateRandomBlock implements PairFunction<Tuple2<MatrixIndexes, Tuple2<Long, Long> >, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = 1616346120426470173L;
		
		int brlen; int bclen; double sparsity; double min; double max; String pdf;
		long rlen; long clen;
		public GenerateRandomBlock(long rlen, long clen, int brlen, int bclen, double sparsity, double min, double max, String pdf) {
			this.rlen = rlen;
			this.clen = clen;
			this.brlen = brlen;
			this.bclen = bclen;
			this.sparsity = sparsity;
			this.min = min;
			this.max = max;
			this.pdf = pdf;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Tuple2<Long, Long>> kv) throws Exception {
			// ------------------------------------------------------------------
			//	Compute local block size: 
			long blockRowIndex = kv._1.getRowIndex();
			long blockColIndex = kv._1.getColumnIndex();
			int lrlen = UtilFunctions.computeBlockSize(rlen, blockRowIndex, brlen);
			int lclen = UtilFunctions.computeBlockSize(clen, blockColIndex, bclen);
			// ------------------------------------------------------------------
			
			long seed = kv._2._1;
			long blockNNZ = kv._2._2;
			
			MatrixBlock blk = new MatrixBlock();
			
			if( LibMatrixDatagen.RAND_PDF_NORMAL.equals(pdf) ) {
				blk.randOperationsInPlace(pdf, lrlen, lclen, lrlen, lclen, new long[]{blockNNZ}, sparsity, Double.NaN, Double.NaN, null, seed); 
			}
			else if( LibMatrixDatagen.RAND_PDF_UNIFORM.equals(pdf) ) {
				blk.randOperationsInPlace(pdf, lrlen, lclen, lrlen, lclen, new long[]{blockNNZ}, sparsity, min, max, null, seed);
			}
			else {
				throw new IOException("Unsupported rand pdf function: "+pdf);
			}
			
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, blk);
		}
		
	}
}

/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixDatagen;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.RandomMatrixGenerator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class DataGenCPInstruction extends UnaryCPInstruction
{
	
	private DataGenMethod method = DataGenMethod.INVALID;
	
	private long rows;
	private long cols;
	private int rowsInBlock;
	private int colsInBlock;
	private double minValue;
	private double maxValue;
	private double sparsity;
	private String pdf, pdfParams;
	private long seed=0;
	
	//sequence specific attributes
	private double seq_from;
	private double seq_to; 
	private double seq_incr;
	
	//sample specific attributes
	private boolean replace;
 	private int numThreads = -1;
	
	public DataGenCPInstruction (Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, 
							  long rows, long cols, int rpb, int cpb,
							  double minValue, double maxValue, double sparsity, long seed,
							  String probabilityDensityFunction, String pdfParams, int k, String opcode, String istr) 
	{
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
		this.numThreads = k;
		this.pdfParams = pdfParams;
	}

	public DataGenCPInstruction (Operator op, DataGenMethod mthd, CPOperand in, CPOperand out, 
			  					long rows, long cols, int rpb, int cpb, double maxValue,
			  					boolean replace, long seed, String opcode, String istr) 
	{
		super(op, in, out, opcode, istr);
		
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.maxValue = maxValue;
		this.replace = replace;
		this.seed = seed;
	}
	
	public DataGenCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out,
							long rows, long cols, int rpb, int cpb, double seqFrom,
							double seqTo, double seqIncr, String opcode, String istr) 
	{
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
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		out.split(s[s.length-1]); // ouput is specified by the last operand

		if ( method == DataGenMethod.RAND ) 
		{
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

			String pdfParams = null;
	        if (!s[10].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) {
	        	pdfParams = s[10];
	        }
	        
			int k = Integer.parseInt(s[11]);
			
			return new DataGenCPInstruction(op, method, null, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, pdfParams, k, opcode, str);
		}
		else if ( method == DataGenMethod.SEQ) 
		{
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
			
			return new DataGenCPInstruction(op, method, null, out, rows, cols, rpb, cpb, from, to, incr, opcode, str);
		}
		else if ( method == DataGenMethod.SAMPLE) 
		{
			// Example Instruction: CP:sample:10:100:false:1000:1000:_mVar2·MATRIX·DOUBLE
			double max = 0;
			long rows = 0, cols;
			boolean replace = false;
			
			if (!s[1].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) 
				max = Double.valueOf(s[1]);
			if (!s[2].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) 
				rows = Double.valueOf(s[2]).longValue();
			cols = 1;
			
			if (!s[3].contains( Lop.VARIABLE_NAME_PLACEHOLDER)) 
				replace = Boolean.valueOf(s[3]);
			
			long seed = Long.parseLong(s[4]);
			int rpb = Integer.parseInt(s[5]);
			int cpb = Integer.parseInt(s[6]);
			
			return new DataGenCPInstruction(op, method, null, out, rows, cols, rpb, cpb, max, replace, seed, opcode, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException
	{
		MatrixBlock soresBlock = null;
		
		//check valid for integer dimensions (we cannot even represent empty blocks with larger dimensions)
		if( rows > Integer.MAX_VALUE || cols > Integer.MAX_VALUE )
			throw new DMLRuntimeException("RandCPInstruction does not support dimensions larger than integer: rows="+rows+", cols="+cols+".");
		
		//process specific datagen operator
		if ( method == DataGenMethod.RAND ) {
			//generate pseudo-random seed (because not specified) 
			long lSeed = seed; //seed per invocation
			if( lSeed == DataGenOp.UNSPECIFIED_SEED ) 
				lSeed = DataGenOp.generateRandomSeed();
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process DataGenCPInstruction rand with seed = "+lSeed+".");
			
			RandomMatrixGenerator rgen = LibMatrixDatagen.createRandomMatrixGenerator(pdf, (int) rows, (int) cols, rowsInBlock, colsInBlock, sparsity, minValue, maxValue, pdfParams);
			soresBlock = MatrixBlock.randOperations(rgen, seed, numThreads);
		}
		else if ( method == DataGenMethod.SEQ ) 
		{
			if( LOG.isTraceEnabled() )
				LOG.trace("Process DataGenCPInstruction seq with seqFrom="+seq_from+", seqTo="+seq_to+", seqIncr"+seq_incr);
			
			soresBlock = MatrixBlock.seqOperations(seq_from, seq_to, seq_incr);
		}
		else if ( method == DataGenMethod.SAMPLE ) 
		{
			long range = UtilFunctions.toLong(maxValue);
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process DataGenCPInstruction sample with range="+range+", size="+rows+", replace"+replace + ", seed=" + seed);
			
			if ( range < rows && replace == false)
				throw new DMLRuntimeException("Sample (size=" + rows + ") larger than population (size=" + range + ") can only be generated with replacement.");
			
			soresBlock = MatrixBlock.sampleOperations(range, (int)rows, replace, seed);
		}
		
		//release created output
		ec.setMatrixOutput(output.getName(), soresBlock);
	}
}

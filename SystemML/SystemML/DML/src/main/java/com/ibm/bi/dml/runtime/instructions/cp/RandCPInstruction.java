/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class RandCPInstruction extends UnaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public long rows;
	public long cols;
	public int rowsInBlock;
	public int colsInBlock;
	public double minValue;
	public double maxValue;
	public double sparsity;
	public String pdf;
	public long seed=0;
	public double seq_from, seq_to, seq_incr;
	DataGenMethod method = DataGenMethod.INVALID;

	
	public RandCPInstruction (Operator op, 
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
							  String istr) {
		super(op, in, out, istr);
		
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

	public RandCPInstruction(Operator op, DataGenMethod mthd, CPOperand in, CPOperand out,
			long rows, long cols, int rpb, int cpb, double seqFrom,
			double seqTo, double seqIncr, String istr) {
		super(op, in, out, istr);
		this.method = mthd;
		this.rows = rows;
		this.cols = cols;
		this.rowsInBlock = rpb;
		this.colsInBlock = cpb;
		this.seq_from = seqFrom;
		this.seq_to = seqTo;
		this.seq_incr = seqIncr;
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
			double minValue = Double.parseDouble(s[5]);
			double maxValue = Double.parseDouble(s[6]);
			double sparsity = Double.parseDouble(s[7]);
			long seed = Long.parseLong(s[8]);
			String pdf = s[9];
			
			return new RandCPInstruction(op, method, null, out, rows, cols, rpb, cpb, minValue, maxValue, sparsity, seed, pdf, str);
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
			return new RandCPInstruction(op, method, in, out, rows, cols, rpb, cpb, from, to, incr, str);
		}
		else 
			throw new DMLRuntimeException("Unrecognized data generation method: " + method);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException
	{
		String output_name = output.get_name();
		MatrixBlock soresBlock = null;
		
		if ( this.method == DataGenMethod.RAND ) {
			//generate pseudo-random seed (because not specified) 
			long lSeed = seed; //seed per invocation
			if( lSeed == DataGenOp.UNSPECIFIED_SEED ) 
				lSeed = DataGenOp.generateRandomSeed();
			
			if( LOG.isTraceEnabled() )
				LOG.trace("Process RandCPInstruction with seed = "+lSeed+".");
			
			soresBlock = MatrixBlock.randOperations((int)rows, (int)cols, rowsInBlock, colsInBlock, sparsity, minValue, maxValue, pdf, seed);
		}
		else if ( this.method == DataGenMethod.SEQ ) {
			// (int)rows, (int)cols, rowsInBlock, colsInBlock, 
			soresBlock = MatrixBlock.seqOperations(seq_from, seq_to, seq_incr);
		}
		ec.setMatrixOutput(output_name, soresBlock);
	}
}

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.OffsetColumnIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

public class AppendGAlignedSPInstruction extends BinarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum AppendType{
		CBIND,
		STRING,
	}
	
	//type (matrix cbind / scalar string concatenation)
	private AppendType _type;
	
	public AppendGAlignedSPInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, AppendType type, String opcode, String istr)
	{
		super(op, in1, in2, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.GAppend;

		_type = type;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in3 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		//4 parts to the instruction besides opcode and execlocation
		//two input args, one output arg and offset = 4
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		in2.split(parts[2]);
		in3.split(parts[3]);
		out.split(parts[4]);
		//String offset_str = parts[4];
		 
		AppendType type = (in1.getDataType()==DataType.MATRIX) ? AppendType.CBIND : AppendType.STRING;
		
		
		if(!opcode.equalsIgnoreCase("galignedappend"))
			throw new DMLRuntimeException("Unknown opcode while parsing a AppendGSPInstruction: " + str);
		else
			return new AppendGAlignedSPInstruction(new ReorgOperator(OffsetColumnIndex.getOffsetColumnIndexFnObject(-1)), 
										   in1, in2, in3, out, type, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		if( _type == AppendType.CBIND )
		{		
			// general case append (map-extend, aggregate)

			
			SparkExecutionContext sec = (SparkExecutionContext)ec;
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mc2 = sec.getMatrixCharacteristics(input2.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			
			if(!mc1.dimsKnown() || !mc2.dimsKnown()) {
				throw new DMLRuntimeException("The dimensions unknown for inputs");
			}
			else if(mc1.getRows() != mc2.getRows()) {
				throw new DMLRuntimeException("The number of rows of inputs should match for append instruction");
			}
			else if(mc1.getRowsPerBlock() != mc2.getRowsPerBlock() || mc1.getColsPerBlock() != mc2.getColsPerBlock()) {
				throw new DMLRuntimeException("The block sizes donot match for input matrices");
			}
			
			if(!mcOut.dimsKnown()) {
				mcOut.set(mc1.getRows(), mc1.getCols() + mc2.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
			
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			
			if(mc1.getCols() % mc1.getColsPerBlock() == 0) {
				// Simple changing of matrix indexes of RHS
				long shiftBy = (long) Math.ceil((double)mc1.getCols() / (double)mc1.getColsPerBlock());
				out = in1.union(
							in2.mapToPair(new ShiftColumnIndex(shiftBy))
						);
			}
			else {
				throw new DMLRuntimeException("Incorrect append instruction when mc1.getCols() % mc1.getColsPerBlock() != 0. Should have used AppendGSP");
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			sec.addLineageRDD(output.getName(), input2.getName());
		}
		else //STRING
		{
			throw new DMLRuntimeException("Error: String append only valid as CPInstruction");
			// InstructionUtils.processStringAppendInstruction(ec, input1, input2, output);
		}		
	}
	
	public static class ShiftColumnIndex implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {

		private static final long serialVersionUID = -5185023611319654242L;
		long shiftBy;
		public ShiftColumnIndex(long shiftBy) {
			this.shiftBy = shiftBy;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			return new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(kv._1.getRowIndex(), kv._1.getColumnIndex()+shiftBy), kv._2);
		}
		
	}
}
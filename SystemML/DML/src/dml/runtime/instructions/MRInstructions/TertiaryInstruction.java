package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixBlock;
import dml.runtime.matrix.io.MatrixBlock1D;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class TertiaryInstruction extends MRInstruction {

	public byte input1, input2, input3;
	public double scalar_input2, scalar_input3;

	public TertiaryInstruction(Operator op, byte in1, byte in2, byte in3, byte out, String istr)
	{
		super(op, out);
		input1 = in1;
		input2 = in2;
		input3 = in3;
		mrtype = MRINSTRUCTION_TYPE.Tertiary;
		instString = istr;
	}
	
	public TertiaryInstruction(Operator op, byte in1, byte in2, double scalar_in3, byte out, String istr)
	{
		super(op, out);
		input1 = in1;
		input2 = in2;
		scalar_input3 = scalar_in3;
		mrtype = MRINSTRUCTION_TYPE.Tertiary;
		instString = istr;
	}
	
	public TertiaryInstruction(Operator op, byte in1, double scalar_in2, double scalar_in3, byte out, String istr)
	{
		super(op, out);
		input1 = in1;
		scalar_input2 = scalar_in2;
		scalar_input3 = scalar_in3;
		mrtype = MRINSTRUCTION_TYPE.Tertiary;
		instString = istr;
	}
	
	public TertiaryInstruction(Operator op, byte in1, double scalar_in2, byte in3, byte out, String istr)
	{
		super(op, out);
		input1 = in1;
		scalar_input2 = scalar_in2;
		input3 = in3;
		mrtype = MRINSTRUCTION_TYPE.Tertiary;
		instString = istr;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		// example instruction string 
		// - ctabletransform:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE:::3:DOUBLE 
		// - ctabletransformscalarweight:::0:DOUBLE:::1:DOUBLE:::1.0:DOUBLE:::3:DOUBLE 
		// - ctabletransformhistogram:::0:DOUBLE:::1.0:DOUBLE:::1.0:DOUBLE:::3:DOUBLE 
		// - ctabletransformweightedhistogram:::0:DOUBLE:::1:INT:::1:DOUBLE:::2:DOUBLE 
		InstructionUtils.checkNumFields ( str, 4 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in1, in2, in3, out;
		
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[4]);
		
		if ( opcode.equalsIgnoreCase("ctabletransform")) {
			in2 = Byte.parseByte(parts[2]);
			in3 = Byte.parseByte(parts[3]);
			return new TertiaryInstruction(null, in1, in2, in3, out, str);
		}
		else if (opcode.equalsIgnoreCase("ctabletransformscalarweight")){
			in2 = Byte.parseByte(parts[2]);
			double scalar_in3 = Double.parseDouble(parts[3]);
			return new TertiaryInstruction(null, in1, in2, scalar_in3, out, str);
		} else if (opcode.equalsIgnoreCase("ctabletransformhistogram")){
			double scalar_in2 = Double.parseDouble(parts[2]);
			double scalar_in3 = Double.parseDouble(parts[3]);
			return new TertiaryInstruction(null, in1, scalar_in2, scalar_in3, out, str);
		} else if (opcode.equalsIgnoreCase("ctabletransformweightedhistogram")){
			double scalar_in2 = Double.parseDouble(parts[2]);
			in3 = Byte.parseByte(parts[3]);
			return new TertiaryInstruction(null, in1, scalar_in2, in3, out, str);
		} else {
			throw new DMLRuntimeException("Unrecognized opcode in Tertiary Instruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if ( valueClass != MatrixBlock1D.class && valueClass != MatrixBlock.class ) {
			throw new DMLRuntimeException("Unexpected value class for TertiaryInstruction.");
		}
		
		String opcode = InstructionUtils.getOpCode(instString);
		
		IndexedMatrixValue in1, in2, in3 = null, out=null;
		in1 = cachedValues.get(input1);
		if ( opcode.equalsIgnoreCase("ctabletransform") ) {
			in2 = cachedValues.get(input2);
			in3 = cachedValues.get(input3);
			if(in1==null || in2==null || in3 == null )
				return;
			
			//allocate space for the output value
			if( (output!=input1 && output!=input2 && output != input3)
				|| (output==input1 && in1==null)
				|| (output==input2 && in2==null)
				|| (output==input3 && in3==null)
				)
				out=cachedValues.holdPlace(output, valueClass);
			else
				out=tempValue;
		}
		else if ( opcode.equalsIgnoreCase("ctabletransformscalarweight")) {
			// 3rd input is a scalar
			in2 = cachedValues.get(input2);
			in3 = null;
			if(in1==null || in2==null )
				return;
			
			//allocate space for the output value
			if( (output!=input1 && output!=input2)
				|| (output==input1 && in1==null)
				|| (output==input2 && in2==null)
				)
				out=cachedValues.holdPlace(output, valueClass);
			else
				out=tempValue;
		} else if (opcode.equalsIgnoreCase("ctabletransformhistogram")) {
			// 2nd and 3rd inputs are scalars
			in2 = null;
			in3 = null;
			if(in1==null )
				return;
			
			//allocate space for the output value
			if( (output!=input1)
				|| (output==input1 && in1==null)
				)
				out=cachedValues.holdPlace(output, valueClass);
			else
				out=tempValue;
		} else if (opcode.equalsIgnoreCase("ctabletransformweightedhistogram")) {
			// 2nd and 3rd inputs are scalars
			in2 = null;
			in3 = cachedValues.get(input3);
			if(in1==null || in3==null)
				return;
			
			//allocate space for the output value
			if( (output!=input1 && output != input3)
					|| (output==input1 && in1==null)
					|| (output==input3 && in3==null)
					)
					out=cachedValues.holdPlace(output, valueClass);
				else
					out=tempValue;
		} else {
			throw new DMLRuntimeException("Unrecognized opcode in Tertiary Instruction: " + instString);
		}
		
		
		// NOTE: this may require modification if Tertiary is used for operations other than ctable()
		MatrixIndexes finalIndexes=new MatrixIndexes(0, 0);
		out.getIndexes().setIndexes(finalIndexes);
		
		// Dirty hack to make maxrow and maxcolumn in ReduceBase.collectOutput_N_Increase_Counter() work
		((MatrixBlock1D) out.getValue()).setNumRows(1);
		((MatrixBlock1D) out.getValue()).setNumColumns(1);
		
		//process instruction
		if ( opcode.equalsIgnoreCase("ctabletransform") ) {
			OperationsOnMatrixValues.performTertiary(in1.getIndexes(), in1.getValue(), in2.getIndexes(), in2.getValue(), in3.getIndexes(), in3.getValue(), out.getIndexes(), out.getValue(), optr);
		} else if ( opcode.equalsIgnoreCase("ctabletransformscalarweight") ) {
			OperationsOnMatrixValues.performTertiary(in1.getIndexes(), in1.getValue(), in2.getIndexes(), in2.getValue(), scalar_input3, out.getIndexes(), out.getValue(), optr);
		} else if ( opcode.equalsIgnoreCase("ctabletransformhistogram") ) {
			OperationsOnMatrixValues.performTertiary(in1.getIndexes(), in1.getValue(), scalar_input2, scalar_input3, out.getIndexes(), out.getValue(), optr);
		} else if ( opcode.equalsIgnoreCase("ctabletransformweightedhistogram") ) {
			OperationsOnMatrixValues.performTertiary(in1.getIndexes(), in1.getValue(), scalar_input2, in3.getIndexes(), in3.getValue(), out.getIndexes(), out.getValue(), optr);
		}
		//put the output value in the cache
		if(out==tempValue) {
			throw new DMLRuntimeException("ctable_transform.processInstruction(): unexpected error.");
			// cachedValues.set(output, out);
		}
		
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		return new byte[]{input1, input2, input3, output};
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		return new byte[]{input1, input2, input3};
	}

}

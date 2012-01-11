package dml.runtime.instructions.MRInstructions;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.SelectOperator;
import dml.runtime.util.UtilFunctions;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class SelectInstruction extends UnaryMRInstructionBase{

	//start and end are all inclusive
	public static class IndexRange
	{
		public long rowStart=0;
		public long rowEnd=0;
		public long colStart=0;
		public long colEnd=0;
		
		public void set(long rs, long re, long cs, long ce)
		{
			rowStart=rs;
			rowEnd=re;
			colStart=cs;
			colEnd=ce;
		}
	}
	
	public IndexRange indexRange=null;
	private IndexRange tempRange=new IndexRange();
	
	public SelectInstruction(Operator op, byte in, byte out, IndexRange rng, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.Select;
		instString = istr;
		indexRange=rng;
	}
	
	private static long parseStartBoundary(String str)
	{
		if(!str.isEmpty())
			return Long.parseLong(str);
		else
			return 1;
	}
	
	private static long parseEndBoundary(String str)
	{
		if(!str.isEmpty())
			return Long.parseLong(str);
		else
			return Long.MAX_VALUE;
	}
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		byte in, out;
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase("sel"))
			throw new DMLRuntimeException("Unknown opcode while parsing a Select: " + str);
		in = Byte.parseByte(parts[1]);
		out = Byte.parseByte(parts[2]);
		IndexRange rng=new IndexRange();
		String[] strs=InstructionUtils.getInstructionPartsWithValueType(parts[3]);
		if(strs.length!=4)
			throw new DMLRuntimeException("ill formated range " + parts[3]);
		rng.rowStart=parseStartBoundary(strs[0]);
		rng.rowEnd=parseEndBoundary(strs[1]);
		rng.colStart=parseStartBoundary(strs[2]);
		rng.colEnd=parseEndBoundary(strs[3]);
		return new SelectInstruction(new SelectOperator(), in, out, rng, str);
		
	}

	private IndexRange getSelected(IndexedMatrixValue in, int blockRowFactor, int blockColFactor) {
		
		long topblock=UtilFunctions.blockIndexCalculation(indexRange.rowStart, blockRowFactor);
		int toprow=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, blockRowFactor);
		long bottomblock=UtilFunctions.blockIndexCalculation(indexRange.rowEnd, blockRowFactor);
		int bottomrow=UtilFunctions.cellInBlockCalculation(indexRange.rowEnd, blockRowFactor);
		
		long leftblock=UtilFunctions.blockIndexCalculation(indexRange.colStart, blockColFactor);
		int leftcol=UtilFunctions.cellInBlockCalculation(indexRange.colStart, blockColFactor);
		long rightblock=UtilFunctions.blockIndexCalculation(indexRange.colEnd, blockColFactor);
		int rightcol=UtilFunctions.cellInBlockCalculation(indexRange.colEnd, blockColFactor);
		if(in.getIndexes().getRowIndex()<topblock || in.getIndexes().getRowIndex()>bottomblock
		   || in.getIndexes().getColumnIndex()<leftblock || in.getIndexes().getColumnIndex()>rightblock)
		{
			tempRange.set(-1,-1,-1,-1);
			return tempRange;
		}
		tempRange.set(0, in.getValue().getNumRows()-1, 0, in.getValue().getNumColumns()-1);
		if(topblock==in.getIndexes().getRowIndex())
			tempRange.rowStart=toprow;
		if(bottomblock==in.getIndexes().getRowIndex())
			tempRange.rowEnd=bottomrow;
		if(leftblock==in.getIndexes().getColumnIndex())
			tempRange.colStart=leftcol;
		if(rightblock==in.getIndexes().getColumnIndex())
			tempRange.colEnd=rightcol;
		
		return tempRange;
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		IndexedMatrixValue in=cachedValues.get(input);
		if(in==null)
			return;
		
		tempRange=getSelected(in, blockRowFactor, blockColFactor);
		if(tempRange.rowStart==-1)
			return;
		
		//allocate space for the output value
		IndexedMatrixValue out;
		if(input==output)
			out=tempValue;
		else
			out=cachedValues.holdPlace(output, valueClass);
		
		//process instruction
		
		OperationsOnMatrixValues.performSelect(in.getIndexes(), in.getValue(), 
				out.getIndexes(), out.getValue(), tempRange);
		
		//put the output value in the cache
		if(out==tempValue)
			cachedValues.set(output, out);
		
	}
}

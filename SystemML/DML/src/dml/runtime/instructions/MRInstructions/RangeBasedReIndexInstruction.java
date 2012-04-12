package dml.runtime.instructions.MRInstructions;

import java.util.ArrayList;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.InstructionUtils;
import dml.runtime.matrix.io.MatrixBlockDSM;
import dml.runtime.matrix.io.MatrixIndexes;
import dml.runtime.matrix.io.MatrixValue;
import dml.runtime.matrix.io.OperationsOnMatrixValues;
import dml.runtime.matrix.mapred.CachedValueMap;
import dml.runtime.matrix.mapred.IndexedMatrixValue;
import dml.runtime.matrix.operators.Operator;
import dml.runtime.matrix.operators.ReIndexOperator;
import dml.runtime.util.UtilFunctions;
import dml.utils.DMLRuntimeException;
import dml.utils.DMLUnsupportedOperationException;

public class RangeBasedReIndexInstruction extends UnaryMRInstructionBase{

	//start and end are all inclusive
	public static class IndexRange
	{
		public final static String DELIMITOR="*";
		public long rowStart=0;
		public long rowEnd=0;
		public long colStart=0;
		public long colEnd=0;
		
		public IndexRange(long rs, long re, long cs, long ce)
		{
			set(rs, re, cs, ce);
		}
		public void set(long rs, long re, long cs, long ce)
		{
			rowStart=rs;
			rowEnd=re;
			colStart=cs;
			colEnd=ce;
		}
		public String toString()
		{
			return rowStart+DELIMITOR+rowEnd+DELIMITOR+colStart+DELIMITOR+colEnd;
		}
		public void set(String str)
		{
			String[] strs=str.split(DELIMITOR);
			if(strs.length!=4)
				throw new RuntimeException("ill formated range " + str);
			rowStart=Long.parseLong(strs[0]);
			rowEnd=Long.parseLong(strs[1]);
			colStart=Long.parseLong(strs[2]);
			colEnd=Long.parseLong(strs[3]);
		}
	}
	
	public IndexRange indexRange=null;
	private IndexRange tempRange=new IndexRange(-1, -1, -1, -1);
	
	public RangeBasedReIndexInstruction(Operator op, byte in, byte out, IndexRange rng, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.RangeReIndex;
		instString = istr;
		indexRange=rng;
	}
	
	
	
	public static Instruction parseInstruction ( String str ) throws DMLRuntimeException {
		
		InstructionUtils.checkNumFields ( str, 6 );
		
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		if(!opcode.equalsIgnoreCase("rangeReIndex"))
			throw new DMLRuntimeException("Unknown opcode while parsing a Select: " + str);
		byte in = Byte.parseByte(parts[1]);
		IndexRange rng=new IndexRange(Long.parseLong(parts[2]), Long.parseLong(parts[3]), Long.parseLong(parts[4]), Long.parseLong(parts[5]));
		byte out = Byte.parseByte(parts[6]);
		return new RangeBasedReIndexInstruction(new ReIndexOperator(), in, out, rng, str);
		
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		
		if(input==output)
			throw new DMLRuntimeException("input cannot be the same for output for "+this.instString);
		
		IndexedMatrixValue in=cachedValues.getFirst(input);
		if(in==null)
			return;
		
		long cellIndexTopRow=UtilFunctions.cellIndexCalculation(in.getIndexes().getRowIndex(), blockRowFactor, 0);
		long cellIndexBottomRow=UtilFunctions.cellIndexCalculation(in.getIndexes().getRowIndex(), blockRowFactor, in.getValue().getNumRows()-1);
		long cellIndexLeftCol=UtilFunctions.cellIndexCalculation(in.getIndexes().getColumnIndex(), blockColFactor, 0);
		long cellIndexRightCol=UtilFunctions.cellIndexCalculation(in.getIndexes().getColumnIndex(), blockColFactor, in.getValue().getNumColumns()-1);
		
		long cellIndexOverlapTop=Math.max(cellIndexTopRow, indexRange.rowStart);
		long cellIndexOverlapBottom=Math.min(cellIndexBottomRow, indexRange.rowEnd);
		long cellIndexOverlapLeft=Math.max(cellIndexLeftCol, indexRange.colStart);
		long cellIndexOverlapRight=Math.min(cellIndexRightCol, indexRange.colEnd);
		
		if(cellIndexOverlapTop>cellIndexOverlapBottom || cellIndexOverlapLeft>cellIndexOverlapRight)
			return;
		
		tempRange.set(UtilFunctions.cellInBlockCalculation(cellIndexOverlapTop, blockRowFactor), 
				UtilFunctions.cellInBlockCalculation(cellIndexOverlapBottom, blockRowFactor), 
				UtilFunctions.cellInBlockCalculation(cellIndexOverlapLeft, blockColFactor), 
				UtilFunctions.cellInBlockCalculation(cellIndexOverlapRight, blockColFactor));
		
		int rowCut=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, blockRowFactor);
		int colCut=UtilFunctions.cellInBlockCalculation(indexRange.colStart, blockColFactor);
		
		int rowsInLastBlock=(int)((indexRange.rowEnd-indexRange.rowStart+1)%blockRowFactor);
		int colsInLastBlock=(int)((indexRange.colEnd-indexRange.colStart+1)%blockColFactor);
		
		long resultBlockIndexTop=UtilFunctions.blockIndexCalculation(cellIndexOverlapTop-indexRange.rowStart+1, blockRowFactor);
		long resultBlockIndexBottom=UtilFunctions.blockIndexCalculation(cellIndexOverlapBottom-indexRange.rowStart+1, blockRowFactor);
		long resultBlockIndexLeft=UtilFunctions.blockIndexCalculation(cellIndexOverlapLeft-indexRange.colStart+1, blockColFactor);
		long resultBlockIndexRight=UtilFunctions.blockIndexCalculation(cellIndexOverlapRight-indexRange.colStart+1, blockColFactor);
		
		int boundaryRlen=blockRowFactor, boundaryClen=blockColFactor;
		long finalBlockIndexBottom=UtilFunctions.blockIndexCalculation(indexRange.rowEnd-indexRange.rowStart+1, blockRowFactor);
		long finalBlockIndexRight=UtilFunctions.blockIndexCalculation(indexRange.colEnd-indexRange.colStart+1, blockColFactor);
		if(resultBlockIndexBottom==finalBlockIndexBottom)
			boundaryRlen=rowsInLastBlock;
		if(resultBlockIndexRight==finalBlockIndexRight)
			boundaryClen=colsInLastBlock;
			
		//allocate space for the output value
		ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(4);
		for(long r=resultBlockIndexTop; r<=resultBlockIndexBottom; r++)
			for(long c=resultBlockIndexLeft; c<=resultBlockIndexRight; c++)
			{
				IndexedMatrixValue out=cachedValues.holdPlace(output, valueClass);
				out.getIndexes().setIndexes(r, c);
				outlist.add(out);
			}
		
		//process instruction
		
		OperationsOnMatrixValues.performSlide(in.getIndexes(), in.getValue(), outlist, tempRange, rowCut, colCut, blockRowFactor, blockColFactor, boundaryRlen, boundaryClen);
	}
	
	public static void main(String[] args) throws Exception {
		
		byte input=1;
		byte output=2;
		RangeBasedReIndexInstruction ins=new RangeBasedReIndexInstruction(new ReIndexOperator(), input, output, new IndexRange(3, 18, 3, 18), "rangeReIndex");
		int blockRowFactor=10;
		int blockColFactor=10;
		
		MatrixBlockDSM m=MatrixBlockDSM.getRandomSparseMatrix(blockRowFactor, blockColFactor, 1, 1);
		m.examSparsity();
		CachedValueMap cachedValues=new CachedValueMap();
		cachedValues.set(input, new MatrixIndexes(2, 2), m);
		
		IndexedMatrixValue tempValue=new IndexedMatrixValue(MatrixBlockDSM.class);
		IndexedMatrixValue zeroInput=new IndexedMatrixValue(MatrixBlockDSM.class);
		
		ins.processInstruction(MatrixBlockDSM.class, cachedValues, tempValue, zeroInput, blockRowFactor, blockColFactor);
		System.out.println(cachedValues.get(output));
	}
}

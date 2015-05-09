/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
// import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.mr.RangeBasedReIndexInstruction.IndexRange;
import com.ibm.bi.dml.runtime.instructions.spark.functions.IsBlockInRange;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class MatrixIndexingSPInstruction  extends UnarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/*
	 * This class implements the matrix indexing functionality inside CP.  
	 * Example instructions: 
	 *     rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
	 *         input=mVar1, output=mVar6, 
	 *         bounds = (Var2,Var3,Var4,Var5)
	 *         rowindex_lower: Var2, rowindex_upper: Var3 
	 *         colindex_lower: Var4, colindex_upper: Var5
	 *     leftIndex:mVar1:mVar2:Var3:Var4:Var5:Var6:mVar7
	 *         triggered by "mVar1[Var3:Var4, Var5:Var6] = mVar2"
	 *         the result is stored in mVar7
	 *  
	 */
	protected CPOperand rowLower, rowUpper, colLower, colUpper;
	
	public MatrixIndexingSPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public MatrixIndexingSPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, CPOperand out, String opcode, String istr){
		super(op, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase("rangeReIndex") ) {
			if ( parts.length == 7 ) {
				// Example: rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				in.split(parts[1]);
				rl.split(parts[2]);
				ru.split(parts[3]);
				cl.split(parts[4]);
				cu.split(parts[5]);
				out.split(parts[6]);
				return new MatrixIndexingSPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase("leftIndex")) {
			if ( parts.length == 8 ) {
				// Example: leftIndex:mVar1:mvar2:Var3:Var4:Var5:Var6:mVar7
				CPOperand lhsInput, rhsInput, rl, ru, cl, cu, out;
				lhsInput = new CPOperand();
				rhsInput = new CPOperand();
				rl = new CPOperand();
				ru = new CPOperand();
				cl = new CPOperand();
				cu = new CPOperand();
				out = new CPOperand();
				lhsInput.split(parts[1]);
				rhsInput.split(parts[2]);
				rl.split(parts[3]);
				ru.split(parts[4]);
				cl.split(parts[5]);
				cu.split(parts[6]);
				out.split(parts[7]);
				return new MatrixIndexingSPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingSPInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();
		
		//get indexing range
		long rl = ec.getScalarInput(rowLower.getName(), rowLower.getValueType(), rowLower.isLiteral()).getLongValue();
		long ru = ec.getScalarInput(rowUpper.getName(), rowUpper.getValueType(), rowUpper.isLiteral()).getLongValue();
		long cl = ec.getScalarInput(colLower.getName(), colLower.getValueType(), colLower.isLiteral()).getLongValue();
		long cu = ec.getScalarInput(colUpper.getName(), colUpper.getValueType(), colUpper.isLiteral()).getLongValue();
		
		//right indexing
		if( opcode.equalsIgnoreCase("rangeReIndex") )
		{
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			MatrixCharacteristics mc = ec.getMatrixCharacteristics(input1.getName());
			if(!mcOut.dimsKnown()) {
				if(!mc.dimsKnown()) {
					throw new DMLRuntimeException("The output dimensions are not specified for MatrixIndexingSPInstruction");
				}
				else {
					mcOut.set(ru-rl+1, cu-cl+1, mc.getRowsPerBlock(), mc.getColsPerBlock());
				}
			}
			
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockedRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out 
				= in1
				.filter(new IsBlockInRange(rl, ru, cl, cu, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()))
				.flatMapToPair(new SliceBlock(rl, ru, cl, cu, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()))
				.groupByKey()
				.mapToPair(new MergeMiniBlocks(mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()));
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase("leftIndex"))
		{
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockedRDDHandleForVariable( input1.getName() );
			Broadcast<MatrixBlock> in2 = null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());

			if(!mcOut.dimsKnown()) {
				throw new DMLRuntimeException("The output dimensions are not specified for MatrixIndexingSPInstruction");
//				MatrixCharacteristics mc = ec.getMatrixCharacteristics(input1.getName());
//				if(!mc.dimsKnown()) {
//					throw new DMLRuntimeException("The output dimensions are not specified for MatrixIndexingSPInstruction");
//				}
//				else {
//					mcOut.set(ru-rl+1, cu-cl+1, mc.getRowsPerBlock(), mc.getColsPerBlock());
//				}
			}
			
			if(input2.getDataType() == DataType.MATRIX) //MATRIX<-MATRIX
			{
				MatrixCharacteristics mcLeft = ec.getMatrixCharacteristics(input1.getName());
				MatrixCharacteristics mcRight = ec.getMatrixCharacteristics(input2.getName());
				if(!mcLeft.dimsKnown() || !mcRight.dimsKnown()) {
					throw new DMLRuntimeException("The input matrix dimensions are not specified for MatrixIndexingSPInstruction");
				}
				
				// TODO: This assumes that RHS matrix is small.
				in2 = sec.getBroadcastForVariable( input2.getName() ); 
				out = in1.mapToPair(new LeftIndexBroadcastMatrix(in2, rl, ru, cl, cu, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(),
						mcLeft.getRows(), mcLeft.getCols(), mcRight.getRows(), mcRight.getCols()));
			}
			else //MATRIX<-SCALAR 
			{
				if(!(rl==ru && cl==cu))
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: ["+rl+":"+ru+","+cl+":"+cu+"]." );
				ScalarObject scalar = sec.getScalarInput(input2.getName(), ValueType.DOUBLE, input2.isLiteral());
				double scalarValue = 0;
				if(scalar instanceof DoubleObject) {
					scalarValue = scalar.getDoubleValue();
				}
				else {
					throw new DMLRuntimeException("Invalid valuetype of input parameter in MatrixIndexingSPInstruction");
				}
				out = in1.mapToPair(new LeftIndexScalar(scalarValue, rl, cl, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()));
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			if( in2 != null)
				sec.addLineageBroadcast(output.getName(), input2.getName());
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in MatrixIndexingSPInstruction.");		
	}
	
	public static class LeftIndexBroadcastMatrix implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 6201253001154384346L;
		
		long rl; long ru; long cl; long cu;
		int brlen; int bclen;
		Broadcast<MatrixBlock> binput;
		long left_rlen; long left_clen; long right_rlen; long right_clen;
		
		public LeftIndexBroadcastMatrix(Broadcast<MatrixBlock> binput, long rl, long ru, long cl, long cu, int brlen, int bclen,
				long left_rlen, long left_clen, long right_rlen, long right_clen) {
			this.rl = rl;
			this.ru = ru;
			this.cl = cl;
			this.cu = cu;
			this.brlen = brlen;
			this.bclen = bclen;
			this.binput = binput;
			this.left_rlen = left_rlen;
			this.left_clen = left_clen;
			this.right_rlen = right_rlen;
			this.right_clen = right_clen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			if(!(new IsBlockInRange(rl, ru, cl, cu, brlen, bclen)).call(kv)) {
				return kv;
			}
			
			long brIndex = kv._1.getRowIndex();
			long bcIndex = kv._1.getColumnIndex();
		
			long bRLowerIndex = (brIndex-1)*brlen + 1;
			long bRUpperIndex = brIndex*brlen;
			long bCLowerIndex = (bcIndex-1)*bclen + 1;
			long bCUpperIndex = bcIndex*bclen;
			
			// TODO: This logic needs to be further tested. it has passed sanity check of our integration test
			// Translate range [rl, ru, cl, cu] to LHS range to perform left indexing operation
			long lhs_rl = Math.max(rl, bRLowerIndex) - bRLowerIndex + 1;
			long lhs_ru = Math.min(ru, bRUpperIndex) - bRLowerIndex + 1;
			long lhs_cl = Math.max(cl, bCLowerIndex) - bCLowerIndex + 1;
			long lhs_cu = Math.min(cu, bCUpperIndex) - bCLowerIndex + 1;
			
			
			// Translate range [rl, ru, cl, cu] to RHS range to perform slice operation
			// Sliced RHS block needs to be between => Math.max(rl, bRLowerIndex), Math.max(cl, bCLowerIndex)
			// and Math.min(ru, bRUpperIndex), Math.min(cu, bCUpperIndex)
			long rhs_rl = Math.max(rl, bRLowerIndex) - rl + 1;
			long rhs_ru = Math.min(ru, bRUpperIndex) - rl + 1;
			long rhs_cl = Math.max(cl, bCLowerIndex) - cl + 1;
			long rhs_cu = Math.min(cu, bCUpperIndex) - cl + 1;
			
			MatrixBlock rhsMatBlock = binput.getValue();
			MatrixBlock slicedRHSMatBlock = (MatrixBlock) rhsMatBlock.sliceOperations(rhs_rl, rhs_ru, rhs_cl, rhs_cu, new MatrixBlock());
			
			MatrixBlock resultBlock = (MatrixBlock) kv._2.leftIndexingOperations(slicedRHSMatBlock, lhs_rl, lhs_ru, lhs_cl, lhs_cu, new MatrixBlock(), true);
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlock);
		}
		
	}
	
	public static class LeftIndexScalar implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {

		private static final long serialVersionUID = -57371729196313671L;
		private double scalarValue;
		long rl; long cl;
		int brlen; int bclen;
		
		public LeftIndexScalar(double scalarValue, long rl, long cl, int brlen, int bclen) {
			this.scalarValue = scalarValue;
			this.rl = rl;
			this.cl = cl;
			this.brlen = brlen;
			this.bclen = bclen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			long brIndex = kv._1.getRowIndex();
			long bcIndex = kv._1.getColumnIndex();
		
			long bRLowerIndex = (brIndex-1)*brlen + 1;
			long bRUpperIndex = brIndex*brlen;
			long bCLowerIndex = (bcIndex-1)*bclen + 1;
			long bCUpperIndex = bcIndex*bclen;
			
			boolean isBlockInRange = !(rl > bRUpperIndex || rl < bRLowerIndex || cl > bCUpperIndex || cl < bCLowerIndex);
			if(isBlockInRange) {
				ScalarObject scalar = new DoubleObject(scalarValue);
				long rowCellIndex = UtilFunctions.cellInBlockCalculation(rl, brlen) + 1; // Since leftIndexingOperations expects 1-based indexing
				long colCellIndex = UtilFunctions.cellInBlockCalculation(cl, bclen) + 1;
				MatrixBlock resultBlock = (MatrixBlock) kv._2.leftIndexingOperations(scalar, rowCellIndex, colCellIndex,  new MatrixBlock(), true);
				return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, resultBlock);
			}
			else {
				return kv;
			}
		}
		
	}
	
	public static class MergeMiniBlocks implements PairFunction<Tuple2<MatrixIndexes,Iterable<MatrixBlock>>, MatrixIndexes,MatrixBlock> {
		private static final long serialVersionUID = -6062101460171640670L;
		long brlen; long bclen;
		
		public MergeMiniBlocks(long brlen, long bclen) {
			this.brlen = brlen;
			this.bclen = bclen;
		}

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Iterable<MatrixBlock>> kv) throws Exception {
			if(!kv._2.iterator().hasNext()) {
				throw new Exception("Expected atleast one MatrixBlock while merging in MatrixIndexingSPInstruction");
			}
			MatrixBlock firstBlock = kv._2.iterator().next();
			
			MatrixBlock retVal = new MatrixBlock(firstBlock.getNumRows(), firstBlock.getNumColumns(), firstBlock.isInSparseFormat());
			for(MatrixBlock miniBlock : kv._2) {
				retVal.merge(miniBlock, true);
			}
			retVal.sortSparseRows();
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, retVal);
		}
		
	}
	
	public static class SliceBlock implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 580877584337511817L;
		
		long rl; long ru; long cl; long cu;
		int brlen; int bclen;
		
		public SliceBlock(long rl, long ru, long cl, long cu, int brlen, int bclen) {
			this.rl = rl;
			this.ru = ru;
			this.cl = cl;
			this.cu = cu;
			this.brlen = brlen;
			this.bclen = bclen;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			
			long cellIndexTopRow=UtilFunctions.cellIndexCalculation(kv._1.getRowIndex(), brlen, 0);
			long cellIndexBottomRow=UtilFunctions.cellIndexCalculation(kv._1.getRowIndex(), brlen, kv._2.getNumRows()-1);
			long cellIndexLeftCol=UtilFunctions.cellIndexCalculation(kv._1.getColumnIndex(), bclen, 0);
			long cellIndexRightCol=UtilFunctions.cellIndexCalculation(kv._1.getColumnIndex(), bclen, kv._2.getNumColumns()-1);
			
			IndexRange indexRange = new IndexRange(rl,ru,cl,cu);
			long cellIndexOverlapTop=Math.max(cellIndexTopRow, indexRange.rowStart);
			long cellIndexOverlapBottom=Math.min(cellIndexBottomRow, indexRange.rowEnd);
			long cellIndexOverlapLeft=Math.max(cellIndexLeftCol, indexRange.colStart);
			long cellIndexOverlapRight=Math.min(cellIndexRightCol, indexRange.colEnd);
			
			IndexRange tempRange = new IndexRange(rl,ru,cl,cu);
			tempRange.set(UtilFunctions.cellInBlockCalculation(cellIndexOverlapTop, brlen), 
					UtilFunctions.cellInBlockCalculation(cellIndexOverlapBottom, brlen), 
					UtilFunctions.cellInBlockCalculation(cellIndexOverlapLeft, bclen), 
					UtilFunctions.cellInBlockCalculation(cellIndexOverlapRight, bclen));
			
			int rowCut=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, brlen);
			int colCut=UtilFunctions.cellInBlockCalculation(indexRange.colStart, bclen);
			
			int rowsInLastBlock=(int)((indexRange.rowEnd-indexRange.rowStart+1)%brlen);
			if(rowsInLastBlock==0) rowsInLastBlock=brlen;
			int colsInLastBlock=(int)((indexRange.colEnd-indexRange.colStart+1)%bclen);
			if(colsInLastBlock==0) colsInLastBlock=bclen;

			long resultBlockIndexTop=UtilFunctions.blockIndexCalculation(cellIndexOverlapTop-indexRange.rowStart+1, brlen);
			long resultBlockIndexBottom=UtilFunctions.blockIndexCalculation(cellIndexOverlapBottom-indexRange.rowStart+1, brlen);
			long resultBlockIndexLeft=UtilFunctions.blockIndexCalculation(cellIndexOverlapLeft-indexRange.colStart+1, bclen);
			long resultBlockIndexRight=UtilFunctions.blockIndexCalculation(cellIndexOverlapRight-indexRange.colStart+1, bclen);

			int boundaryRlen=brlen, boundaryClen=bclen;
			long finalBlockIndexBottom=UtilFunctions.blockIndexCalculation(indexRange.rowEnd-indexRange.rowStart+1, brlen);
			long finalBlockIndexRight=UtilFunctions.blockIndexCalculation(indexRange.colEnd-indexRange.colStart+1, bclen);
			if(resultBlockIndexBottom==finalBlockIndexBottom)
				boundaryRlen=rowsInLastBlock;
			if(resultBlockIndexRight==finalBlockIndexRight)
				boundaryClen=colsInLastBlock;
				
			// allocate space for the output value
			ArrayList<IndexedMatrixValue> outlist=new ArrayList<IndexedMatrixValue>(4);
			for(long r=resultBlockIndexTop; r<=resultBlockIndexBottom; r++) {
				for(long c=resultBlockIndexLeft; c<=resultBlockIndexRight; c++) {
					IndexedMatrixValue out=new IndexedMatrixValue(new MatrixIndexes(r, c), new MatrixBlock());
					outlist.add(out);
				}
			}
			
			//process instruction
			OperationsOnMatrixValues.performSlice(kv._1, kv._2, outlist, tempRange, rowCut, colCut, brlen, bclen, boundaryRlen, boundaryClen);

			final ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			for(IndexedMatrixValue miniBlocks : outlist) {
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(miniBlocks.getIndexes(), (MatrixBlock) miniBlocks.getValue()));
			}
			
			return new Iterable<Tuple2<MatrixIndexes,MatrixBlock>>() {
				@Override
				public Iterator<Tuple2<MatrixIndexes, MatrixBlock>> iterator() {
					return retVal.iterator();
				}
			};
		}
		
	}
}

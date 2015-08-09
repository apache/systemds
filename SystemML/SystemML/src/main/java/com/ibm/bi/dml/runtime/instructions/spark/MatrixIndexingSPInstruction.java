/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.hops.AggBinaryOp.SparkAggType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.cp.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.spark.data.PartitionedMatrixBlock;
import com.ibm.bi.dml.runtime.instructions.spark.functions.IsBlockInRange;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;
import com.ibm.bi.dml.runtime.util.IndexRange;
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
	protected SparkAggType _aggType = null;
	
	public MatrixIndexingSPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, 
			                          CPOperand out, SparkAggType aggtype, String opcode, String istr){
		super(op, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;

		_aggType = aggtype;
	}
	
	public MatrixIndexingSPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu, 
			                          CPOperand out, String opcode, String istr){
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
			if ( parts.length == 8 ) {
				// Example: rangeReIndex:mVar1:Var2:Var3:Var4:Var5:mVar6
				CPOperand in = new CPOperand(parts[1]);
				CPOperand rl = new CPOperand(parts[2]);
				CPOperand ru = new CPOperand(parts[3]);
				CPOperand cl = new CPOperand(parts[4]);
				CPOperand cu = new CPOperand(parts[5]);
				CPOperand out = new CPOperand(parts[6]);
				SparkAggType aggtype = SparkAggType.valueOf(parts[7]);
				return new MatrixIndexingSPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, aggtype, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase("leftIndex") || opcode.equalsIgnoreCase("mapLeftIndex")) {
			if ( parts.length == 8 ) {
				// Example: leftIndex:mVar1:mvar2:Var3:Var4:Var5:Var6:mVar7
				CPOperand lhsInput = new CPOperand(parts[1]);
				CPOperand rhsInput = new CPOperand(parts[2]);
				CPOperand rl = new CPOperand(parts[3]);
				CPOperand ru = new CPOperand(parts[4]);
				CPOperand cl = new CPOperand(parts[5]);
				CPOperand cu = new CPOperand(parts[6]);
				CPOperand out = new CPOperand(parts[7]);
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
			//check and set output dimensions
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
			if(!mcOut.dimsKnown()) {
				if(!mc.dimsKnown())
					throw new DMLRuntimeException("The output dimensions are not specified for MatrixIndexingSPInstruction");
				mcOut.set(ru-rl+1, cu-cl+1, mc.getRowsPerBlock(), mc.getColsPerBlock());
			}
			
			//execute right indexing operation
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out =
					in1.filter(new IsBlockInRange(rl, ru, cl, cu, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()))
				       .flatMapToPair(new SliceBlock(rl, ru, cl, cu, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()));
			
			//aggregation if required 
			if( _aggType != SparkAggType.NONE )
				out = RDDAggregateUtils.mergeByKey(out);
				
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		//left indexing
		else if ( opcode.equalsIgnoreCase("leftIndex") || opcode.equalsIgnoreCase("mapLeftIndex"))
		{
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			Broadcast<PartitionedMatrixBlock> broadcastIn2 = null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> in2 = null;
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());

			if(!mcOut.dimsKnown()) {
				throw new DMLRuntimeException("The output dimensions are not specified for MatrixIndexingSPInstruction");
			}
			
			if(input2.getDataType() == DataType.MATRIX) //MATRIX<-MATRIX
			{
				MatrixCharacteristics mcLeft = ec.getMatrixCharacteristics(input1.getName());
				MatrixCharacteristics mcRight = ec.getMatrixCharacteristics(input2.getName());
				if(!mcLeft.dimsKnown() || !mcRight.dimsKnown()) {
					throw new DMLRuntimeException("The input matrix dimensions are not specified for MatrixIndexingSPInstruction");
				}
				
				if(opcode.equalsIgnoreCase("mapLeftIndex")) {
					broadcastIn2 = sec.getBroadcastForVariable( input2.getName() ); 
					out = in1
							.mapToPair(new LeftIndexBroadcastMatrix(broadcastIn2, rl, ru, cl, cu, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(),
							mcLeft.getRows(), mcLeft.getCols(), mcRight.getRows(), mcRight.getCols()));
				}
				else {
					// Zero-out LHS
					in1 = in1.mapToPair(new ZeroOutLHS(false, mcLeft.getRowsPerBlock(), 
									mcLeft.getColsPerBlock(), rl, ru, cl, cu));
					
					// Slice RHS to merge for LHS
					in2 = sec.getBinaryBlockRDDHandleForVariable( input2.getName() )
						    .flatMapToPair(new SliceRHSForLeftIndexing(rl, cl, mcLeft.getRowsPerBlock(), mcLeft.getColsPerBlock(), mcLeft.getRows(), mcLeft.getCols()));
					
					out = RDDAggregateUtils.mergeByKey(in1.union(in2));
				}
			}
			else //MATRIX<-SCALAR 
			{
				if(!(rl==ru && cl==cu))
					throw new DMLRuntimeException("Invalid index range of scalar leftindexing: ["+rl+":"+ru+","+cl+":"+cu+"]." );
				ScalarObject scalar = sec.getScalarInput(input2.getName(), ValueType.DOUBLE, input2.isLiteral());
				double scalarValue = scalar.getDoubleValue();
				
				out = in1.mapToPair(new LeftIndexScalar(scalarValue, rl, cl, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()));
			}
			
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
			if( broadcastIn2 != null)
				sec.addLineageBroadcast(output.getName(), input2.getName());
			if(in2 != null) 
				sec.addLineageRDD(output.getName(), input2.getName());
		}
		else
			throw new DMLRuntimeException("Invalid opcode (" + opcode +") encountered in MatrixIndexingSPInstruction.");		
	}
		
	public static class SliceRHSForLeftIndexing implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {

		private static final long serialVersionUID = -5292440389141307789L;
		private long rl; 
		private long cl; 
		private int brlen; 
		private int bclen;
		private long lhs_rlen;
		private long lhs_clen;
		
		public SliceRHSForLeftIndexing(long rl, long cl, int brlen, int bclen, long lhs_rlen, long lhs_clen) {
			this.rl = rl;
			this.cl = cl;
			this.brlen = brlen;
			this.bclen = bclen;
			this.lhs_rlen = lhs_rlen;
			this.lhs_clen = lhs_clen;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> rightKV) throws Exception {
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
	
			long start_lhs_globalRowIndex = rl + (rightKV._1.getRowIndex()-1)*brlen;
			long start_lhs_globalColIndex = cl + (rightKV._1.getColumnIndex()-1)*bclen;
			long end_lhs_globalRowIndex = start_lhs_globalRowIndex + rightKV._2.getNumRows() - 1;
			long end_lhs_globalColIndex = start_lhs_globalColIndex + rightKV._2.getNumColumns() - 1;
			
			long start_lhs_rowIndex = UtilFunctions.blockIndexCalculation(start_lhs_globalRowIndex, brlen);
			long end_lhs_rowIndex = UtilFunctions.blockIndexCalculation(end_lhs_globalRowIndex, brlen);
			long start_lhs_colIndex = UtilFunctions.blockIndexCalculation(start_lhs_globalColIndex, bclen);
			long end_lhs_colIndex = UtilFunctions.blockIndexCalculation(end_lhs_globalColIndex, bclen);
			
			for(long leftRowIndex = start_lhs_rowIndex; leftRowIndex <= end_lhs_rowIndex; leftRowIndex++) {
				for(long leftColIndex = start_lhs_colIndex; leftColIndex <= end_lhs_colIndex; leftColIndex++) {
					
					// Calculate global index of right hand side block
					long lhs_rl = Math.max((leftRowIndex-1)*brlen+1, start_lhs_globalRowIndex);
					long lhs_ru = Math.min(leftRowIndex*brlen, end_lhs_globalRowIndex);
					long lhs_cl = Math.max((leftColIndex-1)*bclen+1, start_lhs_globalColIndex);
					long lhs_cu = Math.min(leftColIndex*bclen, end_lhs_globalColIndex);
					
					int lhs_lrl = UtilFunctions.cellInBlockCalculation(lhs_rl, brlen) + 1;
					int lhs_lru = UtilFunctions.cellInBlockCalculation(lhs_ru, brlen) + 1;
					int lhs_lcl = UtilFunctions.cellInBlockCalculation(lhs_cl, bclen) + 1;
					int lhs_lcu = UtilFunctions.cellInBlockCalculation(lhs_cu, bclen) + 1;
					
					long rhs_rl = lhs_rl - rl + 1;
					long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
					long rhs_cl = lhs_cl - cl + 1;
					long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
					
					int rhs_lrl = UtilFunctions.cellInBlockCalculation(rhs_rl, brlen) + 1;
					int rhs_lru = UtilFunctions.cellInBlockCalculation(rhs_ru, brlen) + 1;
					int rhs_lcl = UtilFunctions.cellInBlockCalculation(rhs_cl, bclen) + 1;
					int rhs_lcu = UtilFunctions.cellInBlockCalculation(rhs_cu, bclen) + 1;
					
					MatrixBlock slicedRHSBlk = rightKV._2.sliceOperations(rhs_lrl, rhs_lru, rhs_lcl, rhs_lcu, new MatrixBlock());
					
					int lbrlen = UtilFunctions.computeBlockSize(lhs_rlen, leftRowIndex, brlen);
					int lbclen = UtilFunctions.computeBlockSize(lhs_clen, leftColIndex, bclen);
					MatrixBlock resultBlock = new MatrixBlock(lbrlen, lbclen, false);
					resultBlock = resultBlock.leftIndexingOperations(slicedRHSBlk, lhs_lrl, lhs_lru, lhs_lcl, lhs_lcu, null, true);
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(leftRowIndex, leftColIndex), resultBlock));
				}
			}
			return retVal;
		}
		
	}
	
	public static class ZeroOutLHS implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes,MatrixBlock> {

		private static final long serialVersionUID = 2100523151174865830L;
		boolean complementary = false;
		int brlen;
		int bclen;
		IndexRange indexRange;
		long rl; long ru; long cl; long cu;
		
		public ZeroOutLHS(boolean complementary, int brlen, int bclen, long rl, long ru, long cl, long cu) {
			this.complementary = complementary;
			this.brlen = brlen;
			this.bclen = bclen;
			this.rl = rl;
			this.ru = ru;
			this.cl = cl;
			this.cu = cu;
			this.indexRange = new IndexRange(rl, ru, cl, cu);
		}
		
		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(
				Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			if(!(new IsBlockInRange(rl, ru, cl, cu, brlen, bclen)).call(kv)) {
				return kv;
			}
			
			IndexRange range = UtilFunctions.getSelectedRangeForZeroOut(new IndexedMatrixValue(kv._1, kv._2), brlen, bclen, indexRange);
			if(range.rowStart == -1 && range.rowEnd == -1 && range.colStart == -1 && range.colEnd == -1) {
				throw new Exception("Error while getting range for zero-out");
				// return kv;
			}
			else {
				MatrixBlock zeroBlk = (MatrixBlock) kv._2.zeroOutOperations(new MatrixBlock(), range, complementary);
				return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, zeroBlk);
			}
		}
		
	}
	
	public static class LeftIndexBroadcastMatrix implements PairFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = 6201253001154384346L;
		
		long rl; long ru; long cl; long cu;
		int brlen; int bclen;
		Broadcast<PartitionedMatrixBlock> binput;
		long left_rlen; long left_clen; long right_rlen; long right_clen;
		
		public LeftIndexBroadcastMatrix(Broadcast<PartitionedMatrixBlock> binput, long rl, long ru, long cl, long cu, int brlen, int bclen,
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
			
			// Calculate global index of left hand side block
			long lhs_rl = Math.max(rl, (kv._1.getRowIndex()-1)*brlen + 1);
			long lhs_ru = Math.min(ru, kv._1.getRowIndex()*brlen);
			long lhs_cl = Math.max(cl, (kv._1.getColumnIndex()-1)*bclen + 1);
			long lhs_cu = Math.min(cu, kv._1.getColumnIndex()*bclen);
			
			// Calculate global index of right hand side block
			long rhs_rl = lhs_rl - rl + 1;
			long rhs_ru = rhs_rl + (lhs_ru - lhs_rl);
			long rhs_cl = lhs_cl - cl + 1;
			long rhs_cu = rhs_cl + (lhs_cu - lhs_cl);
			
			// Provide global one-based index to sliceOperations
			PartitionedMatrixBlock rhsMatBlock = binput.getValue();
			MatrixBlock slicedRHSMatBlock = rhsMatBlock.sliceOperations(rhs_rl, rhs_ru, rhs_cl, rhs_cu, new MatrixBlock());
			
			// Provide local one-based index to leftIndexingOperations
			long lhs_lrl = UtilFunctions.cellInBlockCalculation(lhs_rl, brlen) + 1;
			long lhs_lru = UtilFunctions.cellInBlockCalculation(lhs_ru, brlen) + 1;
			long lhs_lcl = UtilFunctions.cellInBlockCalculation(lhs_cl, bclen) + 1;
			long lhs_lcu = UtilFunctions.cellInBlockCalculation(lhs_cu, bclen) + 1;
			MatrixBlock resultBlock = kv._2.leftIndexingOperations(slicedRHSMatBlock, lhs_lrl, lhs_lru, lhs_lcl, lhs_lcu, new MatrixBlock(), false);
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
	
	/**
	 * 
	 */
	public static class SliceBlock implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 580877584337511817L;
		
		private long _rl; 
		private long _ru; 
		private long _cl; 
		private long _cu;
		private int _brlen; 
		private int _bclen;
		
		public SliceBlock(long rl, long ru, long cl, long cu, int brlen, int bclen) {
			_rl = rl;
			_ru = ru;
			_cl = cl;
			_cu = cu;
			_brlen = brlen;
			_bclen = bclen;
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			
			long cellIndexTopRow=UtilFunctions.cellIndexCalculation(kv._1.getRowIndex(), _brlen, 0);
			long cellIndexBottomRow=UtilFunctions.cellIndexCalculation(kv._1.getRowIndex(), _brlen, kv._2.getNumRows()-1);
			long cellIndexLeftCol=UtilFunctions.cellIndexCalculation(kv._1.getColumnIndex(), _bclen, 0);
			long cellIndexRightCol=UtilFunctions.cellIndexCalculation(kv._1.getColumnIndex(), _bclen, kv._2.getNumColumns()-1);
			
			IndexRange indexRange = new IndexRange(_rl,_ru,_cl,_cu);
			long cellIndexOverlapTop=Math.max(cellIndexTopRow, indexRange.rowStart);
			long cellIndexOverlapBottom=Math.min(cellIndexBottomRow, indexRange.rowEnd);
			long cellIndexOverlapLeft=Math.max(cellIndexLeftCol, indexRange.colStart);
			long cellIndexOverlapRight=Math.min(cellIndexRightCol, indexRange.colEnd);
			
			IndexRange tempRange = new IndexRange(_rl,_ru,_cl,_cu);
			tempRange.set(UtilFunctions.cellInBlockCalculation(cellIndexOverlapTop, _brlen), 
					UtilFunctions.cellInBlockCalculation(cellIndexOverlapBottom, _brlen), 
					UtilFunctions.cellInBlockCalculation(cellIndexOverlapLeft, _bclen), 
					UtilFunctions.cellInBlockCalculation(cellIndexOverlapRight, _bclen));
			
			int rowCut=UtilFunctions.cellInBlockCalculation(indexRange.rowStart, _brlen);
			int colCut=UtilFunctions.cellInBlockCalculation(indexRange.colStart, _bclen);
			
			int rowsInLastBlock=(int)((indexRange.rowEnd-indexRange.rowStart+1)%_brlen);
			int colsInLastBlock=(int)((indexRange.colEnd-indexRange.colStart+1)%_bclen);
			rowsInLastBlock = (rowsInLastBlock==0) ? _brlen : rowsInLastBlock;
			colsInLastBlock = (colsInLastBlock==0) ? _bclen : colsInLastBlock;

			long resultBlockIndexTop=UtilFunctions.blockIndexCalculation(cellIndexOverlapTop-indexRange.rowStart+1, _brlen);
			long resultBlockIndexBottom=UtilFunctions.blockIndexCalculation(cellIndexOverlapBottom-indexRange.rowStart+1, _brlen);
			long resultBlockIndexLeft=UtilFunctions.blockIndexCalculation(cellIndexOverlapLeft-indexRange.colStart+1, _bclen);
			long resultBlockIndexRight=UtilFunctions.blockIndexCalculation(cellIndexOverlapRight-indexRange.colStart+1, _bclen);

			int boundaryRlen=_brlen, boundaryClen=_bclen;
			long finalBlockIndexBottom=UtilFunctions.blockIndexCalculation(indexRange.rowEnd-indexRange.rowStart+1, _brlen);
			long finalBlockIndexRight=UtilFunctions.blockIndexCalculation(indexRange.colEnd-indexRange.colStart+1, _bclen);
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
			OperationsOnMatrixValues.performSlice(kv._1, kv._2, outlist, tempRange, rowCut, colCut, _brlen, _bclen, boundaryRlen, boundaryClen);

			final ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			for(IndexedMatrixValue miniBlocks : outlist) {
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(miniBlocks.getIndexes(), (MatrixBlock) miniBlocks.getValue()));
			}
			
			return retVal;
		}
		
	}
}

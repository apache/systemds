/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.functionobjects.DiagIndex;
import com.ibm.bi.dml.runtime.functionobjects.SortIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.FilterDiagBlocksFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.IsBlockInRange;
import com.ibm.bi.dml.runtime.instructions.spark.functions.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.instructions.spark.functions.RDDSortUtils;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ReorgMapFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class ReorgSPInstruction extends UnarySPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//sort-specific attributes (to enable variable attributes)
 	private CPOperand _col = null;
 	private CPOperand _desc = null;
 	private CPOperand _ixret = null;
	 	
	public ReorgSPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr){
		super(op, in, out, opcode, istr);
		_sptype = SPINSTRUCTION_TYPE.Reorg;
	}
	
	public ReorgSPInstruction(Operator op, CPOperand in, CPOperand col, CPOperand desc, CPOperand ixret, CPOperand out, String opcode, String istr){
		this(op, in, out, opcode, istr);
		_col = col;
		_desc = desc;
		_ixret = ixret;
		_sptype = SPINSTRUCTION_TYPE.Reorg;
	}
	
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String opcode = InstructionUtils.getOpCode(str);
		
		if ( opcode.equalsIgnoreCase("r'") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgSPInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("rdiag") ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgSPInstruction(new ReorgOperator(DiagIndex.getDiagIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase("rsort") ) {
			InstructionUtils.checkNumFields(str, 5);
			String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
			in.split(parts[1]);
			out.split(parts[5]);
			CPOperand col = new CPOperand(parts[2]);
			CPOperand desc = new CPOperand(parts[3]);
			CPOperand ixret = new CPOperand(parts[4]);
			
			return new ReorgSPInstruction(new ReorgOperator(SortIndex.getSortIndexFnObject(1,false,false)), 
					                      in, col, desc, ixret, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		String opcode = getOpcode();
		
		if( opcode.equalsIgnoreCase("r'") ) //TRANSPOSE
		{
			//get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );

			//execute transpose reorg operation
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapToPair(new ReorgMapFunction(opcode));
			
			//store output rdd handle
			updateReorgMatrixCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if ( opcode.equalsIgnoreCase("rdiag") ) // DIAG
		{			
			//update and get matrix characteristics
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			updateReorgMatrixCharacteristics(sec); //update mcOut
			
			// get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			if(mc1.getCols() == 1) { // diagV2M
				out = in1.flatMapToPair(new RDDDiagV2MFunction(mcOut));
			}
			else { // diagM2V
				//execute diagM2V operation
				out = in1.filter(new FilterDiagBlocksFunction())
					     .mapToPair(new ReorgMapFunction(opcode));
			}
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if ( opcode.equalsIgnoreCase("rsort") ) //ORDER
		{
			// Sort by column 'col' in ascending/descending order and return either index/value
			
			//get parameters
			long col = ec.getScalarInput(_col.getName(), _col.getValueType(), _col.isLiteral()).getLongValue();
			boolean desc = ec.getScalarInput(_desc.getName(), _desc.getValueType(), _desc.isLiteral()).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();
			
			// get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			
			//sort indexes 
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1
					.filter(new IsBlockInRange(1, mc1.getRows(), col, col, mc1.getRowsPerBlock(), mc1.getColsPerBlock()))
					.mapValues(new ExtractColumn((int)UtilFunctions.cellInBlockCalculation(col, mc1.getColsPerBlock())));
			out = RDDSortUtils.sortIndexesByVal(out, !desc, mc1.getRows(), mc1.getRowsPerBlock());
			
			//sort data if required
			if( !ixret ) 
			{
				// TODO: In first version, we will assume the the indexes can be materialized in the driver.
				// This means materializedSortedIndexes has 2GB limit
				// Later, this will be replaced by PMMJ
				
				MatrixBlock tmp = SparkExecutionContext.toMatrixBlock(out, (int)mc1.getRows(), 1, mc1.getRowsPerBlock(), mc1.getColsPerBlock());
				List<Double> materializedSortedIndexes = DataConverter.convertToDoubleList(tmp);
				
				out = in1.flatMapToPair(new ExtractSortedRows(materializedSortedIndexes, mc1.getRows(), mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock()));
				out = RDDAggregateUtils.mergeByKey(out);
			}
			
			//store output rdd handle
			updateReorgMatrixCharacteristics(sec);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else {
			throw new DMLRuntimeException("Error: Incorrect opcode in ReorgSPInstruction:" + opcode);
		}
	}
	
	/**
	 * 
	 * @param sec
	 * @throws DMLRuntimeException
	 */
	private void updateReorgMatrixCharacteristics(SparkExecutionContext sec) 
		throws DMLRuntimeException
	{
		MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		
		if( !mcOut.dimsKnown() ) 
		{
			if( !mc1.dimsKnown() )
				throw new DMLRuntimeException("Unable to compute output matrix characteristics from input.");
			
			if ( getOpcode().equalsIgnoreCase("r'") ) 
				mcOut.set(mc1.getCols(), mc1.getRows(), mc1.getColsPerBlock(), mc1.getRowsPerBlock());
			else if ( getOpcode().equalsIgnoreCase("rdiag") )
				mcOut.set(mc1.getRows(), (mc1.getCols()>1)?1:mc1.getRows(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			else if ( getOpcode().equalsIgnoreCase("rsort") ) {
				boolean ixret = sec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();
				mcOut.set(mc1.getRows(), ixret?1:mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
			}
		}
	}
	
	/**
	 * 
	 */
	private static class RDDDiagV2MFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 31065772250744103L;
		
		private ReorgOperator _reorgOp = null;
		private MatrixCharacteristics _mcOut = null;
		
		public RDDDiagV2MFunction(MatrixCharacteristics mcOut) 
			throws DMLRuntimeException 
		{
			_reorgOp = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
			_mcOut = mcOut;
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>();
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();
			
			//compute output indexes and reorg data
			long rix = ixIn.getRowIndex();
			MatrixIndexes ixOut = new MatrixIndexes(rix, rix);
			MatrixBlock blkOut = (MatrixBlock) blkIn.reorgOperations(_reorgOp, new MatrixBlock(), -1, -1, -1);
			ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut,blkOut));
			
			// insert newly created empty blocks for entire row
			int numBlocks = (int) Math.ceil((double)_mcOut.getCols()/_mcOut.getColsPerBlock());
			for(int i = 1; i <= numBlocks; i++) {
				if(i != ixOut.getColumnIndex()) {
					int lrlen = UtilFunctions.computeBlockSize(_mcOut.getRows(), rix, _mcOut.getRowsPerBlock());
		    		int lclen = UtilFunctions.computeBlockSize(_mcOut.getCols(), i, _mcOut.getColsPerBlock());
		    		MatrixBlock emptyBlk = new MatrixBlock(lrlen, lclen, true);
					ret.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(rix, i), emptyBlk));
				}
			}
			
			return ret;
		}
	}

	/**
	 *
	 */
	public static class ExtractColumn implements Function<MatrixBlock, MatrixBlock>  
	{
		private static final long serialVersionUID = -1472164797288449559L;
		
		private int _col;
		
		public ExtractColumn(int col) {
			_col = col;
		}
		
		@Override
		public MatrixBlock call(MatrixBlock arg0) 
			throws Exception 
		{
			return arg0.sliceOperations(1, arg0.getNumRows(), _col+1, _col+1, new MatrixBlock());
		}
	}

	
	//TODO MB we need a scalable implementation for this 
	public static class ExtractSortedRows implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = -938021325076154672L;
		private List<Double> sortedIndexes;
		int brlen; int bclen;
		long rlen; long clen;
		public ExtractSortedRows(List<Double> sortedIndexes, long rlen, long clen, int brlen, int bclen) {
			this.sortedIndexes = sortedIndexes;
			this.rlen = rlen;
			this.clen = clen;
			this.brlen = brlen;
			this.bclen = bclen;
		}
		
		private HashMap<Long, Long> efficientSortedIndex = null;
		private long getNewSortedRowID(long currentRowID) throws Exception {
			if(efficientSortedIndex == null) {
				efficientSortedIndex = new HashMap<Long, Long>(sortedIndexes.size());
				for(int i = 0; i < sortedIndexes.size(); i++) {
					efficientSortedIndex.put((long)Math.floor(sortedIndexes.get(i)), (long) i + 1);
				}
				sortedIndexes = null;
			}
			if(!efficientSortedIndex.containsKey(currentRowID)) {
				throw new Exception("The index " + currentRowID + " not found in sorted indexes.");
			}
			return efficientSortedIndex.get(currentRowID);
		}
		
		private long getRowBlockIndex(long globalRowIndex) {
			return UtilFunctions.blockIndexCalculation(globalRowIndex, (int) brlen);
		}
		
		private long getStartGlobalRowIndex(MatrixIndexes blockIndex) {
			return UtilFunctions.cellIndexCalculation(blockIndex.getRowIndex(), brlen, 0);
		}
		
		private long getEndGlobalRowIndex(MatrixIndexes blockIndex) {
			int new_lrlen = UtilFunctions.computeBlockSize(rlen, blockIndex.getRowIndex(), brlen);
			return UtilFunctions.cellIndexCalculation(blockIndex.getRowIndex(), brlen, new_lrlen-1);
		}
		
		private int getCellRowIndex(long globalCellIndex) {
			return UtilFunctions.cellInBlockCalculation(globalCellIndex, brlen);
		}

		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>(brlen);
			for(long i = getStartGlobalRowIndex(kv._1); i <= getEndGlobalRowIndex(kv._1); i++) {
				long newRowCellIndex = getNewSortedRowID(i);
				
				// Shift the row block index according to newRowCellIndex, but column block index remains same
				MatrixIndexes newIndex = new MatrixIndexes(getRowBlockIndex(newRowCellIndex), kv._1.getColumnIndex());
				
				int new_lrlen = UtilFunctions.computeBlockSize(rlen, newIndex.getRowIndex(), brlen);
				int new_lclen = UtilFunctions.computeBlockSize(clen, newIndex.getColumnIndex(), bclen);
				MatrixBlock extractedRowWithNewRowIndex = new MatrixBlock(new_lrlen, new_lclen, true);
				
				SparseRow row = new SparseRow(new_lclen);
				for( int j = 0; j < new_lclen; j++ ) {
					row.append(j, kv._2.getValue(getCellRowIndex(i), j));
				}
				extractedRowWithNewRowIndex.appendRow(getCellRowIndex(newRowCellIndex), row);
				retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(newIndex, extractedRowWithNewRowIndex));
			}
			
			// Since we are doing a reduceByKey which has an implicit combiner, no need to do merging here.
			
			return retVal;
		}
		
	}
	
}


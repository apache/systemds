/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
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
import com.ibm.bi.dml.runtime.functionobjects.IndexFunction;
import com.ibm.bi.dml.runtime.functionobjects.SortIndex;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertColumnRDDToBinaryBlock;
import com.ibm.bi.dml.runtime.instructions.spark.functions.IsBlockInRange;
import com.ibm.bi.dml.runtime.instructions.spark.functions.MergeBlocksFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ReorgMapFunction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;
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
			CPOperand col = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			CPOperand desc = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			CPOperand ixret = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
			col.split(parts[2]);
			desc.split(parts[3]);
			ixret.split(parts[4]);
			
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
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			if(!mcOut.dimsKnown()) {
				if(!mc1.dimsKnown()) {
					throw new DMLRuntimeException("The output dimensions are not specified for ReorgSPInstruction");
				}
				else {
					mcOut.set(mc1.getCols(), mc1.getRows(), mc1.getColsPerBlock(), mc1.getRowsPerBlock());
				}
			}
			
			//get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );

			//execute transpose reorg operation
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = in1.mapToPair(new ReorgMapFunction(opcode));
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if ( opcode.equalsIgnoreCase("rdiag") ) {
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			if(!mc1.dimsKnown()) {
				throw new DMLRuntimeException("Error: Dimensions unknown for instruction: rdiag");
			}
			DiagIndex fnObject = DiagIndex.getDiagIndexFnObject();
			fnObject.computeDimension(mc1, mcOut);
			
			// get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			if(mc1.getCols() == 1) { // diagV2M
				out = in1.flatMapToPair(new RDDV2MReorgMapFunction(mcOut.getRows(), mcOut.getCols(), mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()));
			}
			else { // diagM2V
				//execute diagM2V operation
				out = in1.filter(new RDDFilterM2VFunction()).mapToPair(new ReorgMapFunction("rdiag"));
			}
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if ( opcode.equalsIgnoreCase("rsort") ) {
			// Sort by column 'col' in ascending/descending order and return either index/value
			int col = (int)ec.getScalarInput(_col.getName(), _col.getValueType(), _col.isLiteral()).getLongValue();
			boolean desc = ec.getScalarInput(_desc.getName(), _desc.getValueType(), _desc.isLiteral()).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret.getName(), _ixret.getValueType(), _ixret.isLiteral()).getBooleanValue();
			
			MatrixCharacteristics mc1 = sec.getMatrixCharacteristics(input1.getName());
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			if(!mcOut.dimsKnown()) {
				if(!mc1.dimsKnown()) {
					throw new DMLRuntimeException("The output dimensions are not specified for ReorgSPInstruction");
				}
				else {
					if(ixret) {
						mcOut.set(mc1.getRows(), 1, mc1.getRowsPerBlock(), 1);
					}
					else {
						mcOut.set(mc1.getRows(), mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock());
					}
				}
			}
			// get input rdd handle
			JavaPairRDD<MatrixIndexes,MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable( input1.getName() );
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
			
			
			if(mc1.getRowsPerBlock() <= 0) {
				throw new DMLRuntimeException("Error: Incorrect block size:" + mc1.getRowsPerBlock());
			}
			
			// This shuffles all the entries everywhere but uses Spark's internal sorting functionality (i.e. Timsort)
			// TODO: For optimized version, we can sample and then create a Partitioner and then use repartitionAndSortWithinPartitions
			// Also keep an eye out for JIRA: https://issues.apache.org/jira/browse/SPARK-3461
			JavaRDD<Double> sortedIndexes = 
					in1.filter(new IsBlockInRange(1, mc1.getRows(), col, col, mc1.getRowsPerBlock(), mc1.getColsPerBlock()))
					.flatMapToPair(new ExtractColumn(col, mc1.getRowsPerBlock(), mc1.getRows()))
					.sortByKey(new IndexComparator(!desc), true)
					// .sortByKey(new IndexComparator(!desc), !desc)
					.values();
			
			if(ixret) {
				out = (new ConvertColumnRDDToBinaryBlock()).getBinaryBlockedRDD(sortedIndexes, mc1.getRowsPerBlock(), mc1.getRows(), sec);
			}
			else {
				// TODO: In first version, we will assume the the indexes can be materialized in the driver.
				// This means materializedSortedIndexes has 2GB limit
				// Later, this will be replaced by PMMJ
				List<Double> materializedSortedIndexes = sortedIndexes.collect();
				out = in1.flatMapToPair(new ExtractSortedRows(materializedSortedIndexes, mc1.getRows(), mc1.getCols(), mc1.getRowsPerBlock(), mc1.getColsPerBlock()))
						.reduceByKey(new MergeBlocksFunction());
			}
			
			//store output rdd handle
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else {
			throw new DMLRuntimeException("Error: Incorrect opcode in ReorgSPInstruction:" + opcode);
		}
	}
	
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
	
	public static class IndexComparator implements Comparator<IndexWithValue>, Serializable {
		private static final long serialVersionUID = 5154839870549241343L;
		boolean isAscending;
		public IndexComparator(boolean isAscending) {
			this.isAscending = isAscending;
		}
			
		@Override
		public int compare(IndexWithValue o1, IndexWithValue o2) {
			int retVal = o1.value.compareTo(o2.value);
			if(retVal != 0) {
				if(isAscending) {
					return retVal;
				}
				else {
					return -1 * retVal;
				}
			}
			else {
				// For stable sort
				return o1.globalBlockIndex.compareTo(o2.globalBlockIndex);
			}
		}
		
	}
	
	public static class IndexWithValue implements Serializable {
		private static final long serialVersionUID = -3273385845538526829L;
		public Double value; 
		public Long globalBlockIndex; 
		IndexWithValue(double value, long globalBlockIndex) {
			this.value = value;
			this.globalBlockIndex = globalBlockIndex;
		}
	}
	
	public static class ExtractColumn implements PairFlatMapFunction<Tuple2<MatrixIndexes,MatrixBlock>, IndexWithValue, Double>  {
		private static final long serialVersionUID = -1472164797288449559L;
		int col; int brlen; long rlen;
		public ExtractColumn(int col, int brlen, long rlen) {
			this.col = col;
			this.brlen = brlen;
			this.rlen = rlen;
		}
		@Override
		public Iterable<Tuple2<IndexWithValue, Double>> call(Tuple2<MatrixIndexes, MatrixBlock> kv) throws Exception {
			// ------------------------------------------------------------------
    		//	Compute local block size: 
    		// Example: For matrix: 1500 X 1100 with block length 1000 X 1000
    		// We will have four local block sizes (1000X1000, 1000X100, 500X1000 and 500X1000)
    		long blockRowIndex = kv._1.getRowIndex();
    		int lrlen = UtilFunctions.computeBlockSize(rlen, blockRowIndex, brlen);
    		// ------------------------------------------------------------------
			
			ArrayList<Tuple2<IndexWithValue, Double>> retVal = new ArrayList<Tuple2<IndexWithValue,Double>>(lrlen);
			long cellIndexTopRow = UtilFunctions.cellIndexCalculation(kv._1.getRowIndex(), brlen, 0);
			for(int i = 0; i < lrlen; i++) {
				double val = kv._2.getValue(i, col-1);
				long globalIndex = cellIndexTopRow + i;
				retVal.add(new Tuple2<IndexWithValue, Double>(new IndexWithValue(val, globalIndex), (double) globalIndex));
			}
			return retVal;
		}
		
	}
	
	public static class RDDFilterM2VFunction implements Function<Tuple2<MatrixIndexes,MatrixBlock>, Boolean> {

		private static final long serialVersionUID = -6928954547682014216L;

		@Override
		public Boolean call(Tuple2<MatrixIndexes, MatrixBlock> arg0) throws Exception {
			if(arg0._1.getRowIndex() == arg0._1.getColumnIndex()) {
				return true;
			}
			else {
				return false;
			}
		}
		
	}
	
	private static class RDDV2MReorgMapFunction implements PairFlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, MatrixIndexes, MatrixBlock> 
	{
		private static final long serialVersionUID = 31065772250744103L;
		
		private ReorgOperator _reorgOp = null;
		private IndexFunction _indexFnObject = null;
		long out_rlen; int out_brlen;
		long out_clen; int out_bclen;
		
		public RDDV2MReorgMapFunction(long rlen, long clen, int brlen, int bclen) throws DMLRuntimeException {
			_indexFnObject = DiagIndex.getDiagIndexFnObject();
			_reorgOp = new ReorgOperator(_indexFnObject);
			this.out_rlen = rlen;
			this.out_brlen = brlen;
			this.out_clen = clen;
			this.out_bclen = bclen;
		}
		
		@Override
		public Iterable<Tuple2<MatrixIndexes, MatrixBlock>> call( Tuple2<MatrixIndexes, MatrixBlock> arg0 ) 
			throws Exception 
		{
			int numBlocksPerRow = (int) Math.ceil((double)out_clen/out_bclen);
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> retVal = new ArrayList<Tuple2<MatrixIndexes,MatrixBlock>>(numBlocksPerRow);
			
			MatrixIndexes ixIn = arg0._1();
			MatrixBlock blkIn = arg0._2();

			//swap the matrix indexes
			MatrixIndexes ixOut = new MatrixIndexes(ixIn);
			_indexFnObject.execute(ixIn, ixOut);
			
			//swap the matrix block data
			MatrixBlock blkOut = (MatrixBlock) blkIn.reorgOperations(_reorgOp, new MatrixBlock(), -1, -1, -1);
			
			// insert the block obtained from reorg operation
			retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(ixOut,blkOut));
			
			// Now insert empty blocks
			for(int i = 1; i <= numBlocksPerRow; i++) {
				if(i != ixOut.getColumnIndex()) {
					// ------------------------------------------------------------------
		    		//	Compute local block size: 
		    		// Example: For matrix: 1500 X 1100 with block length 1000 X 1000
		    		// We will have four local block sizes (1000X1000, 1000X100, 500X1000 and 500X1000)
		    		long blockRowIndex = ixOut.getRowIndex();
		    		long blockColIndex = i;
		    		int lrlen = UtilFunctions.computeBlockSize(out_rlen, blockRowIndex, out_brlen);
		    		int lclen = UtilFunctions.computeBlockSize(out_clen, blockColIndex, out_bclen);
		    		// ------------------------------------------------------------------
		    		
					MatrixBlock emptyBlk = new MatrixBlock(lrlen, lclen, true);
					retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(new MatrixIndexes(blockRowIndex, blockColIndex), emptyBlk));
				}
			}
			
			return retVal;
		}
		
	}
}


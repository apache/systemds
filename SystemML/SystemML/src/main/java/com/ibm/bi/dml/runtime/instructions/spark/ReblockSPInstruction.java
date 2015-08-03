/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.data.RDDProperties;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertALToBinaryBlockFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertTextLineToBinaryCellFunction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ExtractBlockForBinaryReblock;
import com.ibm.bi.dml.runtime.instructions.spark.utils.SparkUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class ReblockSPInstruction extends UnarySPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private int brlen; private int bclen;
	private boolean outputEmptyBlocks;
	
	
	public ReblockSPInstruction(Operator op, CPOperand in, CPOperand out, int br, int bc, boolean emptyBlocks,
			String opcode, String instr) {
		super(op, in, out, opcode, instr);
		brlen=br;
		bclen=bc;
		outputEmptyBlocks = emptyBlocks;
	}
	
	public static Instruction parseInstruction(String str)  throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		if(opcode.compareTo("rblk") != 0) {
			throw new DMLRuntimeException("Incorrect opcode for ReblockSPInstruction:" + opcode);
		}
		
		// Example parts of ReblockSPInstruction: [rblk, pREADG·MATRIX·DOUBLE, _mVar1·MATRIX·DOUBLE, 1000, 1000, true]
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		in.split(parts[1]);
		out.split(parts[2]);
		int brlen=Integer.parseInt(parts[3]);
		int bclen=Integer.parseInt(parts[4]);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[5]);
		
		Operator op = null; // no operator for ReblockSPInstruction
		return new ReblockSPInstruction(op, in, out, brlen, bclen, outputEmptyBlocks, opcode, str);
	}
	
	public static JavaPairRDD<MatrixIndexes, MatrixBlock> processBinaryCellReblock(SparkExecutionContext sec, JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells,
			MatrixCharacteristics mc, MatrixCharacteristics mcOut, boolean outputEmptyBlocks,
			int brlen, int bclen) throws DMLRuntimeException {
		
		long numRows = -1;
		long numColumns = -1;
		if(!mcOut.dimsKnown() && !mc.dimsKnown()) {
			throw new DMLRuntimeException("Unknown dimensions while reblock into binary cell format");
		}
		else if(mc.dimsKnown()) {
			numRows = mc.getRows();
			numColumns = mc.getCols();
		}
		else {
			numRows = mcOut.getRows();
			numColumns = mcOut.getCols();
		}
		
		if(numRows <= 0 || numColumns <= 0) {
			throw new DMLRuntimeException("Error: Incorrect input dimensions:" + numRows + "," +  numColumns); 
		}
		
		// TODO: Investigate whether binaryCells.persist() will help here or not
		
		// ----------------------------------------------------------------------------
		// Now merge binary cells into binary blocks
		// Here you provide three "extremely light-weight" functions (that ignores sparsity):
		// 1. cell -> ArrayList (AL)
		// 2. (AL, cell) -> AL
		// 3. (AL, AL) -> AL
		// Then you convert the final AL -> binary blocks (here you take into account the sparsity).
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocksWithoutEmptyBlocks =
				binaryCells.combineByKey(
						new ConvertCellToALFunction(), 
						new AddCellToALFunction(), 
						new MergeALFunction())
						.mapToPair(new ConvertALToBinaryBlockFunction(brlen, bclen, numRows, numColumns));		
		// ----------------------------------------------------------------------------
		
		JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocksWithEmptyBlocks = null;
		if(outputEmptyBlocks) {
			binaryBlocksWithEmptyBlocks = SparkUtils.getRDDWithEmptyBlocks(sec, 
					binaryBlocksWithoutEmptyBlocks, numRows, numColumns, brlen, bclen);
		}
		else {
			binaryBlocksWithEmptyBlocks = binaryBlocksWithoutEmptyBlocks;
		}
		
		return binaryBlocksWithEmptyBlocks;
	}
	
	/**
	 * 
	 * @param sec
	 * @param lines
	 * @throws DMLRuntimeException
	 */
	public void processTextCellReblock(SparkExecutionContext sec, JavaPairRDD<LongWritable, Text> lines) 
		throws DMLRuntimeException 
	{
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		
		//check jdk version (prevent double.parseDouble contention on <jdk8)
		sec.checkAndRaiseValidationWarningJDKVersion();
		
		long numRows = -1;
		long numColumns = -1;
		if(!mcOut.dimsKnown() && !mc.dimsKnown()) {
			throw new DMLRuntimeException("Unknown dimensions in reblock instruction for text format");
		}
		else if(mc.dimsKnown()) {
			numRows = mc.getRows();
			numColumns = mc.getCols();
		}
		else {
			numRows = mcOut.getRows();
			numColumns = mcOut.getCols();
		}
		
		if(numRows <= 0 || numColumns <= 0) {
			throw new DMLRuntimeException("Error: Incorrect input dimensions:" + numRows + "," +  numColumns); 
		}
		
		JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells = 
				lines.mapToPair(new ConvertTextLineToBinaryCellFunction(brlen, bclen))
				.filter(new DropEmptyBinaryCells());
				
		
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = processBinaryCellReblock(sec, binaryCells, mc, mcOut, outputEmptyBlocks, brlen, bclen);
		
		//put output RDD handle into symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
	}

	@SuppressWarnings("unchecked")
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		
		MatrixObject mo = sec.getMatrixObject(input1.getName());
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown() && mc.dimsKnown()) {
			int brlen_out = mcOut.getRowsPerBlock();
			int bclen_out = mcOut.getColsPerBlock();
			// The number of rows and columns remains the same for input and output
			// However, the block size may vary: For example: global dataflow optimization
			mcOut.set(mc.getRows(), mc.getCols(), brlen_out, bclen_out);
		}
		if(mcOut.dimsKnown() && !mc.dimsKnown()) {
			int brlen_in = mc.getRowsPerBlock();
			int bclen_in = mc.getColsPerBlock();
			// The number of rows and columns remains the same for input and output
			// However, the block size may vary: For example: global dataflow optimization
			mc.set(mcOut.getRows(), mcOut.getCols(), brlen_in, bclen_in);
			// System.out.println("In Reblock, 2. Setting " + input1.getName() + " to " + mcOut.toString());
		}
		
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
		
		if(iimd == null) {
			throw new DMLRuntimeException("Error: Metadata not found");
		}
		
		if(iimd.getInputInfo() == InputInfo.TextCellInputInfo || iimd.getInputInfo() == InputInfo.MatrixMarketInputInfo ) {
			JavaPairRDD<LongWritable, Text> lines = (JavaPairRDD<LongWritable, Text>) sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
			processTextCellReblock(sec, lines);
		}
		else if(iimd.getInputInfo() == InputInfo.CSVInputInfo) {
			// HACK ALERT: Until we introduces the rewrite to insert csvrblock for non-persistent read
			// throw new DMLRuntimeException("CSVInputInfo is not supported for ReblockSPInstruction");
			RDDProperties properties = mo.getRddProperties();
			CSVReblockSPInstruction csvInstruction = null;
			boolean hasHeader = false;
			String delim = ",";
			boolean fill = false;
			double missingValue = 0;
			if(properties != null) {
				hasHeader = properties.isHasHeader();
				delim = properties.getDelim();
				fill = properties.isFill();
				missingValue = properties.getMissingValue();
			}
			csvInstruction = new CSVReblockSPInstruction(null, input1, output, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), hasHeader, delim, fill, missingValue, "csvrblk", instString);
			csvInstruction.processInstruction(sec);
			return;
		}
		else if(iimd.getInputInfo()==InputInfo.BinaryCellInputInfo) {
			JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells = (JavaPairRDD<MatrixIndexes, MatrixCell>) sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = processBinaryCellReblock(sec, binaryCells, mc, mcOut, outputEmptyBlocks, brlen, bclen);
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
		}
		else if(iimd.getInputInfo()==InputInfo.BinaryBlockInputInfo) {
			/// HACK ALERT: Workaround for MLContext 
			if(mc.getRowsPerBlock() == mcOut.getRowsPerBlock() && mc.getColsPerBlock() == mcOut.getColsPerBlock()) {
				if(mo.getRDDHandle() != null) {
					JavaPairRDD<MatrixIndexes, MatrixBlock> out = (JavaPairRDD<MatrixIndexes, MatrixBlock>) mo.getRDDHandle().getRDD();
					// SparkUtils.setLineageInfoForExplain(this, out, output.getName());
					sec.setRDDHandleForVariable(output.getName(), out);
					return;
				}
				else {
					throw new DMLRuntimeException("Input RDD is not accessible through buffer pool for ReblockSPInstruction:" + iimd.getInputInfo());
				}
			}
			else {
				// Reblocking to different sizes
				JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
				processBinaryReblock(sec, in1, mcOut.getRows(), mcOut.getCols(), 
						mc.getRowsPerBlock(), mc.getColsPerBlock(),
						mcOut.getRowsPerBlock(), mcOut.getColsPerBlock());
			}
		}
		else {
			throw new DMLRuntimeException("The given InputInfo is not implemented for ReblockSPInstruction:" + iimd.getInputInfo());
		}		
	}
	
	private void processBinaryReblock(SparkExecutionContext sec, JavaPairRDD<MatrixIndexes, MatrixBlock> in1,
			long rlen, long clen, int in_brlen, int in_bclen, int out_brlen, int out_bclen) throws DMLRuntimeException {
		JavaPairRDD<MatrixIndexes, MatrixBlock> out = in1.flatMapToPair(
				new ExtractBlockForBinaryReblock(rlen, clen, in_brlen, in_bclen, out_brlen, out_bclen))
				.groupByKey()
				.mapToPair(new MergeBinBlocks());
		
		// SparkUtils.setLineageInfoForExplain(this, out, output.getName());
		
		//put output RDD handle into symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
	}
	
	public static class MergeBinBlocks implements PairFunction<Tuple2<MatrixIndexes,Iterable<MatrixBlock>>, MatrixIndexes, MatrixBlock> {
		private static final long serialVersionUID = -5071035411478388254L;

		@Override
		public Tuple2<MatrixIndexes, MatrixBlock> call(Tuple2<MatrixIndexes, Iterable<MatrixBlock>> kv) throws Exception {
			int brlen = -1; int bclen = -1;
			long nnz = 0;
			for(MatrixBlock mb : kv._2) {
				if(brlen == -1) {
					brlen = mb.getNumRows();
					bclen = mb.getNumColumns();
				}
				mb.recomputeNonZeros();
				nnz += mb.getNonZeros();
			}
			boolean sparse = MatrixBlock.evalSparseFormatInMemory(brlen, bclen, nnz);
			MatrixBlock retVal = new MatrixBlock(brlen, bclen, sparse, nnz);
			for(MatrixBlock mb : kv._2) {
				retVal.merge(mb, false);
			}
			return new Tuple2<MatrixIndexes, MatrixBlock>(kv._1, retVal);
		}
		
	}
	
	
	
	// ====================================================================================================
	// Three functions passed to combineByKey
	
	public static class ConvertCellToALFunction implements Function<MatrixCell, ArrayList<MatrixCell>> {
		private static final long serialVersionUID = -2458721762929481811L;
		@Override
		public ArrayList<MatrixCell> call(MatrixCell cell) throws Exception {
			ArrayList<MatrixCell> retVal = new ArrayList<MatrixCell>();
			if(cell.getValue() != 0)
				retVal.add(cell);
			return retVal;
		}	
	}
	
	public static class AddCellToALFunction implements Function2<ArrayList<MatrixCell>, MatrixCell, ArrayList<MatrixCell>> {
		private static final long serialVersionUID = -4680403897867388102L;
		@Override
		public ArrayList<MatrixCell> call(ArrayList<MatrixCell> al, MatrixCell cell) throws Exception {
			al.add(cell);
			return al;
		}	
	}
	
	public static class MergeALFunction implements Function2<ArrayList<MatrixCell>, ArrayList<MatrixCell>, ArrayList<MatrixCell>> {
		private static final long serialVersionUID = -8117257799807223694L;
		@Override
		public ArrayList<MatrixCell> call(ArrayList<MatrixCell> al1, ArrayList<MatrixCell> al2) throws Exception {
			al1.addAll(al2);
			return al1;
		}	
	}
	// ====================================================================================================
	
	// This function gets called to check whether to drop binary cell corresponding to header of Matrix market format
	public static class DropEmptyBinaryCells implements Function<Tuple2<MatrixIndexes,MatrixCell>, Boolean> {
		private static final long serialVersionUID = -3672377410407066396L;
		
		@Override
		public Boolean call(Tuple2<MatrixIndexes, MatrixCell> arg0) throws Exception {
			if(arg0._1.getRowIndex() == -1) {
				return false; // Header cell for MatrixMarket format
			}
			else if(arg0._2.getValue() == 0) {
				return false; // empty cell: can be dropped as MatrixBlock can handle sparsity
			}
			return true;
		}
		
	}
}

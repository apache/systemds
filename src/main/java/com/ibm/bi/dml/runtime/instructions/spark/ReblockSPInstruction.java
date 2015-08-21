/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
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
import com.ibm.bi.dml.runtime.instructions.spark.functions.ExtractBlockForBinaryReblock;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDConverterUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class ReblockSPInstruction extends UnarySPInstruction 
{
	
	private int brlen; private int bclen;
	private boolean outputEmptyBlocks;
	
	public ReblockSPInstruction(Operator op, CPOperand in, CPOperand out, int br, int bc, boolean emptyBlocks,
			String opcode, String instr) 
	{
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
	

	@Override
	@SuppressWarnings("unchecked")
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
			//check jdk version (prevent double.parseDouble contention on <jdk8)
			sec.checkAndRaiseValidationWarningJDKVersion();
			JavaPairRDD<LongWritable, Text> lines = (JavaPairRDD<LongWritable, Text>) sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
			
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.textRDDToBinaryBlockRDD(lines, mc, mcOut, sec.getSparkContext(), brlen, bclen, outputEmptyBlocks);
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
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
		else if(iimd.getInputInfo()==InputInfo.BinaryCellInputInfo) 
		{
			JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells = (JavaPairRDD<MatrixIndexes, MatrixCell>) sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.binaryCellRDDToBinaryBlockRDD(binaryCells, mc, mcOut, sec.getSparkContext(), brlen, bclen, outputEmptyBlocks);
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(iimd.getInputInfo()==InputInfo.BinaryBlockInputInfo) 
		{
			/// HACK ALERT: Workaround for MLContext 
			if(mc.getRowsPerBlock() == mcOut.getRowsPerBlock() && mc.getColsPerBlock() == mcOut.getColsPerBlock()) {
				if(mo.getRDDHandle() != null) {
					JavaPairRDD<MatrixIndexes, MatrixBlock> out = (JavaPairRDD<MatrixIndexes, MatrixBlock>) mo.getRDDHandle().getRDD();
					
					//put output RDD handle into symbol table
					sec.setRDDHandleForVariable(output.getName(), out);
					sec.addLineageRDD(output.getName(), input1.getName());
					return;
				}
				else {
					throw new DMLRuntimeException("Input RDD is not accessible through buffer pool for ReblockSPInstruction:" + iimd.getInputInfo());
				}
			}
			else 
			{
				//BINARY BLOCK <- BINARY BLOCK (different sizes)
				
				JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
				
				JavaPairRDD<MatrixIndexes, MatrixBlock> out = in1.flatMapToPair(
						    new ExtractBlockForBinaryReblock(mcOut.getRows(), mcOut.getCols(), mc.getRowsPerBlock(), 
								mc.getColsPerBlock(),mcOut.getRowsPerBlock(), mcOut.getColsPerBlock()))
						.groupByKey()
						.mapToPair(new MergeBinBlocks());
				
				//put output RDD handle into symbol table
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), input1.getName());
			}
		}
		else {
			throw new DMLRuntimeException("The given InputInfo is not implemented for ReblockSPInstruction:" + iimd.getInputInfo());
		}		
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

}

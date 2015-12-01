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

import com.ibm.bi.dml.hops.recompile.Recompiler;
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
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
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
	private int brlen; 
	private int bclen;
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
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if(opcode.compareTo("rblk") != 0) {
			throw new DMLRuntimeException("Incorrect opcode for ReblockSPInstruction:" + opcode);
		}
		
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
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

		//set the output characteristics
		MatrixObject mo = sec.getMatrixObject(input1.getName());
		MatrixCharacteristics mc = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), brlen, bclen, mc.getNonZeros());

		//get the source format form the meta data
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
		if(iimd == null) {
			throw new DMLRuntimeException("Error: Metadata not found");
		}
		
		//check for in-memory reblock (w/ lazy spark context, potential for latency reduction)
		if( Recompiler.checkCPReblock(sec, input1.getName()) ) {
			Recompiler.executeInMemoryReblock(sec, input1.getName(), output.getName());
			return;
		}
		
		if(iimd.getInputInfo() == InputInfo.TextCellInputInfo || iimd.getInputInfo() == InputInfo.MatrixMarketInputInfo ) 
		{
			//check jdk version (prevent double.parseDouble contention on <jdk8)
			sec.checkAndRaiseValidationWarningJDKVersion();
			
			//get the input textcell rdd
			JavaPairRDD<LongWritable, Text> lines = (JavaPairRDD<LongWritable, Text>) 
					sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
			
			//convert textcell to binary block
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
					RDDConverterUtils.textCellToBinaryBlock(sec.getSparkContext(), lines, mcOut, outputEmptyBlocks);
			
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
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.binaryCellToBinaryBlock(sec.getSparkContext(), binaryCells, mcOut, outputEmptyBlocks);
			
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
				JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = sec.getBinaryBlockRDDHandleForVariable(input1.getName());
				
				JavaPairRDD<MatrixIndexes, MatrixBlock> out = 
						in1.flatMapToPair(new ExtractBlockForBinaryReblock(mc, mcOut));
				out = RDDAggregateUtils.mergeByKey( out );
				
				//put output RDD handle into symbol table
				sec.setRDDHandleForVariable(output.getName(), out);
				sec.addLineageRDD(output.getName(), input1.getName());
			}
		}
		else {
			throw new DMLRuntimeException("The given InputInfo is not implemented for ReblockSPInstruction:" + iimd.getInputInfo());
		}		
	}
}

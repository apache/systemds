/*
 * Modifications Copyright 2019 Graz University of Technology
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.tugraz.sysds.runtime.instructions.spark;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheableData;
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.spark.functions.ExtractBlockForBinaryReblock;
import org.tugraz.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDAggregateUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.tugraz.sysds.runtime.io.FileFormatPropertiesCSV;
import org.tugraz.sysds.runtime.io.FileFormatPropertiesMM;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixCell;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.utils.Statistics;

public class ReblockSPInstruction extends UnarySPInstruction {
	private int blen;
	private boolean outputEmptyBlocks;

	private ReblockSPInstruction(Operator op, CPOperand in, CPOperand out, int br, int bc, boolean emptyBlocks,
			String opcode, String instr) {
		super(SPType.Reblock, op, in, out, opcode, instr);
		blen = br;
		blen = bc;
		outputEmptyBlocks = emptyBlocks;
	}

	public static ReblockSPInstruction parseInstruction(String str) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if(!opcode.equals("rblk")) {
			throw new DMLRuntimeException("Incorrect opcode for ReblockSPInstruction:" + opcode);
		}
		
		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen=Integer.parseInt(parts[3]);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[5]);
		
		Operator op = null; // no operator for ReblockSPInstruction
		return new ReblockSPInstruction(op, in, out, blen, blen, outputEmptyBlocks, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;

		//set the output characteristics
		CacheableData<?> obj = sec.getCacheableData(input1.getName());
		DataCharacteristics mc = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), blen, mc.getNonZeros());
		
		//get the source format form the meta data
		MetaDataFormat iimd = (MetaDataFormat) obj.getMetaData();
		if(iimd == null)
			throw new DMLRuntimeException("Error: Metadata not found");
		InputInfo iinfo = iimd.getInputInfo();

		//check for in-memory reblock (w/ lazy spark context, potential for latency reduction)
		if( Recompiler.checkCPReblock(sec, input1.getName()) ) {
			if( input1.getDataType() == DataType.MATRIX )
				Recompiler.executeInMemoryMatrixReblock(sec, input1.getName(), output.getName());
			else if( input1.getDataType() == DataType.FRAME )
				Recompiler.executeInMemoryFrameReblock(sec, input1.getName(), output.getName());
			Statistics.decrementNoOfExecutedSPInst();
			return;
		}
		
		//execute matrix/frame reblock
		if( input1.getDataType() == DataType.MATRIX )
			processMatrixReblockInstruction(sec, iinfo);
		else if( input1.getDataType() == DataType.FRAME )
			processFrameReblockInstruction(sec, iinfo);
	}

	@SuppressWarnings("unchecked")
	protected void processMatrixReblockInstruction(SparkExecutionContext sec, InputInfo iinfo) {
		MatrixObject mo = sec.getMatrixObject(input1.getName());
		DataCharacteristics mc = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		
		if(iinfo == InputInfo.TextCellInputInfo || iinfo == InputInfo.MatrixMarketInputInfo ) {
			//get matrix market file properties if necessary
			FileFormatPropertiesMM mmProps = (iinfo == InputInfo.MatrixMarketInputInfo) ?
				IOUtilFunctions.readAndParseMatrixMarketHeader(mo.getFileName()) : null;
			
			//get the input textcell rdd
			JavaPairRDD<LongWritable, Text> lines = (JavaPairRDD<LongWritable, Text>)
				sec.getRDDHandleForMatrixObject(mo, iinfo);
			
			//convert textcell to binary block
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.textCellToBinaryBlock(
				sec.getSparkContext(), lines, mcOut, outputEmptyBlocks, mmProps);
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(iinfo == InputInfo.CSVInputInfo) {
			// HACK ALERT: Until we introduces the rewrite to insert csvrblock for non-persistent read
			// throw new DMLRuntimeException("CSVInputInfo is not supported for ReblockSPInstruction");
			CSVReblockSPInstruction csvInstruction = null;
			boolean hasHeader = false;
			String delim = ",";
			boolean fill = false;
			double fillValue = 0;
			if(mo.getFileFormatProperties() instanceof FileFormatPropertiesCSV
			   && mo.getFileFormatProperties() != null )
			{
				FileFormatPropertiesCSV props = (FileFormatPropertiesCSV) mo.getFileFormatProperties();
				hasHeader = props.hasHeader();
				delim = props.getDelim();
				fill = props.isFill();
				fillValue = props.getFillValue();
			}
			
			csvInstruction = new CSVReblockSPInstruction(null, input1, output, mcOut.getBlocksize(), mcOut.getBlocksize(), hasHeader, delim, fill, fillValue, "csvrblk", instString);
			csvInstruction.processInstruction(sec);
			return;
		}
		else if(iinfo == InputInfo.BinaryCellInputInfo)
		{
			JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells = (JavaPairRDD<MatrixIndexes, MatrixCell>) sec.getRDDHandleForMatrixObject(mo, iinfo);
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.binaryCellToBinaryBlock(sec.getSparkContext(), binaryCells, mcOut, outputEmptyBlocks);
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(iinfo == InputInfo.BinaryBlockInputInfo)
		{
			//BINARY BLOCK <- BINARY BLOCK (different sizes)
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());
			
			boolean shuffleFreeReblock = mc.dimsKnown() && mcOut.dimsKnown()
				&& (mc.getRows() < mcOut.getBlocksize() || mc.getBlocksize()%mcOut.getBlocksize() == 0)
				&& (mc.getCols() < mcOut.getBlocksize() || mc.getBlocksize()%mcOut.getBlocksize() == 0);
			
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = in1
				.flatMapToPair(new ExtractBlockForBinaryReblock(mc, mcOut));
			if( !shuffleFreeReblock )
				out = RDDAggregateUtils.mergeByKey(out, false);
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else {
			throw new DMLRuntimeException("The given InputInfo is not implemented "
					+ "for ReblockSPInstruction:" + InputInfo.inputInfoToString(iinfo));
		}
	}

	@SuppressWarnings("unchecked")
	protected void processFrameReblockInstruction(SparkExecutionContext sec, InputInfo iinfo)
	{
		FrameObject fo = sec.getFrameObject(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		
		if(iinfo == InputInfo.TextCellInputInfo ) {
			//get the input textcell rdd
			JavaPairRDD<LongWritable, Text> lines = (JavaPairRDD<LongWritable, Text>)
				sec.getRDDHandleForFrameObject(fo, iinfo);
			
			//convert textcell to binary block
			JavaPairRDD<Long, FrameBlock> out =
				FrameRDDConverterUtils.textCellToBinaryBlock(sec.getSparkContext(), lines, mcOut, fo.getSchema());
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(iinfo == InputInfo.CSVInputInfo) {
			// HACK ALERT: Until we introduces the rewrite to insert csvrblock for non-persistent read
			// throw new DMLRuntimeException("CSVInputInfo is not supported for ReblockSPInstruction");
			CSVReblockSPInstruction csvInstruction = null;
			boolean hasHeader = false;
			String delim = ",";
			boolean fill = false;
			double fillValue = 0;
			if(fo.getFileFormatProperties() instanceof FileFormatPropertiesCSV
				&& fo.getFileFormatProperties() != null )
			{
				FileFormatPropertiesCSV props = (FileFormatPropertiesCSV) fo.getFileFormatProperties();
				hasHeader = props.hasHeader();
				delim = props.getDelim();
				fill = props.isFill();
				fillValue = props.getFillValue();
			}
			
			csvInstruction = new CSVReblockSPInstruction(null, input1, output, mcOut.getBlocksize(), mcOut.getBlocksize(), hasHeader, delim, fill, fillValue, "csvrblk", instString);
			csvInstruction.processInstruction(sec);
		}
		else {
			throw new DMLRuntimeException("The given InputInfo is not implemented "
				+ "for ReblockSPInstruction: " + InputInfo.inputInfoToString(iinfo));
		}
	}
}

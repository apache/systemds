/*
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

package org.apache.sysds.runtime.instructions.spark;

import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FileFormatPropertiesLIBSVM;
import org.apache.sysds.runtime.io.FileFormatPropertiesMM;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixCell;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.utils.Statistics;

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

		if(!opcode.equals(Opcodes.RBLK.toString())) {
			throw new DMLRuntimeException("Incorrect opcode for ReblockSPInstruction:" + opcode);
		}

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen=Integer.parseInt(parts[3]);
		boolean outputEmptyBlocks = Boolean.parseBoolean(parts[4]);

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

		//get the source format from the meta data
		MetaDataFormat iimd = (MetaDataFormat) obj.getMetaData();
		if(iimd == null)
			throw new DMLRuntimeException("Error: Metadata not found");

		//check for in-memory reblock (w/ lazy spark context, potential for latency reduction)
		if( Recompiler.checkCPReblock(sec, input1.getName()) ) {
			if( input1.getDataType().isMatrix() || input1.getDataType().isFrame() ) {
				Recompiler.executeInMemoryReblock(sec, input1.getName(), output.getName(),
					iimd.getFileFormat()==FileFormat.BINARY ? getLineageItem(ec).getValue() : null);
			}
			Statistics.decrementNoOfExecutedSPInst();
			return;
		}

		//execute matrix/frame reblock
		if( input1.getDataType() == DataType.MATRIX )
			processMatrixReblockInstruction(sec, iimd.getFileFormat());
		else if(input1.getDataType() == DataType.FRAME)
			processFrameReblockInstruction(sec, iimd.getFileFormat());
	}

	@SuppressWarnings("unchecked")
	protected void processMatrixReblockInstruction(SparkExecutionContext sec, FileFormat fmt) {
		MatrixObject mo = sec.getMatrixObject(input1.getName());
		DataCharacteristics mc = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());

		if(fmt == FileFormat.TEXT || fmt == FileFormat.MM ) {
			//get matrix market file properties if necessary
			FileFormatPropertiesMM mmProps = (fmt == FileFormat.MM) ?
				IOUtilFunctions.readAndParseMatrixMarketHeader(mo.getFileName()) : null;

			//get the input textcell rdd
			JavaPairRDD<LongWritable, Text> lines = (JavaPairRDD<LongWritable, Text>)
				sec.getRDDHandleForMatrixObject(mo, fmt);

			//convert textcell to binary block
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.textCellToBinaryBlock(
				sec.getSparkContext(), lines, mcOut, outputEmptyBlocks, mmProps);

			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(fmt == FileFormat.CSV) {
			// HACK ALERT: Until we introduces the rewrite to insert csvrblock for non-persistent read
			// throw new DMLRuntimeException("CSVInputInfo is not supported for ReblockSPInstruction");
			CSVReblockSPInstruction csvInstruction = null;
			boolean hasHeader = false;
			String delim = ",";
			boolean fill = false;
			double fillValue = 0;
			Set<String> naStrings = null;
			if(mo.getFileFormatProperties() instanceof FileFormatPropertiesCSV
			   && mo.getFileFormatProperties() != null )
			{
				FileFormatPropertiesCSV props = (FileFormatPropertiesCSV) mo.getFileFormatProperties();
				hasHeader = props.hasHeader();
				delim = props.getDelim();
				fill = props.isFill();
				fillValue = props.getFillValue();
				naStrings = props.getNAStrings();
			}

			csvInstruction = new CSVReblockSPInstruction(null, input1, output, mcOut.getBlocksize(), mcOut.getBlocksize(), hasHeader, delim, fill, fillValue, Opcodes.CSVRBLK.toString(), instString, naStrings);
			csvInstruction.processInstruction(sec);
			return;
		}
		else if(fmt == FileFormat.BINARY && mc.getBlocksize() <= 0) {
			//BINARY BLOCK <- BINARY CELL (e.g., after grouped aggregate)
			JavaPairRDD<MatrixIndexes, MatrixCell> binaryCells = (JavaPairRDD<MatrixIndexes, MatrixCell>) sec.getRDDHandleForMatrixObject(mo, FileFormat.BINARY);
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.binaryCellToBinaryBlock(sec.getSparkContext(), binaryCells, mcOut, outputEmptyBlocks);

			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(fmt == FileFormat.BINARY) {
			//BINARY BLOCK <- BINARY BLOCK (different sizes)
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = sec.getBinaryMatrixBlockRDDHandleForVariable(input1.getName());
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.binaryBlockToBinaryBlock(in1, mc, mcOut);
			
			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(fmt == FileFormat.LIBSVM) {
			String delim = IOUtilFunctions.LIBSVM_DELIM;
			String indexDelim = IOUtilFunctions.LIBSVM_INDEX_DELIM;
			if(mo.getFileFormatProperties() instanceof FileFormatPropertiesLIBSVM && mo
				.getFileFormatProperties() != null) {
				FileFormatPropertiesLIBSVM props = (FileFormatPropertiesLIBSVM) mo.getFileFormatProperties();
				delim = props.getDelim();
				indexDelim = props.getIndexDelim();
			}

			LIBSVMReblockSPInstruction libsvmInstruction = new LIBSVMReblockSPInstruction(null, input1, output,
				mcOut.getBlocksize(), mcOut.getBlocksize(), "libsvmblk", delim, indexDelim, instString);
			libsvmInstruction.processInstruction(sec);
		}
		else if(fmt == FileFormat.COMPRESSED){
			JavaPairRDD<MatrixIndexes, MatrixBlock> in1 = (JavaPairRDD<MatrixIndexes, MatrixBlock>) sec
				.getRDDHandleForMatrixObject(mo, FileFormat.COMPRESSED);
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = RDDConverterUtils.binaryBlockToBinaryBlock(in1, mc, mcOut);
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else {
			throw new DMLRuntimeException("The given format is not implemented "
				+ "for ReblockSPInstruction:" + fmt.toString());
		}
	}

	@SuppressWarnings("unchecked")
	protected void processFrameReblockInstruction(SparkExecutionContext sec, FileFormat fmt)
	{
		FrameObject fo = sec.getFrameObject(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());

		if(fmt == FileFormat.TEXT) {
			//get the input textcell rdd
			JavaPairRDD<LongWritable, Text> lines = (JavaPairRDD<LongWritable, Text>)
				sec.getRDDHandleForFrameObject(fo, fmt);

			//convert textcell to binary block
			JavaPairRDD<Long, FrameBlock> out =
				FrameRDDConverterUtils.textCellToBinaryBlock(sec.getSparkContext(), lines, mcOut, fo.getSchema());

			//put output RDD handle into symbol table
			sec.setRDDHandleForVariable(output.getName(), out);
			sec.addLineageRDD(output.getName(), input1.getName());
		}
		else if(fmt == FileFormat.CSV) {
			// HACK ALERT: Until we introduces the rewrite to insert csvrblock for non-persistent read
			// throw new DMLRuntimeException("CSVInputInfo is not supported for ReblockSPInstruction");
			CSVReblockSPInstruction csvInstruction = null;
			boolean hasHeader = false;
			String delim = ",";
			boolean fill = false;
			double fillValue = 0;
			Set<String> naStrings = null;
			if(fo.getFileFormatProperties() instanceof FileFormatPropertiesCSV
				&& fo.getFileFormatProperties() != null )
			{
				FileFormatPropertiesCSV props = (FileFormatPropertiesCSV) fo.getFileFormatProperties();
				hasHeader = props.hasHeader();
				delim = props.getDelim();
				fill = props.isFill();
				fillValue = props.getFillValue();
				naStrings = props.getNAStrings();
			}

			csvInstruction = new CSVReblockSPInstruction(null, input1, output, mcOut.getBlocksize(), mcOut.getBlocksize(), hasHeader, delim, fill, fillValue, Opcodes.CSVRBLK.toString(), instString, naStrings);
			csvInstruction.processInstruction(sec);
		}
		else if(fmt == FileFormat.LIBSVM) {
			String delim = IOUtilFunctions.LIBSVM_DELIM;
			String indexDelim = IOUtilFunctions.LIBSVM_INDEX_DELIM;
			if(fo.getFileFormatProperties() instanceof FileFormatPropertiesLIBSVM && fo
				.getFileFormatProperties() != null) {
				FileFormatPropertiesLIBSVM props = (FileFormatPropertiesLIBSVM) fo.getFileFormatProperties();
				delim = props.getDelim();
				indexDelim = props.getIndexDelim();
			}
			LIBSVMReblockSPInstruction libsvmInstruction = new LIBSVMReblockSPInstruction(null, input1, output,
				mcOut.getBlocksize(), mcOut.getBlocksize(), "libsvmblk", delim, indexDelim, instString);
			libsvmInstruction.processInstruction(sec);

		}

		else {
			throw new DMLRuntimeException("The given format is not implemented "
				+ "for ReblockSPInstruction: " + fmt.toString());
		}
	}
	
	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		//construct reblock lineage without existing createvar lineage
		if( ec.getLineage() == null ) {
			return Pair.of(output.getName(), new LineageItem(
				ProgramConverter.serializeDataObject(input1.getName(), ec.getCacheableData(input1)), "cache_rblk"));
		}
		//default reblock w/ active lineage tracing
		return super.getLineageItem(ec);
	}

	public int getBlockLength() {
		return blen;
	}

	public boolean getOutputEmptyBlocks() {
		return outputEmptyBlocks;
	}
}

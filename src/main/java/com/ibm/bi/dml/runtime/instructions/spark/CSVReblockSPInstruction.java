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

import java.util.HashMap;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

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
import com.ibm.bi.dml.runtime.instructions.spark.data.CountLinesInfo;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertCSVLinesToMatrixBlocks;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertStringToText;
import com.ibm.bi.dml.runtime.instructions.spark.functions.CountLines;
import com.ibm.bi.dml.runtime.instructions.spark.utils.RDDAggregateUtils;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class CSVReblockSPInstruction extends UnarySPInstruction {

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n"
			+ "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private int brlen;
	private int bclen;
	private boolean hasHeader;
	private String delim;
	private boolean fill;
	private double missingValue;

	public CSVReblockSPInstruction(Operator op, CPOperand in, CPOperand out,
			int br, int bc, boolean hasHeader, String delim, boolean fill,
			double missingValue, String opcode, String instr) {
		super(op, in, out, opcode, instr);
		brlen = br;
		bclen = bc;
		this.hasHeader = hasHeader;
		this.delim = delim;
		this.fill = fill;
		this.missingValue = missingValue;
	}

	public static Instruction parseInstruction(String str)
			throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		if (opcode.compareTo("csvrblk") != 0) {
			throw new DMLRuntimeException(
					"Incorrect opcode for CSVReblockSPInstruction:" + opcode);
		}

		// Example parts of CSVReblockSPInstruction:
		// [csvrblk, pREADmissing_val_maps·MATRIX·DOUBLE, _mVar37·MATRIX·DOUBLE,
		// 1000, 1000, false, ,, true, 0.0]
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		in.split(parts[1]);
		out.split(parts[2]);

		int brlen = Integer.parseInt(parts[3]);
		int bclen = Integer.parseInt(parts[4]);

		boolean hasHeader = Boolean.parseBoolean(parts[5]);
		String delim = parts[6];
		boolean fill = Boolean.parseBoolean(parts[7]);
		double missingValue = Double.parseDouble(parts[8]);

		Operator op = null; // no operator for ReblockSPInstruction
		return new CSVReblockSPInstruction(op, in, out, brlen, bclen,
				hasHeader, delim, fill, missingValue, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		//sanity check input info
		MatrixObject mo = sec.getMatrixObject(input1.getName());
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
		if (iimd.getInputInfo() != InputInfo.CSVInputInfo) {
			throw new DMLRuntimeException("The given InputInfo is not implemented for ReblockSPInstruction:"
							+ iimd.getInputInfo());
		}
		
		//check jdk version (prevent double.parseDouble contention on <jdk8)
		sec.checkAndRaiseValidationWarningJDKVersion();
		
		@SuppressWarnings("unchecked")
		JavaPairRDD<LongWritable, Text> csvLines1 = (JavaPairRDD<LongWritable, Text>) sec.getRDDHandleForVariable(input1.getName(), iimd.getInputInfo());
		JavaRDD<String> csvLines = csvLines1.values().map(new ConvertStringToText());
		
		// Since all instructions should read directly from RDD rather than file,
		// changed this logic
		// String fileName = mo.getFileName();
		// JavaRDD<String> csvLines = sec.getSparkContext().textFile(fileName);

		// Compute (if not already computed) the start offset of each
		// partition of our input,
		// RDD, so that we can parse all the partitions in parallel and send
		// each chunk of
		// the matrix to the appropriate block.
		getRowOffsets(csvLines, delim);
		
		// put output RDD handle into symbol table
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		if(!mcOut.dimsKnown()) {
			mcOut.set(numRows, expectedNumColumns, brlen, bclen);
		}
					
		JavaPairRDD<MatrixIndexes, MatrixBlock> chunks = JavaPairRDD.fromJavaRDD(csvLines.mapPartitionsWithIndex(
						new ConvertCSVLinesToMatrixBlocks(rowOffsets, 
								mcOut.getRows(), mcOut.getCols(), mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), 
								hasHeader, delim, fill, missingValue), true));

		// Merge chunks according to their block index
		JavaPairRDD<MatrixIndexes, MatrixBlock> blocksRDD = RDDAggregateUtils.mergeByKey(chunks);
		
		// SparkUtils.setLineageInfoForExplain(this, blocksRDD, output.getName());
		sec.setRDDHandleForVariable(output.getName(), blocksRDD);
		sec.addLineageRDD(output.getName(), input1.getName());
	}
	
	private long expectedNumColumns = -1;
	private long numRows = 0;
	private HashMap<Integer, Long> rowOffsets = null;

	private void getRowOffsets(JavaRDD<String> csvLines, String delim) throws DMLRuntimeException {
		if(rowOffsets == null) {
			// Start by counting the number of lines in each partition.
			JavaRDD<Tuple2<Integer, CountLinesInfo>> lineCounts = csvLines
					.mapPartitionsWithIndex(new CountLines(delim), true);
	
			// Not sure if the sort here is necessary.
			List<Tuple2<Integer, CountLinesInfo>> linesPerPartition = JavaPairRDD
					.fromJavaRDD(lineCounts).sortByKey().collect();
			// lineCounts.sortBy((p: (Int, Long)) => p._1, true, 1).collect()
	
			if(linesPerPartition.size() == 0) {
				throw new DMLRuntimeException("Expected atleast one partition for the CSV input file");
			}
			
			// Compute the offset of the first line in the each partition.
			// This code assumes that partitions are numbered in order, but does
			// not assume that
			// partition numbers are contiguous
			this.rowOffsets = new HashMap<Integer, Long>();
			rowOffsets.put(linesPerPartition.get(0)._1, 0L);
	
			int prevPartNo = linesPerPartition.get(0)._1;
			for (int i = 1; i < linesPerPartition.size(); i++) {
				Integer partNo = linesPerPartition.get(i)._1;
				Long prevOffset = rowOffsets.get(prevPartNo);
				CountLinesInfo info = linesPerPartition.get(i - 1)._2;
				long curOffset = prevOffset + info.getNumLines();
				expectedNumColumns = Math.max(expectedNumColumns, info.getExpectedNumColumns());
				numRows += info.getNumLines();
				rowOffsets.put(partNo, curOffset);
				prevPartNo = partNo;
			}
			CountLinesInfo lastInfo = linesPerPartition.get(linesPerPartition.size() - 1)._2;
			expectedNumColumns = Math.max(expectedNumColumns, lastInfo.getExpectedNumColumns());
			numRows += lastInfo.getNumLines();
		}
	}
}

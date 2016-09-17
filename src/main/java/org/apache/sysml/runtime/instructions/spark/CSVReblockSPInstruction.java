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

package org.apache.sysml.runtime.instructions.spark;

import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.operators.Operator;

public class CSVReblockSPInstruction extends UnarySPInstruction 
{
	private int _brlen;
	private int _bclen;
	private boolean _hasHeader;
	private String _delim;
	private boolean _fill;
	private double _fillValue;

	public CSVReblockSPInstruction(Operator op, CPOperand in, CPOperand out,
			int br, int bc, boolean hasHeader, String delim, boolean fill,
			double fillValue, String opcode, String instr) 
	{
		super(op, in, out, opcode, instr);
		_brlen = br;
		_bclen = bc;
		_hasHeader = hasHeader;
		_delim = delim;
		_fill = fill;
		_fillValue = fillValue;
	}

	public static CSVReblockSPInstruction parseInstruction(String str)
			throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		if( !opcode.equals("csvrblk") )
			throw new DMLRuntimeException("Incorrect opcode for CSVReblockSPInstruction:" + opcode);

		// Example parts of CSVReblockSPInstruction:
		// [csvrblk, pREADmissing_val_maps·MATRIX·DOUBLE, _mVar37·MATRIX·DOUBLE,
		// 1000, 1000, false, ,, true, 0.0]
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int brlen = Integer.parseInt(parts[3]);
		int bclen = Integer.parseInt(parts[4]);
		boolean hasHeader = Boolean.parseBoolean(parts[5]);
		String delim = parts[6];
		boolean fill = Boolean.parseBoolean(parts[7]);
		double fillValue = Double.parseDouble(parts[8]);

		return new CSVReblockSPInstruction(null, in, out, brlen, bclen,
				hasHeader, delim, fill, fillValue, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		//sanity check input info
		CacheableData<?> obj = sec.getCacheableData(input1.getName());
		MatrixFormatMetaData iimd = (MatrixFormatMetaData) obj.getMetaData();
		if (iimd.getInputInfo() != InputInfo.CSVInputInfo) {
			throw new DMLRuntimeException("The given InputInfo is not implemented for "
					+ "CSVReblockSPInstruction:" + iimd.getInputInfo());
		}
		
		//set output characteristics
		MatrixCharacteristics mcIn = sec.getMatrixCharacteristics(input1.getName());
		MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
		mcOut.set(mcIn.getRows(), mcIn.getCols(), _brlen, _bclen);

		//check for in-memory reblock (w/ lazy spark context, potential for latency reduction)
		if( Recompiler.checkCPReblock(sec, input1.getName()) ) {
			if( input1.getDataType() == DataType.MATRIX )
				Recompiler.executeInMemoryMatrixReblock(sec, input1.getName(), output.getName());
			else if( input1.getDataType() == DataType.FRAME )
				Recompiler.executeInMemoryFrameReblock(sec, input1.getName(), output.getName());
			return;
		}
		
		//check jdk version (prevent double.parseDouble contention on <jdk8)
		sec.checkAndRaiseValidationWarningJDKVersion();
		
		//execute matrix/frame csvreblock 
		JavaPairRDD<?,?> out = null;
		if( input1.getDataType() == DataType.MATRIX )
			out = processMatrixCSVReblockInstruction(sec, mcOut);
		else if( input1.getDataType() == DataType.FRAME )
			out = processFrameCSVReblockInstruction(sec, mcOut, ((FrameObject)obj).getSchema());
		
		// put output RDD handle into symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
	}
	
	/**
	 * 
	 * @param sec
	 * @param mcOut
	 * @return
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("unchecked")
	protected JavaPairRDD<MatrixIndexes,MatrixBlock> processMatrixCSVReblockInstruction(SparkExecutionContext sec, MatrixCharacteristics mcOut) 
		throws DMLRuntimeException
	{
		//get input rdd (needs to be longwritable/text for consistency with meta data, in case of
		//serialization issues create longwritableser/textser as serializable wrappers
		JavaPairRDD<LongWritable, Text> in = (JavaPairRDD<LongWritable, Text>) 
				sec.getRDDHandleForVariable(input1.getName(), InputInfo.CSVInputInfo);
			
		//reblock csv to binary block
		return RDDConverterUtils.csvToBinaryBlock(sec.getSparkContext(), 
				in, mcOut, _hasHeader, _delim, _fill, _fillValue);
	}
	
	/**
	 * 
	 * @param sec
	 * @param mcOut
	 * @return
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("unchecked")
	protected JavaPairRDD<Long,FrameBlock> processFrameCSVReblockInstruction(SparkExecutionContext sec, MatrixCharacteristics mcOut, List<ValueType> schema) 
		throws DMLRuntimeException
	{
		//get input rdd (needs to be longwritable/text for consistency with meta data, in case of
		//serialization issues create longwritableser/textser as serializable wrappers
		JavaPairRDD<LongWritable, Text> in = (JavaPairRDD<LongWritable, Text>) 
				sec.getRDDHandleForVariable(input1.getName(), InputInfo.CSVInputInfo);
		
		//reblock csv to binary block
		return FrameRDDConverterUtils.csvToBinaryBlock(sec.getSparkContext(), 
				in, mcOut, schema, _hasHeader, _delim, _fill, _fillValue);
	}
}

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

import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.utils.Statistics;

public class CSVReblockSPInstruction extends UnarySPInstruction {
	
	private int _blen;
	private boolean _hasHeader;
	private String _delim;
	private boolean _fill;
	private double _fillValue;
	private Set<String> _naStrings;

	protected CSVReblockSPInstruction(Operator op, CPOperand in, CPOperand out, int br, int bc, boolean hasHeader,
			String delim, boolean fill, double fillValue, String opcode, String instr, Set<String> naStrings) {
		super(SPType.CSVReblock, op, in, out, opcode, instr);
		_blen = br;
		_blen = bc;
		_hasHeader = hasHeader;
		_delim = delim;
		_fill = fill;
		_fillValue = fillValue;
		_naStrings = naStrings;
	}

	public static CSVReblockSPInstruction parseInstruction(String str) {
		String opcode = InstructionUtils.getOpCode(str);
		if( !opcode.equals(Opcodes.CSVRBLK.toString()) )
			throw new DMLRuntimeException("Incorrect opcode for CSVReblockSPInstruction:" + opcode);

		// Example parts of CSVReblockSPInstruction:
		// [csvrblk, pREADmissing_val_maps·MATRIX·DOUBLE, _mVar37·MATRIX·DOUBLE,
		// 1000, 1000, false, ,, true, 0.0]
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen = Integer.parseInt(parts[3]);
		boolean hasHeader = Boolean.parseBoolean(parts[4]);
		String delim = parts[5];
		boolean fill = Boolean.parseBoolean(parts[6]);
		double fillValue = Double.parseDouble(parts[7]);
		Set<String> naStrings = null;

		String[] naS = parts[8].split(DataExpression.DELIM_NA_STRING_SEP);

		if(naS.length > 0  && !(naS.length ==1 && naS[0].isEmpty())){
			naStrings = new HashSet<>();
			for(String s: naS)
				naStrings.add(s);
		}

		return new CSVReblockSPInstruction(null, in, out, blen, blen,
			hasHeader, delim, fill, fillValue, opcode, str, naStrings);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		//sanity check input info
		CacheableData<?> obj = sec.getCacheableData(input1.getName());
		MetaDataFormat iimd = (MetaDataFormat) obj.getMetaData();
		if (iimd.getFileFormat() != FileFormat.CSV) {
			throw new DMLRuntimeException("The given format is not implemented for "
					+ "CSVReblockSPInstruction:" + iimd.getFileFormat().toString());
		}
		
		//set output characteristics
		DataCharacteristics mcIn = sec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());

		mcOut.set(mcIn.getRows(), mcIn.getCols(), _blen);

		//check for in-memory reblock (w/ lazy spark context, potential for latency reduction)
		if( Recompiler.checkCPReblock(sec, input1.getName()) ) {
			if( input1.getDataType().isMatrix() || input1.getDataType().isFrame() ) {
				Recompiler.executeInMemoryReblock(sec, input1.getName(), output.getName());
			}
			Statistics.decrementNoOfExecutedSPInst();
			return;
		}
		
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

	@SuppressWarnings("unchecked")
	protected JavaPairRDD<MatrixIndexes,MatrixBlock> processMatrixCSVReblockInstruction(SparkExecutionContext sec, DataCharacteristics mcOut) {
		//get input rdd (needs to be longwritable/text for consistency with meta data, in case of
		//serialization issues create longwritableser/textser as serializable wrappers
		JavaPairRDD<LongWritable, Text> in = (JavaPairRDD<LongWritable, Text>)
			sec.getRDDHandleForMatrixObject(sec.getMatrixObject(input1), FileFormat.CSV);
		
		//reblock csv to binary block
		return RDDConverterUtils.csvToBinaryBlock(sec.getSparkContext(),
			in, mcOut, _hasHeader, _delim, _fill, _fillValue, _naStrings);
	}

	@SuppressWarnings("unchecked")
	protected JavaPairRDD<Long,FrameBlock> processFrameCSVReblockInstruction(SparkExecutionContext sec, DataCharacteristics mcOut, ValueType[] schema) {
		//get input rdd (needs to be longwritable/text for consistency with meta data, in case of
		//serialization issues create longwritableser/textser as serializable wrappers
		JavaPairRDD<LongWritable, Text> in = (JavaPairRDD<LongWritable, Text>) 
			sec.getRDDHandleForFrameObject(sec.getFrameObject(input1), FileFormat.CSV);
		
		//reblock csv to binary block
		return FrameRDDConverterUtils.csvToBinaryBlock(sec.getSparkContext(),
			in, mcOut, schema, _hasHeader, _delim, _fill, _fillValue, _naStrings);
	}
}

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

import org.apache.spark.api.java.JavaPairRDD;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.lops.SortKeys;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDSortUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;

/**
 * This class supports two variants of sort operation on a 1-dimensional input matrix. 
 * The two variants are <code> weighted </code> and <code> unweighted </code>.
 * Example instructions: 
 *     sort:mVar1:mVar2 (input=mVar1, output=mVar2)
 *     sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
 *  
 */
public class QuantileSortSPInstruction extends UnarySPInstruction {

	private QuantileSortSPInstruction(CPOperand in, CPOperand out, String opcode, String istr) {
		this(in, null, out, opcode, istr);
	}

	private QuantileSortSPInstruction(CPOperand in1, CPOperand in2, CPOperand out, String opcode,
			String istr) {
		super(SPType.QSort, null, in1, in2, out, opcode, istr);
	}

	public static QuantileSortSPInstruction parseInstruction ( String str ) {
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand in2 = null;
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(SortKeys.OPCODE) ) {
			if ( parts.length == 3 ) {
				// Example: sort:mVar1:mVar2 (input=mVar1, output=mVar2)
				parseUnaryInstruction(str, in1, out);
				return new QuantileSortSPInstruction(in1, out, opcode, str);
			}
			else if ( parts.length == 4 ) {
				// Example: sort:mVar1:mVar2:mVar3 (input=mVar1, weights=mVar2, output=mVar3)
				in2 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
				parseUnaryInstruction(str, in1, in2, out);
				return new QuantileSortSPInstruction(in1, in2, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a SortSPInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		boolean weighted = (input2 != null);
		
		//get input rdds
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryMatrixBlockRDDHandleForVariable( input1.getName() );
		JavaPairRDD<MatrixIndexes,MatrixBlock> inW = weighted ? sec.getBinaryMatrixBlockRDDHandleForVariable( input2.getName() ) : null;
		DataCharacteristics mc = sec.getDataCharacteristics(input1.getName());
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> out = null;
		long clen = -1;
		if( !weighted ) { //W/O WEIGHTS (default)
			out = RDDSortUtils.sortByVal(in, mc.getRows(), mc.getBlocksize());
			clen = 1;
		}
		else { //W/ WEIGHTS
			out = RDDSortUtils.sortByVal(in, inW, mc.getRows(), mc.getBlocksize());
			clen = 2;
		}

		//put output RDD handle into symbol table
		sec.setRDDHandleForVariable(output.getName(), out);
		sec.addLineageRDD(output.getName(), input1.getName());
		if( weighted )
			sec.addLineageRDD(output.getName(), input2.getName());
		
		//update output matrix characteristics
		DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), clen, mc.getBlocksize(), mc.getBlocksize());
	}
}

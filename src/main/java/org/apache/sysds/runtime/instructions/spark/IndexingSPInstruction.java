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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
import org.apache.sysds.lops.LeftIndex.LixCacheType;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;

/**
 * This class implements the matrix indexing functionality inside Spark.
 */
public abstract class IndexingSPInstruction extends UnarySPInstruction {
	protected CPOperand rowLower, rowUpper, colLower, colUpper;
	protected SparkAggType _aggType = null;

	protected IndexingSPInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
			CPOperand out, SparkAggType aggtype, String opcode, String istr) {
		super(SPType.MatrixIndexing, null, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
		_aggType = aggtype;
	}

	protected IndexingSPInstruction(CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(SPType.MatrixIndexing, null, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}
	
	public CPOperand getRowLower() {
		return rowLower;
	}
	
	public CPOperand getRowUpper() {
		return rowUpper;
	}
	
	public CPOperand getColLower() {
		return colLower;
	}
	
	public CPOperand getColUpper() {
		return colUpper;
	}

	public static IndexingSPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) ) {
			if ( parts.length == 8 ) {
				CPOperand in = new CPOperand(parts[1]);
				CPOperand rl = new CPOperand(parts[2]);
				CPOperand ru = new CPOperand(parts[3]);
				CPOperand cl = new CPOperand(parts[4]);
				CPOperand cu = new CPOperand(parts[5]);
				CPOperand out = new CPOperand(parts[6]);
				SparkAggType aggtype = SparkAggType.valueOf(parts[7]);
				if( in.getDataType()==DataType.MATRIX )
					return new MatrixIndexingSPInstruction(in, rl, ru, cl, cu, out, aggtype, opcode, str);
				else
					return new FrameIndexingSPInstruction(in, rl, ru, cl, cu, out, aggtype, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.LEFT_INDEX.toString()) || opcode.equalsIgnoreCase(Opcodes.MAPLEFTINDEX.toString())) {
			if ( parts.length == 9 ) {
				CPOperand lhsInput = new CPOperand(parts[1]);
				CPOperand rhsInput = new CPOperand(parts[2]);
				CPOperand rl = new CPOperand(parts[3]);
				CPOperand ru = new CPOperand(parts[4]);
				CPOperand cl = new CPOperand(parts[5]);
				CPOperand cu = new CPOperand(parts[6]);
				CPOperand out = new CPOperand(parts[7]);
				LixCacheType lixtype = LixCacheType.valueOf(parts[8]);
				if( lhsInput.getDataType()==DataType.MATRIX )
					return new MatrixIndexingSPInstruction(lhsInput, rhsInput, rl, ru, cl, cu, out, lixtype, opcode, str);
				else
					return new FrameIndexingSPInstruction(lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a IndexingSPInstruction: " + str);
		}
	}
}

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

import org.apache.sysml.hops.AggBinaryOp.SparkAggType;
import org.apache.sysml.lops.LeftIndex;
import org.apache.sysml.lops.RightIndex;
import org.apache.sysml.lops.LeftIndex.LixCacheType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;

/**
 * This class implements the matrix indexing functionality inside Spark.  
 */
public abstract class IndexingSPInstruction extends UnarySPInstruction {
	protected CPOperand rowLower, rowUpper, colLower, colUpper;
	protected SparkAggType _aggType = null;

	protected IndexingSPInstruction(Operator op, CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl, CPOperand cu,
			CPOperand out, SparkAggType aggtype, String opcode, String istr) {
		super(SPType.MatrixIndexing, op, in, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
		_aggType = aggtype;
	}

	protected IndexingSPInstruction(Operator op, CPOperand lhsInput, CPOperand rhsInput, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(SPType.MatrixIndexing, op, lhsInput, rhsInput, out, opcode, istr);
		rowLower = rl;
		rowUpper = ru;
		colLower = cl;
		colUpper = cu;
	}

	public static IndexingSPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{	
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(RightIndex.OPCODE) ) {
			if ( parts.length == 8 ) {
				CPOperand in = new CPOperand(parts[1]);
				CPOperand rl = new CPOperand(parts[2]);
				CPOperand ru = new CPOperand(parts[3]);
				CPOperand cl = new CPOperand(parts[4]);
				CPOperand cu = new CPOperand(parts[5]);
				CPOperand out = new CPOperand(parts[6]);
				SparkAggType aggtype = SparkAggType.valueOf(parts[7]);
				if( in.getDataType()==DataType.MATRIX )
					return new MatrixIndexingSPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, aggtype, opcode, str);
				else
					return new FrameIndexingSPInstruction(new SimpleOperator(null), in, rl, ru, cl, cu, out, aggtype, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( opcode.equalsIgnoreCase(LeftIndex.OPCODE) || opcode.equalsIgnoreCase("mapLeftIndex")) {
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
					return new MatrixIndexingSPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, lixtype, opcode, str);
				else
					return new FrameIndexingSPInstruction(new SimpleOperator(null), lhsInput, rhsInput, rl, ru, cl, cu, out, opcode, str);
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

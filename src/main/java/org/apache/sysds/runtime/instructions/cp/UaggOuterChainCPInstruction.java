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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.lops.UAggOuterChain;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixOuterAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

public class UaggOuterChainCPInstruction extends UnaryCPInstruction {
	private final AggregateUnaryOperator _uaggOp;
	private final BinaryOperator _bOp;

	private UaggOuterChainCPInstruction(BinaryOperator bop, AggregateUnaryOperator uaggop, AggregateOperator aggop,
			CPOperand in1, CPOperand in2, CPOperand out, String opcode, String istr) {
		super(CPType.UaggOuterChain, bop, in1, in2, out, opcode, istr);
		_uaggOp = uaggop;
		_bOp = bop;
	}

	public static UaggOuterChainCPInstruction parseInstruction(String str) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];

		if ( opcode.equalsIgnoreCase(UAggOuterChain.OPCODE)) {
			AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[1]);
			BinaryOperator bop = InstructionUtils.parseBinaryOperator(parts[2]);

			CPOperand in1 = new CPOperand(parts[3]);
			CPOperand in2 = new CPOperand(parts[4]);
			CPOperand out = new CPOperand(parts[5]);
					
			//derive aggregation operator from unary operator
			String aopcode = InstructionUtils.deriveAggregateOperatorOpcode(parts[1]);
			CorrectionLocationType corrLoc = InstructionUtils.deriveAggregateOperatorCorrectionLocation(parts[1]);
			AggregateOperator aop = InstructionUtils.parseAggregateOperator(aopcode, corrLoc.toString());

			return new UaggOuterChainCPInstruction(bop, uaggop, aop, in1, in2, out, opcode, str);
		} 
		else {
			throw new DMLRuntimeException("UaggOuterChainCPInstruction.parseInstruction():: Unknown opcode " + opcode);
		}
	}
	
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		boolean rightCached = (_uaggOp.indexFn instanceof ReduceCol || _uaggOp.indexFn instanceof ReduceAll
				|| !LibMatrixOuterAgg.isSupportedUaggOp(_uaggOp, _bOp));

		MatrixBlock mbLeft = null, mbRight = null, mbOut = null;
		//get the main data input
		if( rightCached ) { 
			mbLeft = ec.getMatrixInput(input1.getName());
			mbRight = ec.getMatrixInput(input2.getName());
		}
		else { 
			mbLeft = ec.getMatrixInput(input2.getName());
			mbRight = ec.getMatrixInput(input1.getName());
		}
		
		mbOut = mbLeft.uaggouterchainOperations(mbLeft, mbRight, mbOut, _bOp, _uaggOp);

		//release locks
		ec.releaseMatrixInput(input1.getName(), input2.getName());
		
		if( _uaggOp.aggOp.existsCorrection() )
			mbOut.dropLastRowsOrColumns(_uaggOp.aggOp.correction);
		
		if(_uaggOp.indexFn instanceof ReduceAll ) { //RC AGG (output is scalar)
			ec.setMatrixOutput(output.getName(),
				new MatrixBlock(mbOut.quickGetValue(0, 0)));
		}
		else { //R/C AGG (output is rdd)
			mbOut.examSparsity();
			ec.setMatrixOutput(output.getName(), mbOut);
		}
	}
}

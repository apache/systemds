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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.lops.PickByCount.OperationTypes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class QuantilePickCPInstruction extends BinaryCPInstruction {

	private final OperationTypes _type;
	private final boolean _inmem;

	private QuantilePickCPInstruction(Operator op, CPOperand in, CPOperand out, OperationTypes type, boolean inmem,
			String opcode, String istr) {
		this(op, in, null, out, type, inmem, opcode, istr);
	}

	private QuantilePickCPInstruction(Operator op, CPOperand in, CPOperand in2, CPOperand out, OperationTypes type,
			boolean inmem, String opcode, String istr) {
		super(CPType.QPick, op, in, in2, out, opcode, istr);
		_type = type;
		_inmem = inmem;
	}

	public static QuantilePickCPInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if ( !opcode.equalsIgnoreCase(Opcodes.QPICK.toString()) )
			throw new DMLRuntimeException("Unknown opcode while parsing a QuantilePickCPInstruction: " + str);
		//instruction parsing
		if( parts.length == 4 ) {
			//instructions of length 4 originate from unary - mr-iqm
			//TODO this should be refactored to use pickvaluecount lops
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.IQM;
			boolean inmem = false;
			return new QuantilePickCPInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 5 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand out = new CPOperand(parts[2]);
			OperationTypes ptype = OperationTypes.valueOf(parts[3]);
			boolean inmem = Boolean.parseBoolean(parts[4]);
			return new QuantilePickCPInstruction(null, in1, out, ptype, inmem, opcode, str);
		}
		else if( parts.length == 6 ) {
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			CPOperand out = new CPOperand(parts[3]);
			OperationTypes ptype = OperationTypes.valueOf(parts[4]);
			boolean inmem = Boolean.parseBoolean(parts[5]);
			return new QuantilePickCPInstruction(null, in1, in2, out, ptype, inmem, opcode, str);
		}
		return null;
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		switch( _type ) 
		{
			case VALUEPICK:
				// Handle both in-memory and non-in-memory VALUEPICK by materializing
				// the input matrix and invoking the pick routines. Previously only
				// the in-memory branch was executed which left the output unset
				// when _inmem==false (see SYSTEMDS-3898).
			{
				MatrixBlock matBlock = ec.getMatrixInput(input1.getName());

				if ( input2.getDataType() == DataType.SCALAR ) {
					ScalarObject quantile = ec.getScalarInput(input2);
					// pick value w/ explicit averaging for even-length arrays
					double picked = matBlock.pickValue(
					quantile.getDoubleValue(), matBlock.getLength()%2==0);
					ec.setScalarOutput(output.getName(), new DoubleObject(picked));
				} 
				else {
					MatrixBlock quantiles = ec.getMatrixInput(input2.getName());
					// pick values w/ explicit averaging for even-length arrays
					MatrixBlock resultBlock = matBlock.pickValues(
					quantiles, new MatrixBlock(), matBlock.getLength()%2==0);
					quantiles = null;
					ec.releaseMatrixInput(input2.getName());
					ec.setMatrixOutput(output.getName(), resultBlock);
				}
				ec.releaseMatrixInput(input1.getName());
			}
                        	break;
			case MEDIAN:
				if( _inmem ) //INMEM MEDIAN
				{
					double picked = ec.getMatrixInput(input1.getName()).median();
					ec.setScalarOutput(output.getName(), new DoubleObject(picked));
					ec.releaseMatrixInput(input1.getName());
					break;
				}
				break;
				
			case IQM:
				if( _inmem ) //INMEM IQM
				{
					MatrixBlock matBlock1 = ec.getMatrixInput(input1.getName());
					double iqm = matBlock1.interQuartileMean();
					ec.releaseMatrixInput(input1.getName());
					ec.setScalarOutput(output.getName(), new DoubleObject(iqm));
				}
				break;
				
			default:
				throw new DMLRuntimeException("Unsupported qpick operation type: "+_type);
		}
	}

	public OperationTypes getOperationType() {
		return _type;
	}

	public boolean isInMem() {
		return _inmem;
	}
}

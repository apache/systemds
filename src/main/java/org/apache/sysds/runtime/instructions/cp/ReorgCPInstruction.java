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
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.DiagIndex;
import org.apache.sysds.runtime.functionobjects.RevIndex;
import org.apache.sysds.runtime.functionobjects.RollIndex;
import org.apache.sysds.runtime.functionobjects.SortIndex;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.util.DataConverter;

public class ReorgCPInstruction extends UnaryCPInstruction {
	// sort-specific attributes (to enable variable attributes)
	private final CPOperand _col;
	private final CPOperand _desc;
	private final CPOperand _ixret;
	private final CPOperand _shift;

	/**
	 * for opcodes r' and rdiag
	 *
	 * @param op     operator
	 * @param in     cp input operand
	 * @param out    cp output operand
	 * @param opcode the opcode
	 * @param istr   ?
	 */
	private ReorgCPInstruction(Operator op, CPOperand in, CPOperand out, String opcode, String istr) {
		this(op, in, out, null, null, null, opcode, istr);
	}

	/**
	 * for opcode rsort
	 *
	 * @param op     operator
	 * @param in     cp input operand
	 * @param col    ?
	 * @param desc   ?
	 * @param ixret  ?
	 * @param out    cp output operand
	 * @param opcode the opcode
	 * @param istr   ?
	 */
	private ReorgCPInstruction(Operator op, CPOperand in, CPOperand out, CPOperand col, CPOperand desc, CPOperand ixret,
							   String opcode, String istr) {
		super(CPType.Reorg, op, in, out, opcode, istr);
		_col = col;
		_desc = desc;
		_ixret = ixret;
		_shift = null;
	}

	/**
	 * for opcode roll
	 *
	 * @param op     operator
	 * @param in     cp input operand
	 * @param shift  ?
	 * @param out    cp output operand
	 * @param opcode the opcode
	 * @param istr   ?
	 */
	private ReorgCPInstruction(Operator op, CPOperand in, CPOperand out, CPOperand shift, String opcode, String istr) {
		super(CPType.Reorg, op, in, shift, out, opcode, istr);
		_col = null;
		_desc = null;
		_ixret = null;
		_shift = shift;
	}

	public static ReorgCPInstruction parseInstruction ( String str ) {
		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(Opcodes.TRANSPOSE.toString()) ) {
			InstructionUtils.checkNumFields(str, 2, 3);
			in.split(parts[1]);
			out.split(parts[2]);
			int k = Integer.parseInt(parts[3]);
			return new ReorgCPInstruction(new ReorgOperator(SwapIndex.getSwapIndexFnObject(), k), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.REV.toString()) ) {
			InstructionUtils.checkNumFields(str, 2, 3);
			in.split(parts[1]);
			out.split(parts[2]);
			// Safely parse the number of threads 'k' if it exists
			int k = (parts.length > 3) ? Integer.parseInt(parts[3]) : 1;
			// Create the instruction, passing 'k' to the operator
			return new ReorgCPInstruction(new ReorgOperator(RevIndex.getRevIndexFnObject(), k), in, out, opcode, str);
		}
		else if (opcode.equalsIgnoreCase(Opcodes.ROLL.toString())) {
			InstructionUtils.checkNumFields(str, 3, 4);
			in.split(parts[1]);
			out.split(parts[3]);
			CPOperand shift = new CPOperand(parts[2]);
			int k = (parts.length > 4) ? Integer.parseInt(parts[4]) : 1;

			return new ReorgCPInstruction(new ReorgOperator(new RollIndex(0), k), in, out, shift, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase(Opcodes.DIAG.toString()) ) {
			parseUnaryInstruction(str, in, out); //max 2 operands
			return new ReorgCPInstruction(new ReorgOperator(DiagIndex.getDiagIndexFnObject()), in, out, opcode, str);
		} 
		else if ( opcode.equalsIgnoreCase(Opcodes.SORT.toString()) ) {
			InstructionUtils.checkNumFields(str, 5,6);
			in.split(parts[1]);
			out.split(parts[5]);
			CPOperand col = new CPOperand(parts[2]);
			CPOperand desc = new CPOperand(parts[3]);
			CPOperand ixret = new CPOperand(parts[4]);
			int k = Integer.parseInt(parts[6]);
			return new ReorgCPInstruction(new ReorgOperator(new SortIndex(1,false,false), k), 
				in, out, col, desc, ixret, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a ReorgInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		//acquire inputs
		MatrixBlock matBlock = ec.getMatrixInput(input1.getName());
		ReorgOperator r_op = (ReorgOperator) _optr;
		if( r_op.fn instanceof SortIndex ) {
			//additional attributes for sort
			int[] cols = _col.getDataType().isMatrix() ? DataConverter.convertToIntVector(ec.getMatrixInput(_col.getName())) :
				new int[]{(int)ec.getScalarInput(_col).getLongValue()};
			boolean desc = ec.getScalarInput(_desc).getBooleanValue();
			boolean ixret = ec.getScalarInput(_ixret).getBooleanValue();
			r_op = r_op.setFn(new SortIndex(cols, desc, ixret));
		}

		if (r_op.fn instanceof RollIndex) {
			int shift = (int) ec.getScalarInput(_shift).getLongValue();
			r_op = r_op.setFn(new RollIndex(shift));
		}

		//execute operation
		MatrixBlock soresBlock = matBlock.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0);
		
		//release inputs/outputs
		if( r_op.fn instanceof SortIndex && _col.getDataType().isMatrix() )
			ec.releaseMatrixInput(_col.getName());
		ec.releaseMatrixInput(input1.getName());
		ec.setMatrixOutput(output.getName(), soresBlock);
		if( r_op.fn instanceof DiagIndex && soresBlock.getNumColumns()>1 ) //diagV2M
			ec.getMatrixObject(output.getName()).setDiag(true);
	}

	public CPOperand getIxRet() {
		return _ixret;
	}
}

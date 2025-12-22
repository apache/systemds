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

package org.apache.sysds.runtime.instructions.ooc;

import java.util.List;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.IfElse;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObjectFactory;
import org.apache.sysds.runtime.instructions.cp.StringObject;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;

public class TernaryOOCInstruction extends ComputationOOCInstruction {

	protected TernaryOOCInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out,
		String opcode, String istr) {
		super(OOCType.Ternary, op, in1, in2, in3, out, opcode, istr);
	}

	public static TernaryOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		InstructionUtils.checkNumFields(parts, 4, 5);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		int numThreads = parts.length > 5 ? Integer.parseInt(parts[5]) : 1;
		TernaryOperator op = InstructionUtils.parseTernaryOperator(opcode, numThreads);
		return new TernaryOOCInstruction(op, in1, in2, in3, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		boolean m1 = input1.isMatrix();
		boolean m2 = input2.isMatrix();
		boolean m3 = input3.isMatrix();

		if(!m1 && !m2 && !m3) {
			processScalarInstruction(ec);
			return;
		}

		if(m1 && m2 && m3)
			processThreeMatrixInstruction(ec);
		else if(m1 && m2)
			processTwoMatrixInstruction(ec, 1, 2);
		else if(m1 && m3)
			processTwoMatrixInstruction(ec, 1, 3);
		else if(m2 && m3)
			processTwoMatrixInstruction(ec, 2, 3);
		else if(m1)
			processSingleMatrixInstruction(ec, 1);
		else if(m2)
			processSingleMatrixInstruction(ec, 2);
		else
			processSingleMatrixInstruction(ec, 3);
	}

	private void processScalarInstruction(ExecutionContext ec) {
		TernaryOperator op = (TernaryOperator) _optr;
		if(op.fn instanceof IfElse && output.getValueType() == ValueType.STRING) {
			String value = (ec.getScalarInput(input1).getDoubleValue() != 0 ?
				ec.getScalarInput(input2) : ec.getScalarInput(input3)).getStringValue();
			ec.setScalarOutput(output.getName(), new StringObject(value));
		}
		else {
			double value = op.fn.execute(
				ec.getScalarInput(input1).getDoubleValue(),
				ec.getScalarInput(input2).getDoubleValue(),
				ec.getScalarInput(input3).getDoubleValue());
			ec.setScalarOutput(output.getName(), ScalarObjectFactory
				.createScalarObject(output.getValueType(), value));
		}
	}

	private void processSingleMatrixInstruction(ExecutionContext ec, int matrixPos) {
		MatrixObject mo = getMatrixObject(ec, matrixPos);
		MatrixBlock s1 = input1.isMatrix() ? null : getScalarInputBlock(ec, input1);
		MatrixBlock s2 = input2.isMatrix() ? null : getScalarInputBlock(ec, input2);
		MatrixBlock s3 = input3.isMatrix() ? null : getScalarInputBlock(ec, input3);

		OOCStream<IndexedMatrixValue> qIn = mo.getStreamHandle();
		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);
		qIn.setDownstreamMessageRelay(qOut::messageDownstream);
		qOut.setUpstreamMessageRelay(qIn::messageUpstream);

		mapOOC(qIn, qOut, tmp -> {
			IndexedMatrixValue outVal = new IndexedMatrixValue();
			MatrixBlock op1 = resolveOperandBlock(1, tmp, null, matrixPos, -1, s1, s2, s3);
			MatrixBlock op2 = resolveOperandBlock(2, tmp, null, matrixPos, -1, s1, s2, s3);
			MatrixBlock op3 = resolveOperandBlock(3, tmp, null, matrixPos, -1, s1, s2, s3);
			outVal.set(tmp.getIndexes(),
				op1.ternaryOperations((TernaryOperator)_optr, op2, op3, new MatrixBlock()));
			return outVal;
		});
	}

	private void processTwoMatrixInstruction(ExecutionContext ec, int leftPos, int rightPos) {
		MatrixObject left = getMatrixObject(ec, leftPos);
		MatrixObject right = getMatrixObject(ec, rightPos);
		OOCStream<IndexedMatrixValue> leftStream = left.getStreamHandle();
		OOCStream<IndexedMatrixValue> rightStream = right.getStreamHandle();

		MatrixBlock s1 = input1.isMatrix() ? null : getScalarInputBlock(ec, input1);
		MatrixBlock s2 = input2.isMatrix() ? null : getScalarInputBlock(ec, input2);
		MatrixBlock s3 = input3.isMatrix() ? null : getScalarInputBlock(ec, input3);

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);
		qOut.setUpstreamMessageRelay(msg -> {
			leftStream.messageUpstream(msg.split());
			rightStream.messageUpstream(msg.split());
		});
		leftStream.setDownstreamMessageRelay(qOut::messageDownstream);
		rightStream.setDownstreamMessageRelay(qOut::messageDownstream);

		joinOOC(leftStream, rightStream, qOut, (l, r) -> {
			IndexedMatrixValue outVal = new IndexedMatrixValue();
			MatrixBlock op1 = resolveOperandBlock(1, l, r, leftPos, rightPos, s1, s2, s3);
			MatrixBlock op2 = resolveOperandBlock(2, l, r, leftPos, rightPos, s1, s2, s3);
			MatrixBlock op3 = resolveOperandBlock(3, l, r, leftPos, rightPos, s1, s2, s3);
			outVal.set(l.getIndexes(),
				op1.ternaryOperations((TernaryOperator)_optr, op2, op3, new MatrixBlock()));
			return outVal;
		}, IndexedMatrixValue::getIndexes);
	}

	private void processThreeMatrixInstruction(ExecutionContext ec) {
		MatrixObject m1 = ec.getMatrixObject(input1);
		MatrixObject m2 = ec.getMatrixObject(input2);
		MatrixObject m3 = ec.getMatrixObject(input3);

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		ec.getMatrixObject(output).setStreamHandle(qOut);

		List<OOCStream<IndexedMatrixValue>> streams = List.of(
			m1.getStreamHandle(), m2.getStreamHandle(), m3.getStreamHandle());

		streams.forEach(s -> s.setDownstreamMessageRelay(qOut::messageDownstream));
		qOut.setUpstreamMessageRelay(msg -> streams.forEach(s -> s.messageUpstream(msg)));

		joinOOC(streams, qOut, blocks -> {
			IndexedMatrixValue b1 = blocks.get(0);
			IndexedMatrixValue b2 = blocks.get(1);
			IndexedMatrixValue b3 = blocks.get(2);
			IndexedMatrixValue outVal = new IndexedMatrixValue();
			outVal.set(b1.getIndexes(),
				((MatrixBlock)b1.getValue()).ternaryOperations((TernaryOperator)_optr, (MatrixBlock)b2.getValue(), (MatrixBlock)b3.getValue(), new MatrixBlock()));
			return outVal;
		}, IndexedMatrixValue::getIndexes);
	}

	private MatrixObject getMatrixObject(ExecutionContext ec, int pos) {
		if(pos == 1)
			return ec.getMatrixObject(input1);
		else if(pos == 2)
			return ec.getMatrixObject(input2);
		else if(pos == 3)
			return ec.getMatrixObject(input3);
		else
			throw new DMLRuntimeException("Invalid matrix position: " + pos);
	}

	private MatrixBlock getScalarInputBlock(ExecutionContext ec, CPOperand operand) {
		ScalarObject scalar = ec.getScalarInput(operand);
		return new MatrixBlock(scalar.getDoubleValue());
	}

	private MatrixBlock resolveOperandBlock(int operandPos, IndexedMatrixValue left, IndexedMatrixValue right,
		int leftPos, int rightPos, MatrixBlock s1, MatrixBlock s2, MatrixBlock s3) {
		if(operandPos == leftPos && left != null)
			return (MatrixBlock) left.getValue();
		if(operandPos == rightPos && right != null)
			return (MatrixBlock) right.getValue();

		if(operandPos == 1)
			return s1;
		else if(operandPos == 2)
			return s2;
		else if(operandPos == 3)
			return s3;
		else
			throw new DMLRuntimeException("Invalid operand position: " + operandPos);
	}
}

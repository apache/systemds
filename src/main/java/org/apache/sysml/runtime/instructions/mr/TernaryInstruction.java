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

package org.apache.sysml.runtime.instructions.mr;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.TernaryOperator;

public class TernaryInstruction extends MRInstruction {
	
	private final CPOperand input1, input2, input3, output;
	private final byte ixinput1, ixinput2, ixinput3, ixoutput;
	private final MatrixBlock m1, m2, m3;
	
	private TernaryInstruction(Operator op, CPOperand in1, CPOperand in2, CPOperand in3, CPOperand out, String istr) {
		super(MRType.Ternary, op, Byte.parseByte(out.getName()));
		instString = istr;
		input1 = in1; input2 = in2; input3 = in3; output = out;
		ixinput1 = input1.isMatrix() ? Byte.parseByte(input1.getName()) : -1;
		ixinput2 = input2.isMatrix() ? Byte.parseByte(input2.getName()) : -1;
		ixinput3 = input3.isMatrix() ? Byte.parseByte(input3.getName()) : -1;
		ixoutput = output.isMatrix() ? Byte.parseByte(output.getName()) : -1;
		m1 = input1.isMatrix() ? null :new MatrixBlock(Double.parseDouble(input1.getName()));
		m2 = input2.isMatrix() ? null :new MatrixBlock(Double.parseDouble(input2.getName()));
		m3 = input3.isMatrix() ? null :new MatrixBlock(Double.parseDouble(input3.getName()));
	}

	public static TernaryInstruction parseInstruction ( String str )
		throws DMLRuntimeException
	{
		InstructionUtils.checkNumFields ( str, 4 );
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		CPOperand in1 = new CPOperand(parts[1]);
		CPOperand in2 = new CPOperand(parts[2]);
		CPOperand in3 = new CPOperand(parts[3]);
		CPOperand out = new CPOperand(parts[4]);
		TernaryOperator op = InstructionUtils.parseTernaryOperator(opcode);
		return new TernaryInstruction(op, in1, in2, in3, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues,
			IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLRuntimeException 
	{
		MatrixBlock lm1 = input1.isMatrix() ? (MatrixBlock) cachedValues.getFirst(ixinput1).getValue() : m1;
		MatrixBlock lm2 = input2.isMatrix() ? (MatrixBlock) cachedValues.getFirst(ixinput2).getValue() : m2;
		MatrixBlock lm3 = input3.isMatrix() ? (MatrixBlock) cachedValues.getFirst(ixinput3).getValue() : m3;
		MatrixIndexes ixin = input1.isMatrix() ? cachedValues.getFirst(ixinput1).getIndexes() : input2.isMatrix() ?
			cachedValues.getFirst(ixinput2).getIndexes() : cachedValues.getFirst(ixinput3).getIndexes();
		
		//prepare output
		IndexedMatrixValue out = new IndexedMatrixValue(new MatrixIndexes(), new MatrixBlock());
		out.getIndexes().setIndexes(ixin);
		
		//process instruction
		TernaryOperator op = (TernaryOperator)optr;
		lm1.ternaryOperations(op, lm2, lm3, (MatrixBlock)out.getValue());
		
		//put the output value in the cache
		cachedValues.add(ixoutput, out);
	}
	
	@Override
	public byte[] getInputIndexes() {
		byte[] tmp = getAllIndexes();
		return Arrays.copyOfRange(tmp, 0, tmp.length-1);
	}

	@Override
	public byte[] getAllIndexes() {
		return ArrayUtils.toPrimitive(
			Arrays.stream(new CPOperand[]{input1, input2, input3, output})
			.filter(in -> in.isMatrix()).map(in -> Byte.parseByte(in.getName()))
			.toArray(Byte[]::new));
	}
}

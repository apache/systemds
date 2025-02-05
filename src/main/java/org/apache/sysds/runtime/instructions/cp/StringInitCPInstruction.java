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

import java.util.StringTokenizer;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.Operator;

public class StringInitCPInstruction extends UnaryCPInstruction {
	public static final String DELIM = " ";

	private final long _rlen;
	private final long _clen;
	private final String _data;

	private StringInitCPInstruction(Operator op, CPOperand in, CPOperand out, long rows, long cols,
			int blen, String data, String opcode, String inst) {
		super(CPType.StringInit, op, in, out, opcode, inst);
		_rlen = rows;
		_clen = cols;
		_data = data;
	}

	public long getRows() {
		return _rlen;
	}
	
	public long getCols() {
		return _clen;
	}

	public static StringInitCPInstruction parseInstruction(String str) {
		String opcode = InstructionUtils.getOpCode(str);
		if( !opcode.equals(Opcodes.STRINGINIT.toString()) )
			throw new DMLRuntimeException("Unsupported opcode: "+opcode);
		//parse instruction
		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		InstructionUtils.checkNumFields( s, 5 );
		CPOperand out = new CPOperand(s[s.length-1]); // output is specified by the last operand
		long rows = (s[1].contains( Lop.VARIABLE_NAME_PLACEHOLDER)?-1:Double.valueOf(s[1]).longValue());
		long cols = (s[2].contains( Lop.VARIABLE_NAME_PLACEHOLDER)?-1:Double.valueOf(s[2]).longValue());
		int blen = Integer.parseInt(s[3]);
		String data = s[4];
		return new StringInitCPInstruction(null, null, out, rows, cols, blen, data, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec ) {
		//setup output matrix
		String outName = output.getName();
		MatrixBlock outBlk = new MatrixBlock((int)_rlen, (int)_clen, false);
		
		//init tokenizer 
		StringTokenizer st = new StringTokenizer(_data, DELIM);
		int len = st.countTokens();
		
		//check consistent sizes
		if( len != _rlen*_clen )
			throw new DMLRuntimeException("Invalid matrix string intialization: dimensions=["+_rlen+"x"+_clen+"] vs numValues="+len);
		
		//parse input data string and init matrix
		for( int i=0; i<len; i++ ){
			String sval = st.nextToken();
			Double dval = Double.parseDouble(sval);
			int rix = (int) (i / _clen);
			int cix = (int) (i % _clen);
			outBlk.set(rix, cix, dval);
		}
		
		//examine right output representation
		outBlk.recomputeNonZeros();
		outBlk.examSparsity();
		
		//put output into symbol table
		ec.setMatrixOutput(outName, outBlk);
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		CPOperand rlen = new CPOperand(String.valueOf(getRows()), ValueType.INT64, DataType.SCALAR, true);
		CPOperand clen = new CPOperand(String.valueOf(getCols()), ValueType.INT64, DataType.SCALAR, true);
		CPOperand data = new CPOperand(_data, ValueType.STRING, DataType.SCALAR, true);
		return Pair.of(output.getName(), 
				new LineageItem(getOpcode(), LineageItemUtils.getLineage(ec, rlen, clen, data)));

	}
}

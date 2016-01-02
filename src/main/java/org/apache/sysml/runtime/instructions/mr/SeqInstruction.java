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

import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;


public class SeqInstruction extends DataGenMRInstruction
{
	
	public double fromValue;
	public double toValue;
	public double incrValue;
	
	public SeqInstruction ( Operator op, byte in, byte out, long rows, long cols, int rpb, int cpb, double fromValue, double toValue,
				double incrValue, String baseDir, String istr ) {
		super(op, DataGenMethod.SEQ, in, out, rows, cols, rpb, cpb, baseDir);
		mrtype = MRINSTRUCTION_TYPE.Seq;
		this.fromValue = fromValue;
		this.toValue = toValue;
		this.incrValue = incrValue;
		instString = istr;
	}
	
	public static SeqInstruction parseInstruction(String str) throws DMLRuntimeException 
	{
		InstructionUtils.checkNumFields ( str, 10 );

		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		Operator op = null;
		byte input = Byte.parseByte(parts[1]);
		byte output = Byte.parseByte(parts[2]);
		long rows = Double.valueOf(parts[3]).longValue();
		long cols = Double.valueOf(parts[4]).longValue();
		int rpb = Integer.parseInt(parts[5]);
		int cpb = Integer.parseInt(parts[6]);
		double fromValue = Double.parseDouble(parts[7]);
		double toValue = Double.parseDouble(parts[8]);
		double incrValue = Double.parseDouble(parts[9]);
		String baseDir = parts[10];
		
		return new SeqInstruction(op, input, output, rows, cols, rpb, cpb, fromValue, toValue, incrValue, baseDir, str);
	}

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
			throws DMLUnsupportedOperationException, DMLRuntimeException {
		// TODO Auto-generated method stub
		
	}
	
}

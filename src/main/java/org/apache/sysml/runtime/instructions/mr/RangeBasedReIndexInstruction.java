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

import java.util.ArrayList;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.ReIndexOperator;
import org.apache.sysml.runtime.util.IndexRange;
import org.apache.sysml.runtime.util.UtilFunctions;


public class RangeBasedReIndexInstruction extends UnaryMRInstructionBase
{
	private IndexRange _ixrange = null;
	private boolean _forLeft = false;	
	private long _rlenLhs = -1;
	private long _clenLhs = -1;
	
	public RangeBasedReIndexInstruction(Operator op, byte in, byte out, IndexRange rng, boolean forleft, long rlen, long clen, String istr) {
		super(op, in, out);
		mrtype = MRINSTRUCTION_TYPE.RangeReIndex;
		instString = istr;
		_ixrange = rng;
		_forLeft = forleft;
		_rlenLhs = rlen;
		_clenLhs = clen;
	}
	
	/**
	 * 
	 * @param mcIn
	 * @param mcOut
	 */
	public void computeOutputCharacteristics(MatrixCharacteristics mcIn, MatrixCharacteristics mcOut) {
		if( _forLeft )
			mcOut.set(_rlenLhs, _clenLhs, mcIn.getRowsPerBlock(), mcIn.getColsPerBlock(), -1);
		else
			mcOut.set(_ixrange.rowEnd-_ixrange.rowStart+1, _ixrange.colEnd-_ixrange.colStart+1, 
					mcIn.getRowsPerBlock(), mcIn.getColsPerBlock(), -1);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static RangeBasedReIndexInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{	
		InstructionUtils.checkNumFields ( str, 8 );
		String[] parts = InstructionUtils.getInstructionParts ( str );
		
		String opcode = parts[0];
		boolean forLeft = false;
		if(opcode.equalsIgnoreCase("rangeReIndexForLeft"))
			forLeft=true;
		else if(!opcode.equalsIgnoreCase("rangeReIndex"))
			throw new DMLRuntimeException("Unknown opcode while parsing a Select: " + str);
		byte in = Byte.parseByte(parts[1]); 
		IndexRange rng=new IndexRange(UtilFunctions.parseToLong(parts[2]), 
									  UtilFunctions.parseToLong(parts[3]), 
									  UtilFunctions.parseToLong(parts[4]),
									  UtilFunctions.parseToLong(parts[5]));		
		byte out = Byte.parseByte(parts[6]);
		long rlen = Long.parseLong(parts[7]);
		long clen = Long.parseLong(parts[8]);
		
		return new RangeBasedReIndexInstruction(new ReIndexOperator(), in, out, rng, forLeft, rlen, clen, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue,
			IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLRuntimeException 
	{		
		if(input==output)
			throw new DMLRuntimeException("input cannot be the same for output for "+instString);
		
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(input);
		if( blkList != null ) {
			for(IndexedMatrixValue in : blkList) 
			{
				if( in == null )
					continue;
	
				//process instruction (incl block allocation)
				ArrayList<IndexedMatrixValue> outlist = new ArrayList<IndexedMatrixValue>();
				if( _forLeft )
					OperationsOnMatrixValues.performShift(in, _ixrange, blockRowFactor, blockColFactor, _rlenLhs, _clenLhs, outlist);
				else
					OperationsOnMatrixValues.performSlice(in, _ixrange, blockRowFactor, blockColFactor, outlist);
		
				//put blocks into result cache
				for( IndexedMatrixValue ret : outlist )
					cachedValues.add(output, ret);
			}
		}
	}
}

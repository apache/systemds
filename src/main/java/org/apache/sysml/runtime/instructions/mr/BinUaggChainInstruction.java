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
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.data.OperationsOnMatrixValues;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;


/**
 * 
 */
public class BinUaggChainInstruction extends UnaryInstruction 
{
	
	//operators
	private BinaryOperator _bOp = null;
	private AggregateUnaryOperator _uaggOp = null;
	
	//reused intermediates  
	private MatrixIndexes _tmpIx = null;
	private MatrixValue _tmpVal = null;
	
	/**
	 * 
	 * @param bop
	 * @param uaggop
	 * @param in1
	 * @param out
	 * @param istr
	 */
	public BinUaggChainInstruction(BinaryOperator bop, AggregateUnaryOperator uaggop, byte in1, byte out, String istr)
	{
		super(null, in1, out, istr);
		
		_bOp = bop;
		_uaggOp = uaggop;
		
		_tmpIx = new MatrixIndexes();
		_tmpVal = new MatrixBlock();
		
		mrtype = MRINSTRUCTION_TYPE.BinUaggChain;
		instString = istr;
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static BinUaggChainInstruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 4 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionParts( str );		
		
		BinaryOperator bop = InstructionUtils.parseBinaryOperator(parts[1]);
		AggregateUnaryOperator uaggop = InstructionUtils.parseBasicAggregateUnaryOperator(parts[2]);
		byte in1 = Byte.parseByte(parts[3]);
		byte out = Byte.parseByte(parts[4]);
		
		return new BinUaggChainInstruction(bop, uaggop, in1, out, str);
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			           IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLRuntimeException 
	{
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get( input );
		if( blkList == null )
			return;

		for(IndexedMatrixValue imv : blkList)
		{
			if(imv == null)
				continue;
			
			MatrixIndexes inIx = imv.getIndexes();
			MatrixValue inVal = imv.getValue();
			
			//allocate space for the intermediate and output value
			IndexedMatrixValue iout = cachedValues.holdPlace(output, valueClass);
			MatrixIndexes outIx = iout.getIndexes();
			MatrixValue outVal = iout.getValue();
			
			//process instruction
			OperationsOnMatrixValues.performAggregateUnary( inIx, inVal, _tmpIx, _tmpVal, _uaggOp, blockRowFactor, blockColFactor);
			((MatrixBlock)_tmpVal).dropLastRowsOrColums(_uaggOp.aggOp.correctionLocation);
			OperationsOnMatrixValues.performBinaryIgnoreIndexes(inVal, _tmpVal, outVal, _bOp);
			outIx.setIndexes(inIx);
		}
	}
}

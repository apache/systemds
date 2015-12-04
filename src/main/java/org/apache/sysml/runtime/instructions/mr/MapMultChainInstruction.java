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

import org.apache.sysml.lops.MapMultChain.ChainType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue;
import org.apache.sysml.runtime.matrix.mapred.CachedValueMap;
import org.apache.sysml.runtime.matrix.mapred.DistributedCacheInput;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.MRBaseForCommonInstructions;

/**
 * 
 */
public class MapMultChainInstruction extends MRInstruction implements IDistributedCacheConsumer
{
	private ChainType _chainType = null;
	
	private byte _input1 = -1;
	private byte _input2 = -1;
	private byte _input3 = -1;	
	
	/**
	 * Two matrix inputs - type XtXv
	 * 
	 * @param type
	 * @param in1
	 * @param in2
	 * @param out
	 * @param istr
	 */
	public MapMultChainInstruction(ChainType type, byte in1, byte in2, byte out, String istr)
	{
		super(null, out);
		
		_chainType = type;
		
		_input1 = in1;
		_input2 = in2;
		_input3 = -1;
		
		mrtype = MRINSTRUCTION_TYPE.MapMultChain;
		instString = istr;
	}

	/**
	 * Three matrix inputs - type XtwXv
	 * 
	 * @param type
	 * @param in1
	 * @param in2
	 * @param in3
	 * @param out
	 * @param istr
	 */
	public MapMultChainInstruction(ChainType type, byte in1, byte in2, byte in3, byte out, String istr)
	{
		super(null, out);
		
		_chainType = type;
		
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		
		mrtype = MRINSTRUCTION_TYPE.MapMultChain;
		instString = istr;
	}
	
	public ChainType getChainType()
	{
		return _chainType;
	}
	
	public byte getInput1() {
		return _input1;
	}

	public byte getInput2() {
		return _input2;
	}

	public byte getInput3() {
		return _input3;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction( String str ) 
		throws DMLRuntimeException 
	{		
		//check number of fields (2/3 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 4, 5 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionParts( str );		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		
		if( parts.length==5 )
		{
			byte out = Byte.parseByte(parts[3]);
			ChainType type = ChainType.valueOf(parts[4]);
			
			return new MapMultChainInstruction(type, in1, in2, out, str);
		}
		else //parts.length==6
		{
			byte in3 = Byte.parseByte(parts[3]);
			byte out = Byte.parseByte(parts[4]);
			ChainType type = ChainType.valueOf(parts[5]);
		
			return new MapMultChainInstruction(type, in1, in2, in3, out, str);
		}	
	}
	
	@Override //IDistributedCacheConsumer
	public boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		return (_chainType == ChainType.XtXv) ?
			(index==_input2 && index!=_input1) :
			(index==_input2 && index!=_input1) || (index==_input3 && index!=_input1);
	}
	
	@Override //IDistributedCacheConsumer
	public void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		if( _chainType == ChainType.XtXv ){
			indexes.add(_input2);
		}
		else if( _chainType == ChainType.XtwXv ){
			indexes.add(_input2);
			indexes.add(_input3);	
		}
	}
	
	@Override
	public byte[] getInputIndexes() 
	{
		if( _chainType==ChainType.XtXv )
			return new byte[]{_input1, _input2};
		else
			return new byte[]{_input1, _input2, _input3};
	}

	@Override
	public byte[] getAllIndexes() 
	{
		if( _chainType==ChainType.XtXv )
			return new byte[]{_input1, _input2, output};
		else
			return new byte[]{_input1, _input2, _input3, output};
	}
	

	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			           IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		ArrayList<IndexedMatrixValue> blkList = cachedValues.get(_input1);
		if( blkList !=null )
			for(IndexedMatrixValue imv : blkList)
			{
				if(imv==null)
					continue;
				MatrixIndexes inIx = imv.getIndexes();
				MatrixValue inVal = imv.getValue();
				
				//allocate space for the output value
				IndexedMatrixValue iout = null;
				if(output==_input1)
					iout=tempValue;
				else
					iout=cachedValues.holdPlace(output, valueClass);
				
				MatrixIndexes outIx = iout.getIndexes();
				MatrixValue outVal = iout.getValue();
				
				//process instruction
				if( _chainType == ChainType.XtXv )
					processXtXvOperations(inIx, inVal, outIx, outVal);
				else
					processXtwXvOperations(inIx, inVal, outIx, outVal);
				
				//put the output value in the cache
				if(iout==tempValue)
					cachedValues.add(output, iout);
			}
	}

	
	/**
	 * Chain implementation for r = (t(X)%*%(X%*%v))
	 * (implemented as r = (t(t(X%*%v)%*%X))
	 * 
	 * @param inIx
	 * @param inVal
	 * @param outIx
	 * @param outVal
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	private void processXtXvOperations(MatrixIndexes inIx, MatrixValue inVal, MatrixIndexes outIx, MatrixValue outVal ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		DistributedCacheInput dcInput2 = MRBaseForCommonInstructions.dcValues.get(_input2); //v
		MatrixBlock Xi = (MatrixBlock)inVal;
		MatrixBlock v = (MatrixBlock) dcInput2.getDataBlock(1, 1).getValue();
		
		//process core block operation
		Xi.chainMatrixMultOperations(v, null, (MatrixBlock) outVal, ChainType.XtXv);
		outIx.setIndexes(1, 1);
	}
	
	/**
	 * Chain implementation for r = (t(X)%*%(w*(X%*%v)))
	 * (implemented as r = (t(t((X%*%v)*w)%*%X))
	 * 
	 * @param inIx
	 * @param inVal
	 * @param outIx
	 * @param outVal
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	private void processXtwXvOperations(MatrixIndexes inIx, MatrixValue inVal, MatrixIndexes outIx, MatrixValue outVal )
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		DistributedCacheInput dcInput2 = MRBaseForCommonInstructions.dcValues.get(_input2); //v
		DistributedCacheInput dcInput3 = MRBaseForCommonInstructions.dcValues.get(_input3); //w
		MatrixBlock Xi = (MatrixBlock) inVal;
		MatrixBlock v = (MatrixBlock) dcInput2.getDataBlock(1, 1).getValue();
		MatrixBlock w = (MatrixBlock) dcInput3.getDataBlock((int)inIx.getRowIndex(), 1).getValue();
		
		//process core block operation
		Xi.chainMatrixMultOperations(v, w, (MatrixBlock) outVal, ChainType.XtwXv);
		outIx.setIndexes(1, 1);
	}
}

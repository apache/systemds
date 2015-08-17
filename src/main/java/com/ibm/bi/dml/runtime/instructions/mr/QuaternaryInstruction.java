/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.WeightedSigmoid;
import com.ibm.bi.dml.lops.WeightedSigmoid.WSigmoidType;
import com.ibm.bi.dml.lops.WeightedSigmoidR;
import com.ibm.bi.dml.lops.WeightedSquaredLoss;
import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.lops.WeightedSquaredLossR;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.functionobjects.SwapIndex;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.LibMatrixReorg;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.QuaternaryOperator;
import com.ibm.bi.dml.runtime.matrix.operators.ReorgOperator;

/**
 * 
 */
public class QuaternaryInstruction extends MRInstruction 
{
	
	private byte _input1 = -1;
	private byte _input2 = -1;
	private byte _input3 = -1;	
	private byte _input4 = -1;
	
	private boolean _cacheU = false;
	private boolean _cacheV = false;
	
	/**
	 * 
	 * @param type
	 * @param in1
	 * @param in2
	 * @param out
	 * @param istr
	 */
	public QuaternaryInstruction(Operator op, byte in1, byte in2, byte in3, byte in4, byte out, boolean cacheU, boolean cacheV, String istr)
	{
		super(op, out);
		mrtype = MRINSTRUCTION_TYPE.Quaternary;
		instString = istr;
		
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		_input4 = in4;
		
		_cacheU = cacheU;
		_cacheV = cacheV;
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
	
	public byte getInput4() {
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
		String opcode = InstructionUtils.getOpCode(str);
		
		//validity check
		if (   !WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode)   //mapwsloss
			&& !WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode)  //redwsloss
			&& !WeightedSigmoid.OPCODE.equalsIgnoreCase(opcode)    //mapwsigmoid
			&& !WeightedSigmoidR.OPCODE.equalsIgnoreCase(opcode) ) //redwsigmoid
		{
			throw new DMLRuntimeException("Unexpected opcode in QuaternaryInstruction: " + str);
		}
		
		//instruction parsing
		if(    WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode)    //wsloss
			|| WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode) )
		{
			boolean isRed = WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode);
			
			//check number of fields (4 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields ( str, 8 );
			else
				InstructionUtils.checkNumFields ( str, 6 );
				
			//parse instruction parts (without exec type)
			String[] parts = InstructionUtils.getInstructionParts(str);
			
			byte in1 = Byte.parseByte(parts[1]);
			byte in2 = Byte.parseByte(parts[2]);
			byte in3 = Byte.parseByte(parts[3]);
			byte in4 = Byte.parseByte(parts[4]);
			byte out = Byte.parseByte(parts[5]);
			WeightsType wtype = WeightsType.valueOf(parts[6]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[7]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[8]) : true;
			
			return new QuaternaryInstruction(new QuaternaryOperator(wtype), in1, in2, in3, in4, out, cacheU, cacheV, str);	
		}
		else //wsigmoid
		{
			boolean isRed = WeightedSigmoidR.OPCODE.equalsIgnoreCase(opcode);
			
			//check number of fields (3 inputs, output, type)
			if( isRed )
				InstructionUtils.checkNumFields ( str, 7 );
			else
				InstructionUtils.checkNumFields ( str, 5 );
				
			//parse instruction parts (without exec type)
			String[] parts = InstructionUtils.getInstructionParts(str);
			
			byte in1 = Byte.parseByte(parts[1]);
			byte in2 = Byte.parseByte(parts[2]);
			byte in3 = Byte.parseByte(parts[3]);
			byte out = Byte.parseByte(parts[4]);
			WSigmoidType wtype = WSigmoidType.valueOf(parts[5]);
			
			//in mappers always through distcache, in reducers through distcache/shuffle
			boolean cacheU = isRed ? Boolean.parseBoolean(parts[6]) : true;
			boolean cacheV = isRed ? Boolean.parseBoolean(parts[7]) : true;
			
			return new QuaternaryInstruction(new QuaternaryOperator(wtype), in1, in2, in3, (byte)-1, out, cacheU, cacheV, str);
		}	
	}
	
	/**
	 * 
	 * @param inst
	 * @param index
	 * @return
	 */
	public static boolean isDistCacheOnlyIndex( String inst, byte index )
	{
		boolean ret = false;
		
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		String opcode = parts[1];
		byte in1 = Byte.parseByte(parts[2].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in2 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in3 = Byte.parseByte(parts[4].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in4 = Byte.parseByte(parts[5].split(Instruction.DATATYPE_PREFIX)[0]);
		
		if(    WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode) 
			|| WeightedSigmoid.OPCODE.equalsIgnoreCase(opcode) ) 
		{
			ret = (index==in2 && index!=in1 && index!=in4) 
				|| (index==in3 && index!=in1 && index!=in4);
		}
		else 
		{
			int off = WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode) ? 8 : 7;
			boolean cacheU = Boolean.parseBoolean(parts[off]);
			boolean cacheV = Boolean.parseBoolean(parts[off+1]);
			ret = (cacheU && index==in2 && index!=in1 && index!=in4) 
				|| (cacheV && index==in3 && index!=in1 && index!=in4);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param inst
	 * @param indexes
	 */
	public static void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		String opcode = parts[1];
		byte in1 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in2 = Byte.parseByte(parts[4].split(Instruction.DATATYPE_PREFIX)[0]);
		if(    WeightedSquaredLoss.OPCODE.equalsIgnoreCase(opcode) 
			|| WeightedSigmoid.OPCODE.equalsIgnoreCase(opcode) ) 
		{
			indexes.add(in1);
			indexes.add(in2);
		}
		else
		{
			int off = WeightedSquaredLossR.OPCODE.equalsIgnoreCase(opcode) ? 8 : 7;
			if( Boolean.parseBoolean(parts[off]) )
				indexes.add(in1);
			if( Boolean.parseBoolean(parts[off+1]) )
				indexes.add(in2);
		}
	}
	
	@Override
	public byte[] getInputIndexes() 
	{
		QuaternaryOperator qop = (QuaternaryOperator)optr;
		if( qop.wtype1 == null || qop.wtype1==WeightsType.NONE )
			return new byte[]{_input1, _input2, _input3};
		else
			return new byte[]{_input1, _input2, _input3, _input4};
	}

	@Override
	public byte[] getAllIndexes() 
	{
		QuaternaryOperator qop = (QuaternaryOperator)optr;
		if( qop.wtype1 == null || qop.wtype1==WeightsType.NONE )
			return new byte[]{_input1, _input2, _input3, output};
		else
			return new byte[]{_input1, _input2, _input3, _input4, output};
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
				//Step 1: prepare inputs and output
				if( imv==null )
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
				
				//Step 2: get remaining inputs: Wij, Ui, Vj		
				MatrixValue Xij = inVal;
				
				//get Wij if existing (null of WeightsType.NONE or WSigmoid any type)
				IndexedMatrixValue iWij = cachedValues.getFirst(_input4); 
				MatrixValue Wij = (iWij!=null) ? iWij.getValue() : null;
				
				//get Ui and Vj, potentially through distributed cache
				MatrixValue Ui = (!_cacheU) ? cachedValues.getFirst(_input2).getValue()     //U
					             : MRBaseForCommonInstructions.dcValues.get(_input2)
					                .getDataBlock((int)inIx.getRowIndex(), 1).getValue();	
				MatrixValue Vj = (!_cacheV) ? cachedValues.getFirst(_input3).getValue()     //t(V)
			                     : MRBaseForCommonInstructions.dcValues.get(_input3)
			                        .getDataBlock((int)inIx.getColumnIndex(), 1).getValue(); 
				//handle special input case: //V through shuffle -> t(V)
				if( Ui.getNumColumns()!=Vj.getNumColumns() ) { 
					Vj = LibMatrixReorg.reorg((MatrixBlock) Vj, new MatrixBlock(Vj.getNumColumns(), Vj.getNumRows(), Vj.isInSparseFormat()), new ReorgOperator(SwapIndex.getSwapIndexFnObject()));
				}
				
				//Step 3: process instruction
				Xij.quaternaryOperations((QuaternaryOperator)optr, Ui, Vj, Wij, outVal);
				if( ((QuaternaryOperator)optr).wtype1 != null ) 
					outIx.setIndexes(1, 1); //wsloss
				else
					outIx.setIndexes(inIx); //wsigmoid
				
				//put the output value in the cache
				if(iout==tempValue)
					cachedValues.add(output, iout);
			}
	}
}

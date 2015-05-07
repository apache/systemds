/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.ArrayList;

import com.ibm.bi.dml.lops.WeightedSquaredLoss.WeightsType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;
import com.ibm.bi.dml.runtime.matrix.mapred.MRBaseForCommonInstructions;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;
import com.ibm.bi.dml.runtime.matrix.operators.SimpleOperator;

/**
 * 
 */
public class QuaternaryInstruction extends MRInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private WeightsType _wType = null;
	
	private byte _input1 = -1;
	private byte _input2 = -1;
	private byte _input3 = -1;	
	private byte _input4 = -1;
	
	/**
	 * 
	 * @param type
	 * @param in1
	 * @param in2
	 * @param out
	 * @param istr
	 */
	public QuaternaryInstruction(Operator op, WeightsType type, byte in1, byte in2, byte in3, byte in4, byte out, String istr)
	{
		super(op, out);
		
		_wType = type;
		
		_input1 = in1;
		_input2 = in2;
		_input3 = in3;
		_input4 = in4;
		
		mrtype = MRINSTRUCTION_TYPE.Quaternary;
		instString = istr;
	}
	
	public WeightsType getWeightsType()
	{
		return _wType;
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
		//check number of fields (4 inputs, output, type)
		InstructionUtils.checkNumFields ( str, 6 );
		
		//parse instruction parts (without exec type)
		String[] parts = InstructionUtils.getInstructionParts(str);
		
		String opcode = parts[0];
		if ( !opcode.equalsIgnoreCase("mapwsloss") ) {
			throw new DMLRuntimeException("Unexpected opcode in QuaternaryInstruction: " + str);
		}
		
		byte in1 = Byte.parseByte(parts[1]);
		byte in2 = Byte.parseByte(parts[2]);
		byte in3 = Byte.parseByte(parts[3]);
		byte in4 = Byte.parseByte(parts[4]);
		byte out = Byte.parseByte(parts[5]);
		WeightsType wtype = WeightsType.valueOf(parts[6]);
		
		return new QuaternaryInstruction(new SimpleOperator(null), wtype, in1, in2, in3, in4, out, str);
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
		byte in1 = Byte.parseByte(parts[2].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in2 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in3 = Byte.parseByte(parts[4].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in4 = Byte.parseByte(parts[5].split(Instruction.DATATYPE_PREFIX)[0]);
		ret = (index==in2 && index!=in1 && index!=in4) || (index==in3 && index!=in1 && index!=in4);
		
		return ret;
	}
	
	public static void addDistCacheIndex( String inst, ArrayList<Byte> indexes )
	{
		//parse instruction parts (with exec type)
		String[] parts = inst.split(Instruction.OPERAND_DELIM);
		byte in1 = Byte.parseByte(parts[3].split(Instruction.DATATYPE_PREFIX)[0]);
		byte in2 = Byte.parseByte(parts[4].split(Instruction.DATATYPE_PREFIX)[0]);
		indexes.add(in1);
		indexes.add(in2);
	}
	
	@Override
	public byte[] getInputIndexes() 
	{
		if( _wType==WeightsType.NONE )
			return new byte[]{_input1, _input2, _input3};
		else
			return new byte[]{_input1, _input2, _input3, _input4};
	}

	@Override
	public byte[] getAllIndexes() 
	{
		if( _wType==WeightsType.NONE )
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
				DistributedCacheInput dcInput2 = MRBaseForCommonInstructions.dcValues.get(_input2); //u
				DistributedCacheInput dcInput3 = MRBaseForCommonInstructions.dcValues.get(_input3); //t(v)
				
				MatrixBlock Xi = (MatrixBlock)inVal;
				MatrixBlock u = (MatrixBlock) dcInput2.getDataBlock((int)inIx.getRowIndex(), 1).getValue();
				MatrixBlock v = (MatrixBlock) dcInput3.getDataBlock((int)inIx.getColumnIndex(), 1).getValue();
				
				//process core block operation
				Xi.quaternaryOperations(optr, u, v, null, outVal, _wType);
				outIx.setIndexes(1, 1);
				
				//put the output value in the cache
				if(iout==tempValue)
					cachedValues.add(output, iout);
			}
	}
}

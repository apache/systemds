/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.mr;

import java.util.HashMap;

import com.ibm.bi.dml.lops.Ternary;
import com.ibm.bi.dml.lops.Ternary.OperationTypes;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue;
import com.ibm.bi.dml.runtime.matrix.data.OperationsOnMatrixValues;
import com.ibm.bi.dml.runtime.matrix.mapred.CachedValueMap;
import com.ibm.bi.dml.runtime.matrix.mapred.IndexedMatrixValue;


public class TernaryInstruction extends MRInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private OperationTypes _op;
	
	public byte input1;
	public byte input2;
	public byte input3;
	public double scalar_input2;
	public double scalar_input3;
	private long _outputDim1, _outputDim2;

	/**
	 * Single matrix input
	 * 
	 * @param op
	 * @param in1
	 * @param scalar_in2
	 * @param scalar_in3
	 * @param out
	 * @param istr
	 */
	public TernaryInstruction(OperationTypes op, byte in1, double scalar_in2, double scalar_in3, byte out, long outputDim1, long outputDim2, String istr)
	{
		super(null, out);
		mrtype = MRINSTRUCTION_TYPE.Ternary;
		_op = op;
		input1 = in1;
		scalar_input2 = scalar_in2;
		scalar_input3 = scalar_in3;
		_outputDim1 = outputDim1;
		_outputDim2 = outputDim2;
		instString = istr;
	}
	
	/**
	 * Two matrix inputs
	 * 
	 * @param op
	 * @param in1
	 * @param in2
	 * @param scalar_in3
	 * @param out
	 * @param istr
	 */
	public TernaryInstruction(OperationTypes op, byte in1, byte in2, double scalar_in3, byte out, long outputDim1, long outputDim2, String istr)
	{
		super(null, out);
		mrtype = MRINSTRUCTION_TYPE.Ternary;
		_op = op;
		input1 = in1;
		input2 = in2;
		scalar_input3 = scalar_in3;
		_outputDim1 = outputDim1;
		_outputDim2 = outputDim2;
		instString = istr;
	}
	
	/**
	 * Two matrix input 
	 * 
	 * @param op
	 * @param in1
	 * @param scalar_in2
	 * @param in3
	 * @param out
	 * @param istr
	 */
	public TernaryInstruction(OperationTypes op, byte in1, double scalar_in2, byte in3, byte out, long outputDim1, long outputDim2, String istr)
	{
		super(null, out);
		mrtype = MRINSTRUCTION_TYPE.Ternary;
		_op = op;
		input1 = in1;
		scalar_input2 = scalar_in2;
		input3 = in3;
		_outputDim1 = outputDim1;
		_outputDim2 = outputDim2;
		instString = istr;
	}
	
	/**
	 * Three matrix inputs
	 * 
	 * @param op
	 * @param in1
	 * @param in2
	 * @param in3
	 * @param out
	 * @param istr
	 */
	public TernaryInstruction(OperationTypes op, byte in1, byte in2, byte in3, byte out, long outputDim1, long outputDim2, String istr)
	{
		super(null, out);
		mrtype = MRINSTRUCTION_TYPE.Ternary;
		_op = op;
		input1 = in1;
		input2 = in2;
		input3 = in3;
		_outputDim1 = outputDim1;
		_outputDim2 = outputDim2;
		instString = istr;
	}
	
	public long getOutputDim1() {
		return _outputDim1;
	}
	
	public long getOutputDim2() {
		return _outputDim2;
	}
	
	public boolean knownOutputDims() {
		return (_outputDim1 >0 && _outputDim2>0);
	}
	
	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction ( String str ) 
		throws DMLRuntimeException 
	{		
		// example instruction string 
		// - ctabletransform:::0:DOUBLE:::1:DOUBLE:::2:DOUBLE:::3:DOUBLE 
		// - ctabletransformscalarweight:::0:DOUBLE:::1:DOUBLE:::1.0:DOUBLE:::3:DOUBLE 
		// - ctabletransformhistogram:::0:DOUBLE:::1.0:DOUBLE:::1.0:DOUBLE:::3:DOUBLE 
		// - ctabletransformweightedhistogram:::0:DOUBLE:::1:INT:::1:DOUBLE:::2:DOUBLE 
		
		//check number of fields
		InstructionUtils.checkNumFields ( str, 6 );
		
		//common setup
		byte in1, in2, in3, out;
		String[] parts = InstructionUtils.getInstructionParts ( str );		
		String opcode = parts[0];
		in1 = Byte.parseByte(parts[1]);
		long outputDim1 = (long) Double.parseDouble(parts[4]);
		long outputDim2 = (long) Double.parseDouble(parts[5]);
		out = Byte.parseByte(parts[6]);
		
		OperationTypes op = Ternary.getOperationType(opcode);
		
		switch( op )
		{
			case CTABLE_TRANSFORM: {
				in2 = Byte.parseByte(parts[2]);
				in3 = Byte.parseByte(parts[3]);
				return new TernaryInstruction(op, in1, in2, in3, out, outputDim1, outputDim2, str);
			}
			case CTABLE_TRANSFORM_SCALAR_WEIGHT: {
				in2 = Byte.parseByte(parts[2]);
				double scalar_in3 = Double.parseDouble(parts[3]);
				return new TernaryInstruction(op, in1, in2, scalar_in3, out, outputDim1, outputDim2, str);
			}
			case CTABLE_EXPAND_SCALAR_WEIGHT: {
				double scalar_in2 = Double.parseDouble(parts[2]);
				double type = Double.parseDouble(parts[3]); //used as type (1 left, 0 right)
				return new TernaryInstruction(op, in1, scalar_in2, type, out, outputDim1, outputDim2, str);
			}
			case CTABLE_TRANSFORM_HISTOGRAM: {
				double scalar_in2 = Double.parseDouble(parts[2]);
				double scalar_in3 = Double.parseDouble(parts[3]);
				return new TernaryInstruction(op, in1, scalar_in2, scalar_in3, out, outputDim1, outputDim2, str);
			}
			case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM: {
				double scalar_in2 = Double.parseDouble(parts[2]);
				in3 = Byte.parseByte(parts[3]);
				return new TernaryInstruction(op, in1, scalar_in2, in3, out, outputDim1, outputDim2, str);	
			}
			default:
				throw new DMLRuntimeException("Unrecognized opcode in Ternary Instruction: " + op);	
		}
	}
	
	public void processInstruction(Class<? extends MatrixValue> valueClass, CachedValueMap cachedValues, 
			            IndexedMatrixValue zeroInput, HashMap<Byte, HashMap<MatrixIndexes, Double>> cacheForCtable, HashMap<Byte, MatrixBlock> blockCacheForCtable, 
			            int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{	
		
		IndexedMatrixValue in1, in2, in3 = null;
		in1 = cachedValues.getFirst(input1);
		
		HashMap<MatrixIndexes, Double> ctableResult = null;
		MatrixBlock ctableResultBlock = null;
		
		if ( knownOutputDims() ) {
			if ( blockCacheForCtable != null ) {
				ctableResultBlock = blockCacheForCtable.get(output);
				if ( ctableResultBlock == null ) {
					// From MR, output of ctable is set to be sparse since it is built from a single input block.
					ctableResultBlock = new MatrixBlock((int)_outputDim1, (int)_outputDim2, true);
					blockCacheForCtable.put(output, ctableResultBlock);
				}
			}
			else {
				throw new DMLRuntimeException("Unexpected error in processing table instruction.");
			}
		}
		else {
			//prepare aggregation maps
			ctableResult=cacheForCtable.get(output);
			if(ctableResult==null)
			{
				ctableResult=new HashMap<MatrixIndexes, Double>();
				cacheForCtable.put(output, ctableResult);
			}
		}
		
		//get inputs and process instruction
		switch( _op )
		{
			case CTABLE_TRANSFORM: {
				in2 = cachedValues.getFirst(input2);
				in3 = cachedValues.getFirst(input3);
				if(in1==null || in2==null || in3 == null )
					return;	
				OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), in2.getIndexes(), in2.getValue(), 
						                                 in3.getIndexes(), in3.getValue(), ctableResult, ctableResultBlock, optr);
				break;
			}
			case CTABLE_TRANSFORM_SCALAR_WEIGHT: {
				// 3rd input is a scalar
				in2 = cachedValues.getFirst(input2);
				if(in1==null || in2==null )
					return;
				OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), in2.getIndexes(), in2.getValue(), 
						                                 scalar_input3, ctableResult, ctableResultBlock, optr);
				break;
			}
			case CTABLE_EXPAND_SCALAR_WEIGHT: {
				// 2nd and 3rd input is a scalar
				if(in1==null )
					return;
				OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), scalar_input2, (scalar_input3==1), 
						                                 blockRowFactor, ctableResult, ctableResultBlock, optr);
				break;
			}
			case CTABLE_TRANSFORM_HISTOGRAM: {
				// 2nd and 3rd inputs are scalars
				if(in1==null )
					return;	
				OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), scalar_input2, scalar_input3, ctableResult, ctableResultBlock, optr);
				break;
			}
			case CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM: {
				// 2nd and 3rd inputs are scalars
				in3 = cachedValues.getFirst(input3);
				if(in1==null || in3==null)
					return;
				OperationsOnMatrixValues.performTernary(in1.getIndexes(), in1.getValue(), scalar_input2, 
						                                 in3.getIndexes(), in3.getValue(), ctableResult, ctableResultBlock, optr);		
				break;
			}
			default:
				throw new DMLRuntimeException("Unrecognized opcode in Tertiary Instruction: " + instString);		
		}
	}
	
	@Override
	public void processInstruction(Class<? extends MatrixValue> valueClass,
			CachedValueMap cachedValues, IndexedMatrixValue tempValue, IndexedMatrixValue zeroInput, 
			int blockRowFactor, int blockColFactor)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		throw new DMLRuntimeException("This function should not be called!");
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		return new byte[]{input1, input2, input3, output};
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		return new byte[]{input1, input2, input3};
	}

}

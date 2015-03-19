/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class MapMult extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "mapmm";
	
	public enum CacheType {
		RIGHT,
		RIGHT_PART,
		LEFT,
		LEFT_PART;
		
		public boolean isRightCache(){
			return (this == RIGHT || this == RIGHT_PART);
		}
	}
	
	private CacheType _cacheType = null;
	private boolean _outputEmptyBlocks = true;
	
	//optional attribute for spark exec type
	private boolean _aggregate = true;
	
	/**
	 * Constructor to setup a partial Matrix-Vector Multiplication for MR
	 * 
	 * @param input
	 * @param op
	 * @return 
	 * @throws LopsException
	 */	
	public MapMult(Lop input1, Lop input2, DataType dt, ValueType vt, boolean rightCache, boolean partitioned, boolean emptyBlocks ) 
		throws LopsException 
	{
		super(Lop.Type.MapMult, dt, vt);		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		//setup mapmult parameters
		if( rightCache )
			_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
		else
			_cacheType = partitioned ? CacheType.LEFT_PART : CacheType.LEFT;
		_outputEmptyBlocks = emptyBlocks;
		
		//setup MR parameters 
		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.GMR);
		lps.addCompatibility(JobType.DATAGEN);
		lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
	}

	/**
	 * Constructor to setup a partial Matrix-Vector Multiplication for Spark
	 * 
	 * @param input1
	 * @param input2
	 * @param dt
	 * @param vt
	 * @param rightCache
	 * @param emptyBlocks
	 * @param aggregate
	 * @param et
	 * @throws LopsException
	 */
	public MapMult(Lop input1, Lop input2, DataType dt, ValueType vt, boolean rightCache, boolean partitioned, boolean emptyBlocks, boolean aggregate) 
		throws LopsException 
	{
		super(Lop.Type.MapMult, dt, vt);		
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		//setup mapmult parameters
		if( rightCache )
			_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
		else
			_cacheType = partitioned ? CacheType.LEFT_PART : CacheType.LEFT;
		_outputEmptyBlocks = emptyBlocks;
		_aggregate = aggregate;
		
		//setup MR parameters 
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		lps.setProperties( inputs, ExecType.SPARK, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}

	public String toString() {
		return "Operation = MapMM";
	}
	
	@Override
	public String getInstructions(int input_index1, int input_index2, int output_index)
	{
		//MR instruction generation
		
		StringBuilder sb = new StringBuilder();
		
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(OPCODE);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(1).prepInputOperand(input_index2));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( this.prepOutputOperand(output_index));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_cacheType);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_outputEmptyBlocks);
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output)
	{
		//Spark instruction generation
		
		StringBuilder sb = new StringBuilder();
		
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(OPCODE);
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append(Lop.OPERAND_DELIMITOR);
		
		
		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append( this.prepOutputOperand(output));
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(_cacheType);
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(_outputEmptyBlocks);
		sb.append(Lop.OPERAND_DELIMITOR);
		
		sb.append(_aggregate);
		
		return sb.toString();
	}

	@Override
	public boolean usesDistributedCache() 
	{
		return true;
	}
	
	@Override
	public int[] distributedCacheInputIndex() 
	{	
		switch( _cacheType )
		{
			// first input is from distributed cache
			case LEFT:
			case LEFT_PART: 
				return new int[]{1};
			
			// second input is from distributed cache
			case RIGHT:
			case RIGHT_PART: 
				return new int[]{2};
		}
				
		return new int[]{-1}; //error
	}
}

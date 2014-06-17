/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


public class AppendM extends Lop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "mappend";
	
	public enum CacheType {
		RIGHT,
		RIGHT_PART,
	}

	private CacheType _cacheType = null;
	
	
	public AppendM(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, boolean partitioned) 
	{
		super(Lop.Type.Append, dt, vt);
		init(input1, input2, input3, dt, vt);
		
		//partitioned right input
		_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
	}
	
	public void init(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt) 
	{
		this.addInput(input1);
		input1.addOutput(this);

		this.addInput(input2);
		input2.addOutput(this);
		
		this.addInput(input3);
		input3.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		lps.addCompatibility(JobType.GMR);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
	}
	
	@Override
	public String toString() {

		return "Operation = AppendM"; 
	}

	//called when append executes in MR
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( this.lps.execType );
		sb.append( OPERAND_DELIMITOR );
		sb.append( "mappend" );
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input_index2+""));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepScalarInputOperand(getExecType()));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output_index+"") );
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_cacheType);
		
		return sb.toString();	
	}

	public boolean usesDistributedCache() {
		return true;
	}
	
	public int distributedCacheInputIndex() {
		return 2; // second input is from distributed cache
	}
}

/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import java.util.StringTokenizer;

import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class StringInitCPInstruction extends UnaryCPInstruction
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String DELIM = " ";
	
	private long _rlen = -1;
	private long _clen = -1;
	private String _data = null;
		
	public StringInitCPInstruction( Operator op, CPOperand in, CPOperand out, long rows, long cols, int rpb, int cpb, 
			                        String data, String opcode, String inst) 
	{
		super(op, in, out, opcode, inst);
		_rlen = rows;
		_clen = cols;
		_data = data;
	}

	/**
	 * 
	 * @param str
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static Instruction parseInstruction(String str) 
		throws DMLRuntimeException 
	{
		String opcode = InstructionUtils.getOpCode(str);
		if( !opcode.equals(DataGen.SINIT_OPCODE) )
			throw new DMLRuntimeException("Unsupported opcode: "+opcode);
		
		//check fields
		InstructionUtils.checkNumFields( str, 6 );
		
		//parse instruction
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		String[] s = InstructionUtils.getInstructionPartsWithValueType ( str );
		out.split(s[s.length-1]); // ouput is specified by the last operand

		long rows = (s[1].contains( Lop.VARIABLE_NAME_PLACEHOLDER)?-1:Double.valueOf(s[1]).longValue());
		long cols = (s[2].contains( Lop.VARIABLE_NAME_PLACEHOLDER)?-1:Double.valueOf(s[2]).longValue());
        int rpb = Integer.parseInt(s[3]);
		int cpb = Integer.parseInt(s[4]);
		String data = s[5];
		
		return new StringInitCPInstruction(null, null, out, rows, cols, rpb, cpb, data, opcode, str);
	}
	
	@Override
	public void processInstruction( ExecutionContext ec )
		throws DMLRuntimeException
	{
		//setup output matrix
		String outName = output.get_name();
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
			outBlk.quickSetValue(rix, cix, dval);
		}
		
		//examine right output representation
		outBlk.recomputeNonZeros();
		outBlk.examSparsity();
		
		//put output into symbol table
		ec.setMatrixOutput(outName, outBlk);
	}
}

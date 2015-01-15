/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.io.MatrixWriterFactory;
import com.ibm.bi.dml.runtime.io.WriterBinaryBlock;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

/**
 * 
 * 
 */
public class DataPartitionCPInstruction extends UnaryCPInstruction
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private PDataPartitionFormat _pformat = null;
	
	public DataPartitionCPInstruction(Operator op, CPOperand in1, PDataPartitionFormat pformat, CPOperand out, String opcode, String istr)
	{
		super(op, in1, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.MMTSJ;
		_pformat = pformat;
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
		CPOperand in1 = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		
		InstructionUtils.checkNumFields ( str, 3 );
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		in1.split(parts[1]);
		out.split(parts[2]);
		PDataPartitionFormat pformat = PDataPartitionFormat.valueOf(parts[3]);
		 
		if(!opcode.equalsIgnoreCase("partition"))
			throw new DMLRuntimeException("Unknown opcode while parsing an DataPartitionCPInstruction: " + str);
		else
			return new DataPartitionCPInstruction(new Operator(true), in1, pformat, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec)
		throws DMLUnsupportedOperationException, DMLRuntimeException 
	{
		//get input
		MatrixObject moIn = (MatrixObject) ec.getVariable(input1.get_name());
		MatrixBlock mb = moIn.acquireRead();
		
		//execute operations 
		MatrixObject moOut = (MatrixObject) ec.getVariable(output.get_name());		
		String fname = moOut.getFileName();
		moOut.setPartitioned(_pformat, -1); //modify meta data output
		try
		{
			//write matrix partitions to hdfs
			WriterBinaryBlock writer = (WriterBinaryBlock) MatrixWriterFactory.createMatrixWriter(OutputInfo.BinaryBlockOutputInfo);
			writer.writePartitionedBinaryBlockMatrixToHDFS(
					   new Path(fname), new JobConf(), mb, moIn.getNumRows(), moIn.getNumColumns(), 
					   (int)moIn.getNumRowsPerBlock(), (int)moIn.getNumColumnsPerBlock(), _pformat);
			
			//ensure correctness of output characteristics (required if input unknown during compile and no recompile)
			MatrixCharacteristics mc = new MatrixCharacteristics(moIn.getNumRows(), moIn.getNumColumns(), (int)moIn.getNumRowsPerBlock(), (int)moIn.getNumColumnsPerBlock(), moIn.getNnz()); 
			MatrixFormatMetaData meta = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			moOut.setMetaData(meta);
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Failed to execute data partitioning instruction.", ex);
		}
		
		//release input
		ec.releaseMatrixInput(input1.get_name());		
	}
}

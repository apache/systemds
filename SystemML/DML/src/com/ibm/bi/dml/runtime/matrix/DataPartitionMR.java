/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerRemoteMR;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;

public class DataPartitionMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	public static boolean partitioned = false;
	public static PDataPartitionFormat pformat = PDataPartitionFormat.ROW_BLOCK_WISE_N;
	public static int N = DMLTranslator.DMLBlockSize*4;
	
	public static JobReturn runJob(MRJobInstruction jobinst, MatrixObject[] inputMatrices, String shuffleInst, byte[] resultIndices, MatrixObject[] outputMatrices, int numReducers, int replication) throws DMLRuntimeException {
		MatrixCharacteristics[] sts = new MatrixCharacteristics[outputMatrices.length];
		
		processPartitionInstructions(shuffleInst, inputMatrices, resultIndices, outputMatrices, numReducers, replication, sts);
		
		JobReturn ret = new JobReturn(sts, true);
		return ret;
	}
	
	private static void processPartitionInstructions(String shuffleInst, MatrixObject[] inputMatrices, byte[] resultIndices, MatrixObject[] outputMatrices, int numReducers, int replication, MatrixCharacteristics[] sts) throws DMLRuntimeException {
		int i=0;
		for(String inst : shuffleInst.split(Instruction.INSTRUCTION_DELIM)) {
			if( InstructionUtils.getOpCode(inst).equalsIgnoreCase("partition") ) {
				//long begin = System.currentTimeMillis();
				String[] parts = InstructionUtils.getInstructionParts(inst);
				int input_index = Integer.parseInt(parts[1]);
				int output_index = Integer.parseInt(parts[2]);
				
				MatrixObject in = inputMatrices[input_index];
				MatrixObject out = outputMatrices[findResultIndex(resultIndices, output_index)];
				
				PDataPartitionFormat format = pformat;
				int _n = N; //16000
				
				DataPartitioner dpart = null;
				//if( dp == PDataPartitioner.LOCAL )
				//	dpart = new DataPartitionerLocal(format, _n);
				//else if( dp == PDataPartitioner.REMOTE_MR )
				dpart = new DataPartitionerRemoteMR(format, _n, 4, numReducers, replication, 1, false);
				
				out = dpart.createPartitionedMatrixObject(in, out, true);
				
				sts[i] = ((MatrixDimensionsMetaData)out.getMetaData()).getMatrixCharacteristics();
				//System.out.println("Partioning complete: " + (System.currentTimeMillis()-begin)/1000 + " sec");
				i++;
				partitioned = true;
			}
		}
	}

	private static int findResultIndex(byte[] resultIndices, int output_index) {
		for(int i=0; i < resultIndices.length; i++) {
			if(resultIndices[i] == output_index) 
				return i;
		}
		return -1;
	}
	
}

package com.ibm.bi.dml.runtime.matrix;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerRemoteMR;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.utils.DMLRuntimeException;

public class DataPartitionMR {

	public static boolean partitioned = false;
	public static PDataPartitionFormat pformat = PDataPartitionFormat.ROW_BLOCK_WISE_N;
	public static int N = DMLTranslator.DMLBlockSize*2;
	
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
				
				out = dpart.createPartitionedMatrixObject(in, out);
				
				sts[i] = ((MatrixDimensionsMetaData)out.getMetaData()).getMatrixCharacteristics();
				System.out.println("Partioning complete!!");
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

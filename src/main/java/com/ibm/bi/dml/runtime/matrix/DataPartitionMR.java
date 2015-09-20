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


package com.ibm.bi.dml.runtime.matrix;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.DataPartitionerRemoteMR;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.mapred.DistributedCacheInput;

public class DataPartitionMR 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private DataPartitionMR() {
		//prevent instantiation via private constructor
	}
	
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
				
				PDataPartitionFormat pformat = PDataPartitionFormat.valueOf(parts[3]);
				long rlen = in.getNumRows();
				long clen = in.getNumColumns();
				long brlen = in.getNumRowsPerBlock();
				long bclen = in.getNumColumnsPerBlock();
				long N = -1;
				switch( pformat )
				{
					case ROW_BLOCK_WISE_N:
					{
						long numRowBlocks = (long)Math.ceil(((double)DistributedCacheInput.PARTITION_SIZE)/clen/brlen); 
						N = numRowBlocks * brlen;
						break;
					}
					case COLUMN_BLOCK_WISE_N:
					{
						long numColBlocks = (long)Math.ceil(((double)DistributedCacheInput.PARTITION_SIZE)/rlen/bclen); 
						N = numColBlocks * bclen;
						break;
					}
					
					default: 
						throw new DMLRuntimeException("Unsupported partition format for distributed cache input: "+pformat);
				}
				
				DataPartitioner dpart = new DataPartitionerRemoteMR(pformat, (int)N, -1, numReducers, replication, 4, false, true);
				out = dpart.createPartitionedMatrixObject(in, out, true);
				
				sts[i] = out.getMatrixCharacteristics();
				i++;
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

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class ReblockOOCInstruction extends ComputationOOCInstruction {
	private int blen;

	private ReblockOOCInstruction(Operator op, CPOperand in, CPOperand out, 
		int br, int bc, String opcode, String instr)
	{
		super(OOCType.Reblock, op, in, out, opcode, instr);
		blen = br;
		blen = bc;
	}

	public static ReblockOOCInstruction parseInstruction(String str) {
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equals(Opcodes.RBLK.toString()))
			throw new DMLRuntimeException("Incorrect opcode for ReblockOOCInstruction:" + opcode);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen=Integer.parseInt(parts[3]);
		return new ReblockOOCInstruction(null, in, out, blen, blen, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		//set the output characteristics
		MatrixObject min = ec.getMatrixObject(input1);
		DataCharacteristics mc = ec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = ec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), blen, mc.getNonZeros());

		//get the source format from the meta data
		//MetaDataFormat iimd = (MetaDataFormat) min.getMetaData();
		//TODO support other formats than binary 
		
		//create queue, spawn thread for asynchronous reading, and return
		OOCStream<IndexedMatrixValue> q = createWritableStream();
		submitOOCTask(() -> readBinaryBlock(q, min.getFileName()), q);
		
		MatrixObject mout = ec.getMatrixObject(output);
		mout.setStreamHandle(q);
	}
	
	@SuppressWarnings("resource")
	private void readBinaryBlock(OOCStream<IndexedMatrixValue> q, String fname) {
		try {
			//prepare file access
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());	
			Path path = new Path( fname ); 
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			
			//check existence and non-empty file
			MatrixReader.checkValidInputFile(fs, path); 
			
			//core reading
			for( Path lpath : IOUtilFunctions.getSequenceFilePaths(fs, path) ) { //1..N files 
				//directly read from sequence files (individual partfiles)
				try( SequenceFile.Reader reader = new SequenceFile
					.Reader(job, SequenceFile.Reader.file(lpath)) )
				{
					MatrixIndexes key = new MatrixIndexes();
					MatrixBlock value = new MatrixBlock();
					while( reader.next(key, value) )
						q.enqueue(new IndexedMatrixValue(key, new MatrixBlock(value)));
				}
			}
			q.closeInput();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}

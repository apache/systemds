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

package org.apache.sysds.runtime.instructions.cpfile;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.IndexRange;

/**
 * This instruction is used if a single partition is too large to fit in memory.
 * Hence, the partition is not read but we just return a new matrix with the
 * respective partition file name. For this reason this is a no-op but due to
 * the requirement for direct partition access only applicable for ROWWISE and
 * COLWISE partition formats. 
 * 
 */
public final class MatrixIndexingCPFileInstruction extends IndexingCPInstruction {

	private MatrixIndexingCPFileInstruction(CPOperand in, CPOperand rl, CPOperand ru, CPOperand cl,
			CPOperand cu, CPOperand out, String opcode, String istr) {
		super(in, rl, ru, cl, cu, out, opcode, istr);
	}

	public static MatrixIndexingCPFileInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		
		if ( opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) ) {
			if ( parts.length == 7 ) {
				CPOperand in, rl, ru, cl, cu, out;
				in = new CPOperand(parts[1]);
				rl = new CPOperand(parts[2]);
				ru = new CPOperand(parts[3]);
				cl = new CPOperand(parts[4]);
				cu = new CPOperand(parts[5]);
				out = new CPOperand(parts[6]);
				return new MatrixIndexingCPFileInstruction(in, rl, ru, cl, cu, out, opcode, str);
			}
			else {
				throw new DMLRuntimeException("Invalid number of operands in instruction: " + str);
			}
		} 
		else if ( parts[0].equalsIgnoreCase(Opcodes.LEFT_INDEX.toString())) {
			throw new DMLRuntimeException("Invalid opcode while parsing a MatrixIndexingCPFileInstruction: " + str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode while parsing a MatrixIndexingCPFileInstruction: " + str);
		}
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		IndexRange ixrange = getIndexRange(ec).add(1);
		MatrixObject mo = ec.getMatrixObject(input1.getName());
		
		if( mo.isPartitioned() && opcode.equalsIgnoreCase(Opcodes.RIGHT_INDEX.toString()) )
		{
			MetaDataFormat meta = (MetaDataFormat)mo.getMetaData();
			DataCharacteristics mc = meta.getDataCharacteristics();
			String pfname = mo.getPartitionFileName( ixrange, mc.getBlocksize());
			
			if( HDFSTool.existsFileOnHDFS(pfname) ) { //default
				//create output matrix object
				MatrixObject mobj = new MatrixObject(mo.getValueType(), pfname );
				DataCharacteristics mcNew = null;
				switch( mo.getPartitionFormat() ) {
					case ROW_WISE:
						mcNew = new MatrixCharacteristics( 1, mc.getCols(), mc.getBlocksize(), mc.getBlocksize() );
						break;
					case ROW_BLOCK_WISE_N:
						mcNew = new MatrixCharacteristics( mo.getPartitionSize(), mc.getCols(), mc.getBlocksize(), mc.getBlocksize() );
						break;
					case COLUMN_WISE:
						mcNew = new MatrixCharacteristics( mc.getRows(), 1, mc.getBlocksize(), mc.getBlocksize() );
						break;
					case COLUMN_BLOCK_WISE_N:
						mcNew = new MatrixCharacteristics( mc.getRows(), mo.getPartitionSize(), mc.getBlocksize(), mc.getBlocksize() );
						break;
					default:
						throw new DMLRuntimeException("Unsupported partition format for CP_FILE "+Opcodes.RIGHT_INDEX.toString()+": "+ mo.getPartitionFormat());
				}
				
				MetaDataFormat metaNew = new MetaDataFormat(mcNew, meta.getFileFormat());
				mobj.setMetaData(metaNew);
				
				//note: disable cleanup to ensure that the partitioning file is not deleted 
				//(e.g., for nested loops or reused partitioned matrices across loops)
				mobj.enableCleanup(false);
				
				//put output object into symbol table
				ec.setVariable(output.getName(), mobj);
			}
			else { //empty matrix partition
				//note: for binary cell data partitioning empty partitions are not materialized
				MatrixBlock resultBlock = mo.readMatrixPartition( ixrange );
				ec.setMatrixOutput(output.getName(), resultBlock);
			}
		}
		else {
			throw new DMLRuntimeException("Invalid opcode or index predicate "
				+ "for MatrixIndexingCPFileInstruction: " + instString);
		}
	}
}

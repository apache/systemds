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

package org.apache.sysds.runtime.controlprogram.caching.prescientbuffer;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.ooc.ReblockOOCInstruction;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.ArrayList;

/**
 * IOTraceGenerator is responsible for analyzing the program plan (LOP DAG)
 * and generating the predictive I/O trace, before runtime.
 */
public class IOTraceGenerator {

	/**
	 * Generate the IOTrace for the execution plan.
	 * This is utilized by ExecutionContext
	 *
	 * @param program the entire program
	 * @return IOTrace object with trace data
	 */
	public static IOTrace generateTrace(Program program, ExecutionContext ec) {
		IOTrace _trace = new IOTrace();

		// Use a long array as a "pass-by-reference" wrapper for the logical time
		// so it can be incremented correctly inside the recursive calls.
		long[] logicalTime = new long[]{0};

		// Start the recursive traversal
		traverseProgramBlocks(program.getProgramBlocks(), ec, _trace, logicalTime);

		return _trace;
	}

	/**
	 * Recursively traverses a list of program blocks.
	 *
	 * @param programBlocks The list of blocks to traverse
	 * @param ec The ExecutionContext
	 * @param trace The trace object to populate
	 * @param logicalTime A pass-by-reference counter for logical time
	 */
	private static void traverseProgramBlocks(ArrayList<ProgramBlock> programBlocks, ExecutionContext ec, IOTrace trace, long[] logicalTime) {

		if (programBlocks == null) {
			return;
		}

		for (ProgramBlock pb : programBlocks) {
			if (pb instanceof BasicProgramBlock) {
				// --- Base Case ---
				// This block has instructions, so process them.
				BasicProgramBlock bpb = (BasicProgramBlock) pb;
				for (Instruction inst : bpb.getInstructions()) {
					logicalTime[0]++; // Increment logical time for each instruction
					processInstruction(inst, ec, trace, logicalTime[0]);
				}
			}
			else if (pb instanceof IfProgramBlock) {
				// --- Recursive Step ---
				// Traverse into the 'if' and 'else' bodies
				IfProgramBlock ifpb = (IfProgramBlock) pb;
				traverseProgramBlocks(ifpb.getChildBlocksIfBody(), ec, trace, logicalTime);
				traverseProgramBlocks(ifpb.getChildBlocksElseBody(), ec, trace, logicalTime);
			}
			else if (pb instanceof WhileProgramBlock) {
				// --- Recursive Step ---
				// For a static trace, we can only traverse the body once.
				// A more advanced tracer might try to unroll N times.
				WhileProgramBlock wpb = (WhileProgramBlock) pb;
				traverseProgramBlocks(wpb.getChildBlocks(), ec, trace, logicalTime);
			}
			else if (pb instanceof ForProgramBlock) {
				// --- Recursive Step ---
				// Similar to While, just traverse the body once for a static trace.
				ForProgramBlock fpb = (ForProgramBlock) pb;
				traverseProgramBlocks(fpb.getChildBlocks(), ec, trace, logicalTime);
			}
			else if (pb instanceof FunctionProgramBlock) {
				// --- Recursive Step ---
				// Traverse into the function body
				FunctionProgramBlock fnpb = (FunctionProgramBlock) pb;
				traverseProgramBlocks(fnpb.getChildBlocks(), ec, trace, logicalTime);
			}
		}
	}

	/**
	 * Processes a single instruction and records its I/O access pattern in the trace.
	 *
	 * @param inst The instruction to process
	 * @param ec The ExecutionContext
	 * @param trace The trace object to populate
	 * @param logicalTime The logical time of this instruction
	 */
	private static void processInstruction(Instruction inst, ExecutionContext ec, IOTrace trace, long logicalTime) {

		// --- This is your specific logic for OOC instructions ---

		if (inst instanceof ReblockOOCInstruction) {
			ReblockOOCInstruction rblk = (ReblockOOCInstruction) inst;
			CPOperand input = rblk.input1;

			// We need the file name and data characteristics from the metadata
			String fname = ec.getMatrixObject(input).getFileName();
			DataCharacteristics mc = ec.getDataCharacteristics(input.getName());

			if (mc == null || !mc.dimsKnown()) {
				throw new DMLRuntimeException("OOC Trace Generator: DataCharacteristics not available for " + input.getName());
			}

			long numRowBlocks = mc.getNumRowBlocks();
			long numColBlocks = mc.getNumColBlocks();

			for (long i = 1; i <= numRowBlocks; i++) {
				for (long j = 1; j <= numColBlocks; j++) {

					String blockID = createBlockID(fname, i, j);
					trace.recordAccess(blockID);
				}
			}
		}

		// (Transpose, MatrixVector, Tee, etc.) to build the full trace.
		// else if (inst instanceof MatrixVectorOOCInstruction) {
		//     // ... handle matrix-vector read pattern ...
		// }
	}

	private static String createBlockID(String fname, long rowIndex, long colIndex) {
		System.out.println(fname + "_" + rowIndex + "_" + colIndex);
		return fname + "_" + rowIndex + "_" + colIndex;
	}
}

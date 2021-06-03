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

package org.apache.sysds.runtime.lineage;


import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Queue;
import java.util.LinkedList;

public class LineageDebugger {
	public static final int POS_NAN = 0;
	public static final int POS_POSITIVE_INFINITY = 1;
	public static final int POS_NEGATIVE_INFINITY = 2;
	
	public static void maintainSpecialValueBits(Lineage lineage, Instruction inst, ExecutionContext ec) {
		ArrayList<CPOperand> outputs = new ArrayList<>();
		
		// Only CP instructions are supported right now
		CPOperand singleOutput =
				inst instanceof ComputationCPInstruction ? ((ComputationCPInstruction) inst).getOutput() :
				inst instanceof BuiltinNaryCPInstruction ? ((BuiltinNaryCPInstruction) inst).getOutput() :
				inst instanceof SqlCPInstruction ? ((SqlCPInstruction) inst).getOutput() :
				inst instanceof VariableCPInstruction ? ((VariableCPInstruction) inst).getOutput() :
				null;
		if (singleOutput != null)
			outputs.add(singleOutput);
		
		Collection<CPOperand> multiOutputs =
				inst instanceof MultiReturnBuiltinCPInstruction ? ((MultiReturnBuiltinCPInstruction) inst).getOutputs() :
				inst instanceof MultiReturnParameterizedBuiltinCPInstruction ? ((MultiReturnParameterizedBuiltinCPInstruction) inst).getOutputs() :
				null;
		if (multiOutputs != null)
			outputs.addAll(multiOutputs);
		
		for (CPOperand output : outputs) {
			CacheableData<MatrixBlock> cd = ec.getMatrixObject(output);
			MatrixBlock mb = cd.acquireReadAndRelease();
			LineageItem li = lineage.get(output);
			
			updateSpecialValueBit(mb, li, POS_NAN, Double.NaN);
			updateSpecialValueBit(mb, li, POS_POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
			updateSpecialValueBit(mb, li, POS_NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
		}
	}
	
	public static LineageItem firstOccurrenceOfNR(LineageItem li, int pos) {
		if (!li.getSpecialValueBit(pos))
			return null;
		
		LineageItem tmp;
		Queue<LineageItem> q = new LinkedList<>();
		q.add(li);
		
		while ((tmp = q.poll()) != null) {
			if (tmp.isVisited())
				continue;
			
			if (tmp.getInputs() != null) {
				boolean flag = false;
				for (LineageItem in : tmp.getInputs()) {
					flag |= in.getSpecialValueBit(pos);
					q.add(in);
				}
				if (!flag)
					break;
			}
			tmp.setVisited(true);
		}
		li.resetVisitStatusNR();
		return tmp;
	}
	
	private static void updateSpecialValueBit(MatrixBlock mb, LineageItem li, int pos, double pattern) {
		boolean flag = false;
		for (LineageItem input : li.getInputs()) {
			if (input.getSpecialValueBit(pos)) {
				flag = true;
				break;
			}
		}
		flag |= mb.containsValue(pattern);
		li.setSpecialValueBit(pos, flag);
	}
}

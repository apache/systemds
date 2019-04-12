/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.lineage;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.utils.Explain;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class Lineage {
	
	private static Map<String, LineageItem> lineage_traces = new HashMap<>();
	
	private Lineage() {
	}
	
	public static void trace(Instruction inst, ExecutionContext ec) {
		if (inst instanceof VariableCPInstruction) {
			VariableCPInstruction vcp_inst = ((VariableCPInstruction) inst);
			switch (vcp_inst.getVariableOpcode()) {
				case CreateVariable: {
					createVariableInstruction(vcp_inst);
					break;
				}
				case RemoveVariable: {
					removeInstruction(vcp_inst);
					break;
				}
				case AssignVariable:
				case CopyVariable: {
					copyInstruction(vcp_inst);
					break;
				}
				case Write: {
					writeInstruction(vcp_inst, ec);
					break;
				}
				case MoveVariable: {
					moveInstruction(vcp_inst);
					break;
				}
				default:
					throw new DMLRuntimeException("Unknown VariableCPInstruction (" + inst.getOpcode() + ") traced.");
			}
		} else if (inst instanceof LineageTraceable) {
			addLineageItem(((LineageTraceable) inst).getLineageItem());
		} else
			throw new DMLRuntimeException("Unknown Instruction (" + inst.getOpcode() + ") traced.");
	}
	
	public static void removeLineageItem(String key) {
		if (!lineage_traces.containsKey(key))
			return;
		
		removeInputLinks(lineage_traces.get(key));
		lineage_traces.remove(key);
	}
	
	public static void addLineageItem(LineageItem li) {
		if (lineage_traces.get(li.getKey()) != null) {
			removeInputLinks(lineage_traces.get(li.getKey()));
			lineage_traces.remove(li.getKey());
		}
		lineage_traces.put(li.getKey(), li);
	}
	
	public static LineageItem getOrCreate(CPOperand variable) {
		if (variable == null)
			return null;
		if (!lineage_traces.containsKey(variable.getName()))
			return new LineageItem(variable);
		return lineage_traces.get(variable.getName());
	}
	
	private static void writeInstruction(VariableCPInstruction inst, ExecutionContext ec) {
		LineageItem li = lineage_traces.get(inst.getInput1().getName());
		String desc = Explain.explain(li);
		
		String fname = ec.getScalarInput(inst.getInput2().getName(), Types.ValueType.STRING, inst.getInput2().isLiteral()).getStringValue();
		fname += ".lineage";
		
		try {
			HDFSTool.writeStringToHDFS(desc, fname);
			FileSystem fs = IOUtilFunctions.getFileSystem(fname);
			if (fs instanceof LocalFileSystem) {
				Path path = new Path(fname);
				IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
			}
			
		} catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
	}
	
	private static void removeInstruction(VariableCPInstruction inst) {
		for (CPOperand input : inst.getInputs())
			removeLineageItem(input.getName());
	}
	
	private static void moveInstruction(VariableCPInstruction inst) {
		if (inst.getInput2().getName().equals("__pred")) {
			removeLineageItem(inst.getInput1().getName());
		} else {
			ArrayList<LineageItem> lineages = new ArrayList<>();
			if (lineage_traces.containsKey(inst.getInput1().getName()))
				lineages.add(lineage_traces.get(inst.getInput1().getName()));
			else {
				lineages.add(getOrCreate(inst.getInput1()));
				if (inst.getInput3() != null)
					lineages.add(getOrCreate(inst.getInput3()));
			}
			addLineageItem(new LineageItem(inst.getInput2(), lineages, inst.getOpcode()));
		}
	}
	
	private static void copyInstruction(VariableCPInstruction inst) {
		ArrayList<LineageItem> lineages = new ArrayList<>();
		if (lineage_traces.containsKey(inst.getInput1().getName())) {
			lineages.add(lineage_traces.get(inst.getInput1().getName()));
		} else {
			lineages.add(getOrCreate(inst.getInput1()));
		}
		addLineageItem(new LineageItem(inst.getInput2(), lineages, inst.getOpcode()));
	}
	
	private static void createVariableInstruction(VariableCPInstruction inst) {
		ArrayList<LineageItem> lineages = new ArrayList<>();
		lineages.add(lineage_traces.getOrDefault(inst.getInput2(),
				new LineageItem(inst.getInput2())));
		lineages.add(lineage_traces.getOrDefault(inst.getInput3(),
				new LineageItem(inst.getInput3())));
		addLineageItem(new LineageItem(inst.getInput1(), lineages, inst.getOpcode()));
	}
	
	public static LineageItem get(CPOperand variable) {
		if (variable == null)
			return null;
		return lineage_traces.get(variable.getName());
	}
	
	private static void removeInputLinks(LineageItem li) {
		if (li.getOutputs().isEmpty()) {
			List<LineageItem> inputs = li.getInputs();
			li.removeAllInputs();
			for (LineageItem input : inputs) {
				input.getOutputs().remove(li);
				removeInputLinks(input);
			}
		}
	}
}

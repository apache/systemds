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
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class Lineage {
	
	private static Map<String, LineageItem> lineage_traces = new HashMap<>();
	
	private Lineage() {
	}
	
	public static void trace(Instruction inst, ExecutionContext ec) {
		if (!(inst instanceof LineageTraceable))
			throw new DMLRuntimeException("Unknown Instruction (" + inst.getOpcode() + ") traced.");
		
		LineageItem li = ((LineageTraceable) inst).getLineageItem();
		
		if (inst instanceof VariableCPInstruction) {
			VariableCPInstruction vcp_inst = ((VariableCPInstruction) inst);
			
			switch (vcp_inst.getVariableOpcode()) {
				case AssignVariable:
				case CopyVariable: {
					processCopyLI(li);
					break;
				}
				case Read:
				case CreateVariable: {
					addLineageItem(li);
					break;
				}
				case RemoveVariable: {
					for (CPOperand input : vcp_inst.getInputs())
						removeLineageItem(input.getName());
					break;
				}
				case Write: {
					processWriteLI(vcp_inst, ec);
					break;
				}
				case MoveVariable: {
					processMoveLI(li);
					break;
				}
				default:
					throw new DMLRuntimeException("Unknown VariableCPInstruction (" + inst.getOpcode() + ") traced.");
			}
		} else
			addLineageItem(li);
	}
	
	private static void processCopyLI(LineageItem li) {
		if (li.getInputs().size() != 1)
			throw new DMLRuntimeException("AssignVariable and CopyVaribale must have one input lineage item!");
		
		if (lineage_traces.get(li.getName()) != null) {
			removeInputLinks(lineage_traces.get(li.getName()));
			lineage_traces.remove(li.getName());
		}
		lineage_traces.put(li.getName(), li.getInputs().get(0));
	}
	
	public static void removeLineageItem(String key) {
		if (!lineage_traces.containsKey(key))
			return;
		
		removeInputLinks(lineage_traces.get(key));
		lineage_traces.remove(key);
	}
	
	public static void addLineageItem(LineageItem li) {
		if (lineage_traces.get(li.getName()) != null) {
			removeInputLinks(lineage_traces.get(li.getName()));
			lineage_traces.remove(li.getName());
		}
		lineage_traces.put(li.getName(), li);
	}
	
	private static void processWriteLI(VariableCPInstruction inst, ExecutionContext ec) {
		String desc = Explain.explain(get(inst.getInput1()));
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
	
	private static void processMoveLI(LineageItem li) {
		if (li.getName().equals("__pred"))
			removeLineageItem(li.getInputs().get(0).getName());
		else
			addLineageItem(li);
	}
	
	public static LineageItem getOrCreate(CPOperand variable) {
		if (variable == null)
			return null;
		if (!lineage_traces.containsKey(variable.getName()))
			return new LineageItem(variable.getName(), variable.getLineageLiteral());
		return lineage_traces.get(variable.getName());
	}
	
	public static LineageItem get(CPOperand variable) {
		if (variable == null)
			return null;
		return lineage_traces.get(variable.getName());
	}
	
	public static boolean contains(CPOperand variable) {
		return lineage_traces.containsKey(variable.getName());
	}
	
	private static void removeInputLinks(LineageItem li) {
		if (li.getOutputs().isEmpty()) {
			List<LineageItem> inputs = li.getInputs();
			li.removeAllInputs();
			if (inputs != null)
				for (LineageItem input : inputs) {
					input.getOutputs().remove(li);
					removeInputLinks(input);
				}
		}
	}
}

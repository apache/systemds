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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.EvalNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.spark.WriteSPInstruction;
import org.apache.sysds.runtime.lineage.LineageItem.LineageItemType;
import org.apache.sysds.utils.Explain;

import java.util.HashMap;
import java.util.Map;

public class LineageMap {
	
	private final Map<String, LineageItem> _traces;
	private final Map<String, LineageItem> _literals;
	
	public LineageMap() {
		_traces = new HashMap<>();
		_literals = new HashMap<>();
	}
	
	public LineageMap(LineageMap that) {
		this();
		_traces.putAll(that._traces);
		_literals.putAll(that._literals);
	}
	
	public void trace(Instruction inst, ExecutionContext ec) {
		if( inst instanceof FunctionCallCPInstruction || inst instanceof EvalNaryCPInstruction)
			return; // no need for lineage tracing
		if (!(inst instanceof LineageTraceable))
			throw new DMLRuntimeException("Unknown Instruction (" + inst.getOpcode() + ") traced.");
		LineageTraceable linst = (LineageTraceable) inst;
		if( linst.hasSingleLineage() ) {
			trace(inst, ec, linst.getLineageItem(ec));
		}
		else {
			Pair<String, LineageItem>[] items = linst.getLineageItems(ec);
			if (items == null || items.length < 1)
				trace(inst, ec, null);
			else {
				for (Pair<String, LineageItem> li : items)
					trace(inst, ec, cleanupInputLiterals(li, ec));
			}
		}
	}
	
	public void processDedupItem(LineageMap lm, Long path, LineageItem[] liinputs, String name) {
		String delim = LineageDedupUtils.DEDUP_DELIM;
		for (Map.Entry<String, LineageItem> entry : lm._traces.entrySet()) {
			// Encode everything needed by the recomputation logic in the
			// opcode to map this lineage item to the right patch.
			String opcode = LineageItem.dedupItemOpcode + delim + entry.getKey()
				+ delim + name + delim + path.toString();
			LineageItem li = new LineageItem(opcode, liinputs);
			addLineageItem(Pair.of(entry.getKey(), li));
		}
	}
	
	public LineageItem getOrCreate(CPOperand variable) {
		if (variable == null)
			return null;
		String varname = variable.getName();
		//handle literals (never in traces)
		if (variable.isLiteral()) {
			LineageItem ret = _literals.get(varname);
			if (ret == null)
				_literals.put(varname, ret = new LineageItem(variable.getLineageLiteral()));
			return ret;
		}
		//handle variables
		LineageItem ret = _traces.get(variable.getName());
		return (ret != null) ? ret :
			new LineageItem(variable.getLineageLiteral());
	}
	
	public LineageItem get(String varName) {
		return _traces.get(varName);
	}
	
	public LineageItem set(String varName, LineageItem li) {
		return _traces.put(varName, li);
	}
	
	public LineageItem setLiteral(String varName, LineageItem li) {
		return _literals.put(varName, li);
	}
	
	public LineageItem get(CPOperand variable) {
		if (variable == null)
			return null;
		return _traces.get(variable.getName());
	}
	
	public boolean contains(CPOperand variable) {
		return _traces.containsKey(variable.getName());
	}
	
	public boolean containsKey(String key) {
		return _traces.containsKey(key);
	}
	
	public void resetLineageMaps() {
		_traces.clear();
		_literals.clear();
	}
	
	public Map<String, LineageItem> getTraces() {
		return _traces;
	}
	
	public Map<String, LineageItem> getLiterals() {
		return _literals;
	}
	
	private void trace(Instruction inst, ExecutionContext ec, Pair<String, LineageItem> li) {
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
					if (li != null)
						addLineageItem(li);
					break;
				}
				case RemoveVariable: {
					for (CPOperand input : vcp_inst.getInputs())
						removeLineageItem(input.getName());
					break;
				}
				case Write: {
					processWriteLI(vcp_inst.getInput1(), vcp_inst.getInput2(), ec);
					break;
				}
				case MoveVariable: {
					moveLineageItem(vcp_inst.getInput1().getName(), vcp_inst.getInput2().getName());
					break;
				}
				case CastAsBooleanVariable:
				case CastAsDoubleVariable:
				case CastAsIntegerVariable:
				case CastAsScalarVariable:
				case CastAsMatrixVariable:
				case CastAsFrameVariable: {
					addLineageItem(li);
					break;
				}
				default:
					throw new DMLRuntimeException("Unknown VariableCPInstruction (" + inst.getOpcode() + ") traced.");
			}
		}
		else if (inst instanceof WriteSPInstruction){
			processWriteLI(((WriteSPInstruction) inst).getInput1(), ((WriteSPInstruction) inst).getInput2(), ec);
		}
		else
			addLineageItem(li);
		
	}
	
	private Pair<String, LineageItem> cleanupInputLiterals(Pair<String, LineageItem> li, ExecutionContext ec) {
		LineageItem item = li.getValue();
		if( item.getInputs() == null )
			return li;
		// fix literals referring to variables (e.g., for/parfor loop variable)
		for(int i=0; i<item.getInputs().length; i++) {
			LineageItem tmp = item.getInputs()[i];
			if( tmp.getType() != LineageItemType.Literal)
				continue;
			//check if CPOperand is not a literal, w/o parsing
			if( tmp.getData().endsWith("false") ) {
				CPOperand cp = new CPOperand(tmp.getData());
				if( cp.getDataType().isScalar() ) {
					cp.setLiteral(ec.getScalarInput(cp));
					item.getInputs()[i] = getOrCreate(cp);
				}
			}
		}
		return li;
	}
	
	private void processCopyLI(Pair<String, LineageItem> li) {
		if (li.getValue().getInputs().length != 1)
			throw new DMLRuntimeException("AssignVariable and CopyVariable must have one input lineage item!");
		//add item or overwrite existing item
		_traces.put(li.getKey(), li.getValue().getInputs()[0]);
	}
	
	private void moveLineageItem(String keyFrom, String keyTo) {
		LineageItem input = removeLineageItem(keyFrom);
		if (!keyTo.equals("__pred"))
			_traces.put(keyTo, input);
	}
	
	private LineageItem removeLineageItem(String key) {
		//remove item if present
		return _traces.remove(key);
	}
	
	private void addLineageItem(Pair<String, LineageItem> li) {
		//add item or overwrite existing item
		_traces.put(li.getKey(), li.getValue());
	}
	
	private void processWriteLI(CPOperand input1, CPOperand input2, ExecutionContext ec) {
		LineageItem li = get(input1);
		String fName = ec.getScalarInput(input2.getName(), Types.ValueType.STRING, input2.isLiteral()).getStringValue();
		
		if (DMLScript.LINEAGE_DEDUP) {
			// gracefully serialize the dedup maps without decompressing
			LineageItemUtils.writeTraceToHDFS(LineageDedupUtils.mergeExplainDedupBlocks(ec), fName + ".lineage.dedup");
		}
		LineageItemUtils.writeTraceToHDFS(Explain.explain(li), fName + ".lineage");
	}
}

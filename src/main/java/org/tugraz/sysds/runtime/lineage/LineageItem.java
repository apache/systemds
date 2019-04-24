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

import java.util.ArrayList;
import java.util.List;

import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;

public class LineageItem {
	private static IDSequence _idSeq = new IDSequence();
	
	private final long _id;
	private final String _opcode;
	private final String _name;
	private final String _representation;
	private List<LineageItem> _inputs;
	private List<LineageItem> _outputs;
	private boolean _visited = false;
	
	public enum LineageItemType {Literal, Creation, Instruction}
	
	public LineageItem(long id, LineageItem li) {
		_id = id;
		_opcode = li._opcode;
		_name = li._name;
		_representation = li._representation;
		_inputs = li._inputs;
		_outputs = li._outputs;
	}
	
	public LineageItem(long id, String name, String representation) {
		this(id, name, representation, null, "");
	}
	
	public LineageItem(long id, String name, List<LineageItem> inputs, String opcode) {
		this(id, name, "", inputs, opcode);
	}
	
	public LineageItem(String name, String representation) {
		this(_idSeq.getNextID(), name, representation, null, "");
	}
	
	public LineageItem(String name, String representation, String opcode) {
		this(_idSeq.getNextID(), name, representation, null, opcode);
	}
	
	public LineageItem(String name, String representation, List<LineageItem> inputs, String opcode) {
		this(_idSeq.getNextID(), name, representation, inputs, opcode);
	}
	
	public LineageItem(String name, List<LineageItem> inputs, String opcode) {
		this(_idSeq.getNextID(), name, "", inputs, opcode);
	}
	
	public LineageItem(long id, String name, String representation, List<LineageItem> inputs, String opcode) {
		_id = id;
		_opcode = opcode;
		_name = name;
		_representation = representation;
		
		if (inputs != null) {
			_inputs = new ArrayList<>(inputs);
			for (LineageItem li : _inputs)
				li._outputs.add(this);
		} else
			_inputs = null;
		_outputs = new ArrayList<>();
	}
	
	public LineageItem(String name) {
		this(_idSeq.getNextID(), name);
	}
	
	public LineageItem(long id, String name) {
		_id = id;
		_opcode = "";
		_name = name;
		_representation = name;
		_inputs = null;
		_outputs = new ArrayList<>();
	}
	
	public List<LineageItem> getInputs() {
		return _inputs;
	}
	
	public void removeAllInputs() {
		_inputs = new ArrayList<>();
	}
	
	public List<LineageItem> getOutputs() {
		return _outputs;
	}
	
	public String getName() {
		return _name;
	}
	
	public String getRepresentation() {
		return _representation;
	}
	
	public boolean isVisited() {
		return _visited;
	}
	
	public void setVisited() {
		setVisited(true);
	}
	
	public void setVisited(boolean flag) {
		_visited = flag;
	}
	
	public long getId() {
		return _id;
	}
	
	public String getOpcode() {
		return _opcode;
	}
	
	@Override
	public String toString() {
		return LineageItemUtils.explainSingleLineageItem(this);
	}
	
	public boolean isLeaf() {
		if (_inputs == null)
			return true;
		return _inputs.isEmpty();
	}
	
	public boolean isInstruction() {
		return !_opcode.isEmpty();
	}
	
	public LineageItem resetVisitStatus() {
		if (!isVisited())
			return this;
		if (_inputs != null && !_inputs.isEmpty())
			for (LineageItem li : getInputs())
				li.resetVisitStatus();
		setVisited(false);
		return this;
	}
	
	public static void resetVisitStatus(List<LineageItem> lis) {
		if (lis != null)
			for (LineageItem liRoot : lis)
				liRoot.resetVisitStatus();
	}
	
	public static void resetIDSequence() {
		_idSeq.reset(-1);
	}
}

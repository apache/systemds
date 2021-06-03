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

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.util.UtilFunctions;

public class LineageItem {
	private static IDSequence _idSeq = new IDSequence();
	
	private final long _id;
	private final String _opcode;
	private final String _data;
	private LineageItem[] _inputs;
	private int _hash = 0;
	private LineageItem _dedupPatch;
	private long _distLeaf2Node;
	private final BooleanArray32 _specialValueBits;  // TODO: Move this to a new subclass
	// init visited to true to ensure visited items are
	// not hidden when used as inputs to new items
	private boolean _visited = true;
	
	public enum LineageItemType {Literal, Creation, Instruction, Dedup}
	public static final String dedupItemOpcode = "dedup";
	
	public LineageItem() {
		this("");
	}
	
	public LineageItem(String data) {
		this(_idSeq.getNextID(), data);
	}

	public LineageItem(long id, String data) {
		this(id, data, "", null, 0);
	}
	
	public LineageItem(String data, String opcode) {
		this(_idSeq.getNextID(), data, opcode, null, 0);
	}
	
	public LineageItem(String opcode, LineageItem[] inputs) { 
		this(_idSeq.getNextID(), "", opcode, inputs, 0);
	}

	public LineageItem(String data, String opcode, LineageItem[] inputs) {
		this(_idSeq.getNextID(), data, opcode, inputs, 0);
	}

	public LineageItem(String opcode, LineageItem dedupPatch, LineageItem[] inputs) { 
		this(_idSeq.getNextID(), "", opcode, inputs, 0);
		// maintain a pointer to the dedup patch
		_dedupPatch = dedupPatch;
		_hash = _dedupPatch._hash;
	}

	public LineageItem(String opcode, LineageItem dedupPatch, int dpatchHash, LineageItem[] inputs) { 
		this(_idSeq.getNextID(), "", opcode, inputs, 0);
		// maintain a pointer to the dedup patch
		_dedupPatch = dedupPatch;
		_hash = dpatchHash;
	}
	
	public LineageItem(LineageItem li) {
		this(_idSeq.getNextID(), li);
	}
	
	public LineageItem(long id, LineageItem li) {
		this(id, li._data, li._opcode, li._inputs, 0);
	}

	public LineageItem(long id, String data, String opcode) {
		this(id, data, opcode, null, 0);
	}
	
	public LineageItem(long id, String data, String opcode, LineageItem[] inputs, int specialValueBits) {
		_id = id;
		_opcode = opcode;
		_data = data;
		_inputs = inputs;
		// materialize hash on construction 
		// (constant time operation if input hashes constructed)
		_hash = hashCode();
		// store the distance of this node from the leaves. (O(#inputs)) operation
		_distLeaf2Node = distLeaf2Node();
		_specialValueBits = new BooleanArray32(specialValueBits);
	}
	
	public LineageItem[] getInputs() {
		return _inputs;
	}
	
	public void resetInputs() {
		_inputs = null;
		//_hash = 0;
		// Keep the hash for equality check
	}
	
	public void setInput(int i, LineageItem item) {
		_inputs[i] = item;
		_hash = 0; //reset hash
	}
	
	public String getData() {
		return _data;
	}
	
	public void fixHash() {
		_hash = 0;
		_hash = hashCode();
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
	
	public void setSpecialValueBit(int pos, boolean flag) {
		_specialValueBits.set(pos, flag);
	}
	
	public void setSpecialValueBits(int value) {
		_specialValueBits.setValue(value);
	}

	private long distLeaf2Node() {
		// Derive height only if the corresponding reuse
		// policy is selected, otherwise set -1.
		if (LineageCacheConfig.ReuseCacheType.isNone()
			|| !LineageCacheConfig.isDagHeightBased())
			return -1;

		if (_inputs != null && _inputs.length > 0) {
			// find the input with highest height
			long maxDistance = _inputs[0].getDistLeaf2Node();
			for (int i=1; i<_inputs.length; i++)
				if (_inputs[i].getDistLeaf2Node() > maxDistance)
					maxDistance = _inputs[i].getDistLeaf2Node();
			return maxDistance + 1;
		}
		else
			return 1;  //leaf node
	}
	
	public long getId() {
		return _id;
	}
	
	public String getOpcode() {
		return _opcode;
	}
	
	public boolean getSpecialValueBit(int pos) {
		return _specialValueBits.get(pos);
	}

	public int getSpecialValueBits() {
		return _specialValueBits.getValue();
	}

	public boolean isPlaceholder() {
		return _opcode.startsWith(LineageItemUtils.LPLACEHOLDER);
	}
	
	public void setDistLeaf2Node(long d) {
		_distLeaf2Node = d;
	}
	
	public long getDistLeaf2Node() {
		return _distLeaf2Node;
	}
	
	public LineageItem getDedupPatch() {
		return _dedupPatch;
	}
	
	public LineageItemType getType() {
		if (_opcode.startsWith(dedupItemOpcode))
			return LineageItemType.Dedup;
		if (isLeaf() && isInstruction())
			return LineageItemType.Creation;
		else if (isLeaf() && !isInstruction())
			return LineageItemType.Literal;
		else if (!isLeaf() && isInstruction())
			return LineageItemType.Instruction;
		else
			throw new DMLRuntimeException("An inner node could not be a literal!");
	}
	
	@Override
	public String toString() {
		return LineageItemUtils.explainSingleLineageItem(this);
	}
	
	@Override
	public boolean equals(Object o) {
		if (!(o instanceof LineageItem))
			return false;
		
		resetVisitStatusNR();
		//boolean ret = equalsLINR((LineageItem) o);
		boolean ret = equalsLINR_dedup((LineageItem) o);
		resetVisitStatusNR();
		return ret;
	}
	
	@SuppressWarnings("unused")
	private boolean equalsLI(LineageItem that) {
		if (isVisited() || this == that)
			return true;
		
		boolean ret = _opcode.equals(that._opcode);
		ret &= _data.equals(that._data);
		ret &= (hashCode() == that.hashCode());
		if( ret && _inputs != null && _inputs.length == that._inputs.length )
			for (int i = 0; i < _inputs.length; i++)
				ret &= _inputs[i].equalsLI(that._inputs[i]);
		
		setVisited();
		return ret;
	}
	
	private boolean equalsLINR(LineageItem that) {
		Stack<LineageItem> s1 = new Stack<>();
		Stack<LineageItem> s2 = new Stack<>();
		s1.push(this);
		s2.push(that);
		boolean ret = false;
		while (!s1.empty() && !s2.empty()) {
			LineageItem li1 = s1.pop();
			LineageItem li2 = s2.pop();
			if (li1.isVisited() || li1 == li2)
				return true;

			ret = li1._opcode.equals(li2._opcode);
			ret &= li1._data.equals(li2._data);
			ret &= (li1.hashCode() == li2.hashCode());
			if (!ret) break;
			if (ret && li1._inputs != null && li1._inputs.length == li2._inputs.length)
				for (int i=0; i<li1._inputs.length; i++) {
					s1.push(li1.getInputs()[i]);
					s2.push(li2.getInputs()[i]);
				}
			li1.setVisited();
		}
		
		return ret;
	}

	// Deduplication aware equality check
	private boolean equalsLINR_dedup(LineageItem that) {
		Stack<LineageItem> s1 = new Stack<>();
		Stack<LineageItem> s2 = new Stack<>();
		s1.push(this);
		s2.push(that);
		//boolean ret = false;
		boolean ret = true;
		while (!s1.empty() && !s2.empty()) {
			LineageItem li1 = s1.pop();
			LineageItem li2 = s2.pop();

			if (li1.isVisited() || li1 == li2)
				// skip this sub-DAG.
				continue;

			if (!li1.isDedup() && !li2.isDedup()) {
				// Opcodes don't match if either entry is dedup
				ret = li1._opcode.equals(li2._opcode);
				ret &= li1._data.equals(li2._data);
			}
			ret &= (li1.hashCode() == li2.hashCode());
			if (!ret) break;

			if (ret && li1._inputs != null && li1._inputs.length == li2._inputs.length || li1.isDedup() || li2.isDedup())
				for (int i=0; i<li1._inputs.length; i++) {
					// Find the i'th inputs. If the input is a non-leaf placeholder, read the inputs to it,
					// else read the normal input or the leaf placeholder.
					LineageItem in1 = li1._inputs[i].isPlaceholder() ? 
							li1._inputs[i]._inputs != null ? li1._inputs[i]._inputs[0]:li1._inputs[i] : li1._inputs[i];
					LineageItem in2 = li2._inputs[i].isPlaceholder() ? 
							li2._inputs[i]._inputs != null ? li2._inputs[i]._inputs[0]:li2._inputs[i] : li2._inputs[i];

					// If either input is a dedup node, match the corresponding dedup patch DAG to the 
					// sub-dag of the non-dedup DAG. If matched, push the inputs into the stacks in a
					// order-preserving way.
					if (in1.isDedup() && !in2.isDedup()) {
						Map<Integer, LineageItem> phMap = new HashMap<>();
						in1._dedupPatch.resetVisitStatusNR();
						ret = equalsDedupPatch(in1._dedupPatch, in2, phMap);
						in1.setVisited();
						if (!ret) {
							li1.setVisited();
							return false;
						}
						for (Map.Entry<Integer, LineageItem> ph : phMap.entrySet()) {
							s1.push(in1._inputs[ph.getKey()]);
							s2.push(ph.getValue());
						}
					}
					else if (in2.isDedup() && !in1.isDedup()) {
						Map<Integer, LineageItem> phMap = new HashMap<>();
						ret = equalsDedupPatch(in1, in2._dedupPatch, phMap);
						if (!ret) {
							li1.setVisited();
							return false;
						}
						for (Map.Entry<Integer, LineageItem> ph : phMap.entrySet()) {
							s1.push(ph.getValue());
							s1.push(in2._inputs[ph.getKey()]);
						}
					}
					// If both inputs are dedup nodes, compare the corresponding patches
					// and push all the inputs into the stacks.
					else if (in1.isDedup() && in2.isDedup()) {
						in1._dedupPatch.resetVisitStatusNR();
						in2._dedupPatch.resetVisitStatusNR();
						ret = in1._dedupPatch.equalsLINR(in2._dedupPatch);
						in1.setVisited();
						if (!ret) {
							li1.setVisited();
							return false;
						}
						if (in1._inputs.length == in2._inputs.length)
							// FIXME: Two dedup nodes can have matching patches but different #inputs
							for (int j=0; j<in1._inputs.length; j++) {
								s1.push(in1.getInputs()[j]);
								s2.push(in2.getInputs()[j]);
							}
						else {
							li1.setVisited();
							return false;
						}
					}
					else {
						s1.push(in1);
						s2.push(in2);
					}
				}
			li1.setVisited();
		}
		return ret;
	}
	
	// Compare a dedup patch with a sub-DAG, and map the inputs of the sub-dag
	// to the placeholder inputs of the dedup patch
	private static boolean equalsDedupPatch(LineageItem dli1, LineageItem dli2, Map<Integer, LineageItem> phMap) {
		Stack<LineageItem> s1 = new Stack<>();
		Stack<LineageItem> s2 = new Stack<>();
		s1.push(dli1);
		s2.push(dli2);
		boolean ret = true;
		while (!s1.empty() && !s2.empty()) {
			LineageItem li1 = s1.pop();
			LineageItem li2 = s2.pop();
			if (li1.isVisited() || li1 == li2)
				continue; //FIXME: fill phMap
			ret = li1._opcode.equals(li2._opcode);
			ret &= li1._data.equals(li2._data);
			//ret &= (li1.hashCode() == li2.hashCode());
			// Do not match the hash codes, as the hash of a dedup patch node doesn't represent the whole dag.
			if (!ret) break;
			if (ret && li1._inputs != null && li1._inputs.length == li2._inputs.length)
				for (int i=0; i<li1._inputs.length; i++) {
					LineageItem in1 = li1.getInputs()[i];
					LineageItem in2 = li2.getInputs()[i];
					in1 = in1.isPlaceholder() ? in1._inputs != null ? in1.getInputs()[0] : in1 : in1;
					in2 = in2.isPlaceholder() ? in2._inputs != null ? in2.getInputs()[0] : in2 : in2;
					if (in1.isPlaceholder() && in1._inputs == null && !in2.isPlaceholder()) {
						int phId = Integer.parseInt(in1.getOpcode().substring(3));
						phMap.put(phId, in2);
						continue;
					}
					if (in2.isPlaceholder() && in2._inputs == null && !in1.isPlaceholder()) {
						int phId = Integer.parseInt(in2.getOpcode().substring(3));
						phMap.put(phId, in1);
						continue;
					}
					if (in1.isPlaceholder() && in2.isPlaceholder())
						continue;

					s1.push(in1);
					s2.push(in2);
				}
			li1.setVisited();
		}
		return ret;
	}
	
	@Override
	public int hashCode() {
		if (_hash == 0) {
			if (isPlaceholder() && _inputs != null)
				return _inputs[0].hashCode();

			//compute hash over opcode and all inputs
			int h = UtilFunctions.intHashCode(
				_opcode.hashCode(), _data.hashCode());
			if (_inputs != null)
				for (LineageItem li : _inputs)
					h = UtilFunctions.intHashCodeRobust(li.hashCode(), h);
			_hash = h;
		}
		return _hash;
	}

	public LineageItem deepCopy() { //bottom-up
		if (isLeaf())
			return new LineageItem(this);
		
		LineageItem[] copyInputs = new LineageItem[getInputs().length];
		for (int i=0; i<_inputs.length; i++) 
			copyInputs[i] = _inputs[i].deepCopy();
		return new LineageItem(_opcode, copyInputs);
	}
	
	public boolean isLeaf() {
		return _inputs == null || _inputs.length == 0;
	}
	
	public boolean isInstruction() {
		return !_opcode.isEmpty();
	}
	
	public boolean isDedup() {
		return _opcode.startsWith(dedupItemOpcode);
	}
	
	/**
	 * Non-recursive equivalent of {@link #resetVisitStatus()} 
	 * for robustness with regard to stack overflow errors.
	 */
	public void resetVisitStatusNR() {
		Stack<LineageItem> q = new Stack<>();
		q.push(this);
		while( !q.empty() ) {
			LineageItem tmp = q.pop();
			if( !tmp.isVisited() )
				continue;
			if (tmp.getInputs() != null)
				for (LineageItem li : tmp.getInputs())
					q.push(li);
			tmp.setVisited(false);
		}
	}
	
	/**
	 * Non-recursive equivalent of {@link #resetVisitStatus(LineageItem[])} 
	 * for robustness with regard to stack overflow errors.
	 * 
	 * @param lis root lineage items
	 */
	public static void resetVisitStatusNR(LineageItem[] lis) {
		if (lis != null)
			for (LineageItem liRoot : lis)
				liRoot.resetVisitStatusNR();
	}
	
	@Deprecated
	public void resetVisitStatus() {
		if (!isVisited())
			return;
		if (_inputs != null)
			for (LineageItem li : getInputs())
				li.resetVisitStatus();
		setVisited(false);
	}
	
	@Deprecated
	public static void resetVisitStatus(LineageItem[] lis) {
		if (lis != null)
			for (LineageItem liRoot : lis)
				liRoot.resetVisitStatus();
	}
	
	public static void resetIDSequence() {
		_idSeq.reset(-1);
	}
}

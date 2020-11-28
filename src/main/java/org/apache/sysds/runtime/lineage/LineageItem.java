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

import java.util.Stack;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.util.UtilFunctions;

public class LineageItem {
	private static IDSequence _idSeq = new IDSequence();
	
	private final long _id;
	private final String _opcode;
	private final String _data;
	private final LineageItem[] _inputs;
	private int _hash = 0;
	private long _distLeaf2Node;
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
		this(id, data, "", null);
	}
	
	public LineageItem(String data, String opcode) {
		this(_idSeq.getNextID(), data, opcode, null);
	}
	
	public LineageItem(String opcode, LineageItem[] inputs) { 
		this(_idSeq.getNextID(), "", opcode, inputs);
	}

	public LineageItem(String data, String opcode, LineageItem[] inputs) {
		this(_idSeq.getNextID(), data, opcode, inputs);
	}
	
	public LineageItem(LineageItem li) {
		this(_idSeq.getNextID(), li);
	}
	
	public LineageItem(long id, LineageItem li) {
		this(id, li._data, li._opcode, li._inputs);
	}

	public LineageItem(long id, String data, String opcode) {
		this(id, data, opcode, null);
	}
	
	public LineageItem(long id, String data, String opcode, LineageItem[] inputs) {
		_id = id;
		_opcode = opcode;
		_data = data;
		_inputs = inputs;
		// materialize hash on construction 
		// (constant time operation if input hashes constructed)
		_hash = hashCode();
		// store the distance of this node from the leaves. (O(#inputs)) operation
		_distLeaf2Node = distLeaf2Node();
	}
	
	public LineageItem[] getInputs() {
		return _inputs;
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
	
	public void setDistLeaf2Node(long d) {
		_distLeaf2Node = d;
	}
	
	public long getDistLeaf2Node() {
		return _distLeaf2Node;
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
		boolean ret = equalsLINR((LineageItem) o);
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
	
	@Override
	public int hashCode() {
		if (_hash == 0) {
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

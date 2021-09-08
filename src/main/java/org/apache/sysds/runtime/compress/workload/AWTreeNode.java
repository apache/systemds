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

package org.apache.sysds.runtime.compress.workload;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * A workload tree is a compact representation of the operations on a matrix and derived intermediates, including the
 * basic control structure and inlined functions as well as links to categories.
 * 
 * The intension is to provide the ability to look at a variable and the methods performed on this variable, pruning
 * away the rest of the DAG.
 * 
 */
public abstract class AWTreeNode {

	protected static final Log LOG = LogFactory.getLog(AWTreeNode.class.getName());

	public enum WTNodeType {
		ROOT, FCALL, IF, WHILE, FOR, PARFOR, BASIC_BLOCK;

		public boolean isLoop() {
			return this == WHILE || this == FOR || this == PARFOR;
		}
	}

	private final WTNodeType _type;
	protected final List<WTreeNode> _children = new ArrayList<>();
	protected final List<Op> _ops = new ArrayList<>();

	public AWTreeNode(WTNodeType type) {
		_type = type;
	}

	public WTNodeType getType() {
		return _type;
	}

	public List<WTreeNode> getChildNodes() {
		return _children;
	}

	public void addChild(WTreeNode node) {
		_children.add(node);
	}

	public List<Op> getOps() {
		return _ops;
	}

	public void addOp(Op op) {
		_ops.add(op);
	}

	public boolean isEmpty() {
		return _ops.isEmpty() && _children.isEmpty();
	}

	protected String explain(int level) {
		StringBuilder sb = new StringBuilder();
		// append indentation
		for(int i = 0; i < level; i++)
			sb.append("--");
		// append node summary
		sb.append(_type.name());
		sb.append("\n");
		for(Op hop : _ops) {
			for(int i = 0; i < level + 1; i++)
				sb.append("--");
			sb.append(hop.toString());
			sb.append("\n");
		}
		// append child nodes
		if(!_children.isEmpty())
			for(AWTreeNode n : _children)
				sb.append(n.explain(level + 1));
		return sb.toString();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\n---------------------------Workload Tree:---------------------------------------\n");
		sb.append(this.explain(0));
		sb.append("--------------------------------------------------------------------------------\n");
		return sb.toString();
	}
}

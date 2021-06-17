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

/**
 * A Node in the WTree, this is used for any nodes that are not the root.
 */
public class WTreeNode extends AWTreeNode {
	private static final long serialVersionUID = 253971354140571481L;
	
	private final List<Op> _ops = new ArrayList<>();

	public WTreeNode(WTNodeType type) {
		super(type);
	}

	public List<Op> getOps() {
		return _ops;
	}

	public void addOp(Op op) {
		_ops.add(op);
	}

	@Override
	public boolean isEmpty() {
		return _ops.isEmpty() && super.isEmpty();
	}

	@Override
	protected String explain(int level) {
		StringBuilder sb = new StringBuilder();
		sb.append(super.explain(level));
		for(Op hop : _ops) {
			for(int i = 0; i < level + 1; i++)
				sb.append("--");
			sb.append(hop.toString());
			sb.append("\n");
		}
		return sb.toString();
	}
}

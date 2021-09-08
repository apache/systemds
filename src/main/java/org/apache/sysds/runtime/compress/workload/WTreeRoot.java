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

import org.apache.sysds.hops.Hop;

/**
 * The root node of the tree, located at the top of the tree.
 * 
 * This represent a single Hop that have a result that is used on subsequent operations.
 */
public class WTreeRoot extends AWTreeNode {

	private final Hop _root;

	private boolean isDecompressing = false;

	public WTreeRoot(Hop root) {
		super(WTNodeType.ROOT);
		_root = root;
	}

	/**
	 * Get the Root hop instruction, that is producing a result used in the rest of the tree.
	 * 
	 * @return The root hop
	 */
	public Hop getRoot() {
		return _root;
	}

	public boolean isDecompressing() {
		return isDecompressing;
	}

	public void setDecompressing() {
		isDecompressing = true;
	}

	@Override
	protected String explain(int level) {
		StringBuilder sb = new StringBuilder();
	
		// append node summary
		sb.append("ROOT : " + _root.toString());
		sb.append("\n");

		// append child nodes
		if(!_children.isEmpty())
			for(AWTreeNode n : _children)
				sb.append(n.explain(level + 1));
		return sb.toString();
	}

}

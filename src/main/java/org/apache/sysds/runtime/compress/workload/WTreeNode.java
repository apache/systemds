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

import org.apache.sysds.hops.Hop;

/**
 * A workload tree is a compact representation of the operations
 * on a compressed matrix and derived intermediates, including
 * the basic control structure and inlined functions as well
 * as links to categories 
 * 
 * TODO separate classes for inner and leaf nodes?
 */
public class WTreeNode
{
	public enum WTNodeType{
		MAIN,
		FCALL,
		IF,
		WHILE,
		FOR,
		PARFOR,
		BASIC_BLOCK;
		public boolean isLoop() {
			return this == WHILE ||
				this == FOR || this == PARFOR;
		}
	}
	
	private final WTNodeType _type;
	private final List<WTreeNode> _childs = new ArrayList<>();
	private final List<Hop> _cops = new ArrayList<>();
	private int _beginLine = -1;
	private int _endLine = -1;
	
	public WTreeNode(WTNodeType type) {
		_type = type;
	}
	
	public WTNodeType getType() {
		return _type;
	}
	
	public List<WTreeNode> getChildNodes() {
		return _childs;
	}
	
	public void addChild(WTreeNode node) {
		_childs.add(node);
	}
	
	public List<Hop> getCompressedOps() {
		return _cops;
	}
	
	public void addCompressedOp(Hop hop) {
		_cops.add(hop);
	}
	
	public void setLineNumbers(int begin, int end) {
		_beginLine = begin;
		_endLine = end;
	}
	
	public String explain(int level) {
		StringBuilder sb = new StringBuilder();
		//append indentation
		for( int i=0; i<level; i++ )
			sb.append("--");
		//append node summary
		sb.append(_type.name());
		if( _beginLine>=0 && _endLine>=0 ) {
			sb.append(" (lines ");
			sb.append(_beginLine);
			sb.append("-");
			sb.append(_endLine);
			sb.append(")");
		}
		sb.append("\n");
		//append child nodes
		if( !_childs.isEmpty() )
			for( WTreeNode n : _childs )
				sb.append(n.explain(level+1));
		else if( !_cops.isEmpty() ) {
			for( Hop hop : _cops ) {
				for( int i=0; i<level+1; i++ )
					sb.append("--");
				sb.append(hop.toString());
				sb.append("\n");
			}
		}
		return sb.toString();
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("Workload Tree:\n");
		sb.append("--------------------------------------------------------------------------------\n");
		sb.append(this.explain(1));
		sb.append("--------------------------------------------------------------------------------\n");
		return sb.toString();
	}
}

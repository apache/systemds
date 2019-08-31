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

import org.tugraz.sysds.runtime.controlprogram.*;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;

import java.util.ArrayList;

public class LineageDedupBlock {
	private ArrayList<DistinctPaths> _blocks = new ArrayList<>();
	
	public LineageMap getMap(int block, Long path) {
		return block < _blocks.size() && _blocks.get(block).pathExists(path) ?
			_blocks.get(block).getMap(path) : null;
	}
	
	public LineageMap getActiveMap() {
		return _blocks.get(_blocks.size() - 1).getActiveMap();
	}
	
	public void traceIfProgramBlock(IfProgramBlock ipb, ExecutionContext ec) {
		_blocks.get(_blocks.size() - 1).traceIfProgramBlock(ipb, ec);
	}
	
	public void traceBasicProgramBlock(BasicProgramBlock bpb, ExecutionContext ec) {
		_blocks.get(_blocks.size() - 1).traceBasicProgramBlock(bpb, ec);
	}
	
	public void splitBlocks() {
		if (!_blocks.get(_blocks.size() - 1).empty())
			_blocks.add(new DistinctPaths());
	}
	
	public void addBlock() {
		_blocks.add(new DistinctPaths());
	}
	
	public void removeLastBlockIfEmpty() {
		if (_blocks.size() > 0 && _blocks.get(_blocks.size() - 1).empty())
			_blocks.remove(_blocks.size() - 1);
	}
}

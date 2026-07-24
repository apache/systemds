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

package org.apache.sysds.runtime.ooc.primitives;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.planning.OOCPlanner;
import org.apache.sysds.runtime.ooc.stream.StreamContext;

public abstract class OOCPrimitive {
	private final StreamContext _context;
	private final List<OOCPrimitive> _children;
	private final List<OOCPrimitive> _parents;
	private final AtomicBoolean _started;
	private final AtomicBoolean _executionStarted;
	protected OOCAccessPattern _pattern;
	protected MemoryAllowance _allowance;

	protected OOCPrimitive(StreamContext context, List<OOCPrimitive> children) {
		_context = context;
		_parents = new ArrayList<>();
		List<OOCPrimitive> uniqueChildren = new ArrayList<>(children.size());
		for(OOCPrimitive child : children) {
			if(containsIdentity(uniqueChildren, child))
				continue;
			uniqueChildren.add(child);
			child.addParent(this);
		}
		_children = List.copyOf(uniqueChildren);
		_started = new AtomicBoolean();
		_executionStarted = new AtomicBoolean();
		_pattern = OOCAccessPattern.UNSET;
	}

	public final StreamContext getContext() {
		return _context;
	}

	public final List<OOCPrimitive> getChildren() {
		return _children;
	}

	public final List<OOCPrimitive> getParents() {
		return List.copyOf(_parents);
	}

	private void addParent(OOCPrimitive parent) {
		if(!containsIdentity(_parents, parent))
			_parents.add(parent);
	}

	protected final void inferParentPatterns() {
		for(OOCPrimitive parent : _parents)
			if(parent._pattern.isUnset())
				parent.inferPatterns();
	}

	public final OOCAccessPattern getAccessPattern() {
		return _pattern;
	}

	public final boolean hasStartedExecution() {
		return _executionStarted.get();
	}

	public final void start() {
		if(_started.compareAndSet(false, true))
			OOCPlanner.compile(this);
	}

	public final void tryStartExecution() {
		if(_executionStarted.compareAndSet(false, true)) {
			_allowance = new SyncMemoryAllowance(GlobalMemoryBroker.get());
			startExecution();
		}
	}

	public final void onComplete() {
		_allowance.shutdown();
	}

	public final void inferPatterns() {
		if(!hasStartedExecution())
			inferPatternsInternal();
	}

	public final void requestPattern(OOCAccessPattern accessPattern) {
		if(!hasStartedExecution() && _pattern != accessPattern)
			requestPatternInternal(accessPattern);
	}

	private static boolean containsIdentity(List<OOCPrimitive> primitives, OOCPrimitive primitive) {
		for(OOCPrimitive current : primitives)
			if(current == primitive)
				return true;
		return false;
	}

	protected abstract void startExecution();

	protected abstract void inferPatternsInternal();

	protected abstract void requestPatternInternal(OOCAccessPattern accessPattern);
}

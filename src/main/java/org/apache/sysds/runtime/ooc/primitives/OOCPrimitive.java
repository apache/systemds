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
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.memory.GlobalMemoryBroker;
import org.apache.sysds.runtime.ooc.memory.MemoryAllowance;
import org.apache.sysds.runtime.ooc.memory.SyncMemoryAllowance;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.planning.OOCPlanner;
import org.apache.sysds.runtime.ooc.planning.OOCStoreLayout;
import org.apache.sysds.runtime.ooc.store.MaterializedStore;
import org.apache.sysds.runtime.ooc.stream.StreamContext;

public abstract class OOCPrimitive {
	private final StreamContext _context;
	private final Set<OOCPrimitive> _children;
	private final Set<OOCPrimitive> _parents;
	private final List<InputSlot> _inputs;
	private final AtomicBoolean _started;
	private final AtomicBoolean _executionStarted;
	protected OOCAccessPattern _pattern;
	protected MemoryAllowance _allowance;

	protected OOCPrimitive(StreamContext context, List<OOCPrimitive> children) {
		this(context);
		children.stream().filter(Objects::nonNull).forEach(child -> {
			_children.add(child);
			child._parents.add(this);
		});
	}

	protected OOCPrimitive(StreamContext context, OOCStreamable<?>... inputs) {
		this(context);
		for(OOCStreamable<?> input : inputs)
			_inputs.add(new InputSlot(input));
		rebuildInputChildren();
	}

	private OOCPrimitive(StreamContext context) {
		_context = context;
		_children = new HashSet<>();
		_parents = new HashSet<>();
		_inputs = new ArrayList<>();
		_started = new AtomicBoolean();
		_executionStarted = new AtomicBoolean();
		_pattern = OOCAccessPattern.UNSET;
	}

	public final StreamContext getContext() {
		return _context;
	}

	public final Set<OOCPrimitive> getChildren() {
		return _children;
	}

	public final Set<OOCPrimitive> getParents() {
		return _parents;
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

	public List<OOCMaterializedInputRequest> requiredMaterializedInputs() {
		return List.of();
	}

	public final OOCStreamable<?> getInput(int index) {
		return _inputs.get(index)._source;
	}

	public final OOCPrimitive getChildPrimitiveAt(int index) {
		return _inputs.get(index)._primitive;
	}

	public final void installMaterializedInput(int index, MaterializeOOCPrimitive boundary) {
		if(hasStartedExecution())
			throw new IllegalStateException("Cannot replace an input after primitive execution started.");
		InputSlot input = _inputs.get(index);
		input._primitive = boundary;
		rebuildInputChildren();
	}

	public final synchronized void transferInputHandle(int index) {
		InputSlot input = _inputs.get(index);
		if(!input._handleReserved)
			throw new IllegalStateException("Input " + index + " no longer owns a lazy handle.");
		input._handleReserved = false;
	}

	public final void discardInputHandle(int index) {
		OOCStreamable<?> source;
		synchronized(this) {
			InputSlot input = _inputs.get(index);
			if(!input._handleReserved)
				return;
			input._handleReserved = false;
			source = input._source;
		}
		source.discardHandle();
	}

	@SuppressWarnings("unchecked")
	protected final <T> OOCStream<T> getInputReadStream(int index) {
		transferInputHandle(index);
		return (OOCStream<T>) _inputs.get(index)._source.getReservedReadStream();
	}

	protected final OOCFuture<MaterializedStore<IndexedMatrixValue>> getMaterializedInput(int index) {
		OOCFuture<MaterializedStore<IndexedMatrixValue>> materialized = ((MaterializeOOCPrimitive) _inputs
			.get(index)._primitive).store();
		if(materialized == null)
			throw new IllegalStateException("Input " + index + " was not materialized by the planner.");
		return materialized;
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
		for(int i = 0; i < _inputs.size(); i++)
			discardInputHandle(i);
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

	private void rebuildInputChildren() {
		List<OOCPrimitive> next = new ArrayList<>();
		for(InputSlot input : _inputs)
			if(input._primitive != null)
				next.add(input._primitive);
		for(OOCPrimitive child : _children)
			if(!next.contains(child))
				child._parents.remove(this);
		for(OOCPrimitive child : next)
			child._parents.add(this);
		_children.clear();
		_children.addAll(next);
	}

	protected abstract void startExecution();

	protected abstract void inferPatternsInternal();

	protected abstract void requestPatternInternal(OOCAccessPattern accessPattern);

	private static final class InputSlot {
		private final OOCStreamable<?> _source;
		private OOCPrimitive _primitive;
		private boolean _handleReserved;

		private InputSlot(OOCStreamable<?> source) {
			_source = source;
			_primitive = source.getPrimitive();
			_handleReserved = true;
			source.reserveLazyHandle();
		}
	}

	public record OOCMaterializedInputRequest(int inputIndex, OOCStoreLayout layout, int expectedReaders) {
	}
}

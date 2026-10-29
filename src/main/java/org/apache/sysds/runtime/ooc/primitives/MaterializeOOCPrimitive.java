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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.ToIntFunction;

import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.cache.OOCCacheManager;
import org.apache.sysds.runtime.ooc.cache.OOCFuture;
import org.apache.sysds.runtime.ooc.planning.OOCAccessPattern;
import org.apache.sysds.runtime.ooc.planning.OOCStoreLayout;
import org.apache.sysds.runtime.ooc.store.MaterializedStore;
import org.apache.sysds.runtime.ooc.store.OOCStreamMaterializer;
import org.apache.sysds.runtime.ooc.stream.StreamContext;

public final class MaterializeOOCPrimitive extends OOCPrimitive {
	private final OOCStreamable<IndexedMatrixValue> _source;
	private final OOCStoreLayout _layout;
	private final OOCFuture<MaterializedStore<IndexedMatrixValue>> _store;
	private final AtomicBoolean _finished;
	private final boolean _reusable;
	private final List<Consumer<OOCStream.QueueCallback<IndexedMatrixValue>>> _liveConsumers;
	private MaterializedStore<IndexedMatrixValue> _materializedStore;
	private int _expectedReaders;
	private int _consumers;

	public MaterializeOOCPrimitive(OOCStreamable<IndexedMatrixValue> source, OOCStoreLayout layout,
		StreamContext context) {
		this(source, layout, context, false);
	}

	private MaterializeOOCPrimitive(OOCStreamable<IndexedMatrixValue> source, OOCStoreLayout layout,
		StreamContext context, boolean reusable) {
		super(context, source);
		_source = source;
		_layout = layout;
		_store = new OOCFuture<>();
		_finished = new AtomicBoolean();
		_reusable = reusable;
		_liveConsumers = new ArrayList<>();
	}

	public static MaterializeOOCPrimitive reusable(OOCStreamable<IndexedMatrixValue> source) {
		return reusable(source, OOCStoreLayout.ROW_MAJOR);
	}

	public static MaterializeOOCPrimitive reusable(OOCStreamable<IndexedMatrixValue> source, OOCStoreLayout layout) {
		return new MaterializeOOCPrimitive(source, layout, null, true);
	}

	public synchronized boolean registerRequest(int expectedReaders,
		Consumer<OOCStream.QueueCallback<IndexedMatrixValue>> liveConsumer) {
		if(expectedReaders <= 0)
			throw new IllegalArgumentException("Materialization request requires at least one reader.");
		boolean live = !hasStartedExecution();
		if(_materializedStore == null) {
			if(!_reusable)
				_expectedReaders = Math.addExact(_expectedReaders, expectedReaders);
			_consumers = Math.addExact(_consumers, 1);
		}
		else
			_materializedStore.registerConsumer(expectedReaders);
		if(live && liveConsumer != null)
			_liveConsumers.add(liveConsumer);
		return live;
	}

	public OOCFuture<MaterializedStore<IndexedMatrixValue>> store() {
		return _store;
	}

	@Override
	protected void inferPatternsInternal() {
		_pattern = OOCAccessPattern.ROW_MAJOR;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(OOCAccessPattern.ROW_MAJOR);
		inferParentPatterns();
	}

	@Override
	protected void requestPatternInternal(OOCAccessPattern accessPattern) {
		_pattern = OOCAccessPattern.ROW_MAJOR;
		for(OOCPrimitive child : getChildren())
			child.requestPattern(OOCAccessPattern.ROW_MAJOR);
	}

	@Override
	protected void startExecution() {
		try {
			OOCStream<IndexedMatrixValue> source = getInputReadStream(0);
			DataCharacteristics characteristics = _source.getDataCharacteristics();
			boolean logicalLayout = characteristics != null && characteristics.dimsKnown() &&
				characteristics.getBlocksize() > 0;
			ToIntFunction<MatrixIndexes> linearize = logicalLayout ? indexes -> _layout.linearize(indexes,
				characteristics) : null;
			MaterializedStore<IndexedMatrixValue> store;
			synchronized(this) {
				int consumers = _reusable ? 1 + _consumers : _consumers;
				store = new MaterializedStore<>(OOCCacheManager.getGlobalCache(), CachingStream._streamSeq.getNextID(),
					_reusable ? -1 : _expectedReaders, consumers, logicalLayout ? _layout : null,
					logicalLayout ? characteristics : null);
				_materializedStore = store;
			}
			AtomicInteger nextIndex = new AtomicInteger();
			ToIntFunction<MatrixIndexes> publicationIndex = linearize != null ? linearize : ignored -> nextIndex
				.getAndIncrement();
			OOCStreamMaterializer materializer = new OOCStreamMaterializer(store, publicationIndex, _allowance,
				_liveConsumers);
			materializer.completion().whenComplete((ignored, error) -> {
				if(error != null)
					fail(error);
				finish();
			});
			if(getContext() != null)
				getContext().addInStream(source);
			_store.complete(store);
			materializer.attach(source);
		}
		catch(Throwable failure) {
			_store.completeExceptionally(failure);
			fail(failure);
			finish();
		}
	}

	private void finish() {
		if(_finished.compareAndSet(false, true))
			onComplete();
	}
}

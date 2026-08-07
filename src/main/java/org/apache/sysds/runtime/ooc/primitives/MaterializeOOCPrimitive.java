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

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.ToIntFunction;

import org.apache.sysds.runtime.DMLRuntimeException;
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
	}

	public static MaterializeOOCPrimitive reusable(OOCStreamable<IndexedMatrixValue> source) {
		return new MaterializeOOCPrimitive(source, OOCStoreLayout.ROW_MAJOR, null, true);
	}

	public synchronized void registerRequest(int expectedReaders) {
		if(_reusable)
			throw new IllegalStateException("Reusable materialization registers readers dynamically.");
		if(expectedReaders <= 0)
			throw new IllegalArgumentException("Materialization request requires at least one reader.");
		if(hasStartedExecution())
			throw new IllegalStateException("Cannot register a consumer after materialization started.");
		_expectedReaders = Math.addExact(_expectedReaders, expectedReaders);
		_consumers = Math.addExact(_consumers, 1);
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
			MaterializedStore<IndexedMatrixValue> store = _reusable ? new MaterializedStore<>(
				OOCCacheManager.getGlobalCache(),
				CachingStream._streamSeq.getNextID()) : new MaterializedStore<>(OOCCacheManager.getGlobalCache(),
					CachingStream._streamSeq.getNextID(), _expectedReaders, _consumers);
			DataCharacteristics characteristics = _source.getDataCharacteristics();
			AtomicInteger nextIndex = new AtomicInteger();
			ToIntFunction<MatrixIndexes> linearize;
			if(_reusable &&
				(characteristics == null || !characteristics.dimsKnown() || characteristics.getBlocksize() <= 0))
				linearize = ignored -> nextIndex.getAndIncrement();
			else
				linearize = indexes -> _layout.linearize(indexes, characteristics);
			OOCStreamMaterializer materializer = new OOCStreamMaterializer(store, linearize, _allowance);
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

	private void fail(Throwable error) {
		if(getContext() != null)
			getContext().failAll(DMLRuntimeException.of(error));
	}

	private void finish() {
		if(_finished.compareAndSet(false, true))
			onComplete();
	}
}

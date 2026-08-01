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

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
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
	private int _expectedReaders;
	private int _consumers;

	public MaterializeOOCPrimitive(OOCStreamable<IndexedMatrixValue> source, OOCStoreLayout layout,
		StreamContext context) {
		super(context, source.getPrimitive() == null ? List.of() : List.of(source.getPrimitive()));
		_source = source;
		_layout = layout;
		_store = new OOCFuture<>();
		_finished = new AtomicBoolean();
	}

	public synchronized void registerRequest(int expectedReaders) {
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
			OOCStream<IndexedMatrixValue> source = _source.getReservedReadStream();
			MaterializedStore<IndexedMatrixValue> store = new MaterializedStore<>(OOCCacheManager.getGlobalCache(),
				CachingStream._streamSeq.getNextID(), _expectedReaders, _consumers);
			OOCStreamMaterializer materializer = new OOCStreamMaterializer(store,
				indexes -> _layout.linearize(indexes, _source.getDataCharacteristics()), _allowance);
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

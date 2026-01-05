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

package org.apache.sysds.runtime.ooc.stream;

import org.apache.sysds.runtime.instructions.ooc.SubscribableTaskQueue;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.ooc.cache.OOCIOHandler;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;

import java.util.concurrent.ConcurrentHashMap;

public class OOCSourceStream extends SubscribableTaskQueue<IndexedMatrixValue> {
	private final ConcurrentHashMap<MatrixIndexes, OOCIOHandler.SourceBlockDescriptor> _idx;

	public OOCSourceStream() {
		this._idx = new ConcurrentHashMap<>();
	}

	public void enqueue(IndexedMatrixValue value, OOCIOHandler.SourceBlockDescriptor descriptor) {
		if(descriptor == null)
			throw new IllegalArgumentException("Source descriptor must not be null");
		MatrixIndexes key = new MatrixIndexes(descriptor.indexes);
		_idx.put(key, descriptor);
		super.enqueue(value);
	}

	@Override
	public void enqueue(IndexedMatrixValue val) {
		throw new UnsupportedOperationException("Use enqueue(value, descriptor) for source streams");
	}

	public OOCIOHandler.SourceBlockDescriptor getDescriptor(MatrixIndexes indexes) {
		return _idx.get(indexes);
	}
}

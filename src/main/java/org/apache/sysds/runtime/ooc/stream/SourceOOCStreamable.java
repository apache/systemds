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

import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.ooc.CachingStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.ooc.OOCStreamable;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.function.BiFunction;
import java.util.function.Consumer;

public class SourceOOCStreamable implements OOCStreamable<IndexedMatrixValue> {
	private final CacheableData<?> _data;

	public SourceOOCStreamable(CacheableData<?> data) {
		_data = data;
	}

	@Override
	public OOCStream<IndexedMatrixValue> getReadStream() {
		return _data.getStreamHandle();
	}

	@Override
	public OOCStream<IndexedMatrixValue> getWriteStream() {
		return _data.getStreamHandle();
	}

	@Override
	public boolean hasStreamCache() {
		return false;
	}

	@Override
	public CachingStream getStreamCache() {
		return null;
	}

	@Override
	public boolean isProcessed() {
		return false;
	}

	@Override
	public DataCharacteristics getDataCharacteristics() {
		return _data.getDataCharacteristics();
	}

	@Override
	public CacheableData<?> getData() {
		return _data;
	}

	@Override
	public void setData(CacheableData<?> data) {
		throw new UnsupportedOperationException();
	}

	@Override
	public void messageUpstream(OOCStreamMessage msg) {

	}

	@Override
	public void messageDownstream(OOCStreamMessage msg) {

	}

	@Override
	public void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {

	}

	@Override
	public void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {

	}

	@Override
	public void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay) {

	}

	@Override
	public void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay) {

	}

	@Override
	public void clearUpstreamMessageRelays() {

	}

	@Override
	public void clearDownstreamMessageRelays() {

	}

	@Override
	public void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {

	}
}

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

package org.apache.sysds.runtime.instructions.ooc;

import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.ooc.stream.message.OOCStreamMessage;
import org.apache.sysds.runtime.util.IndexRange;

import java.util.function.BiFunction;
import java.util.function.Consumer;

public interface OOCStreamable<T> {
	OOCStream<T> getReadStream();

	OOCStream<T> getWriteStream();

	boolean isProcessed();

	DataCharacteristics getDataCharacteristics();

	CacheableData<?> getData();

	void setData(CacheableData<?> data);

	void messageUpstream(OOCStreamMessage msg);

	void messageDownstream(OOCStreamMessage msg);

	void setUpstreamMessageRelay(Consumer<OOCStreamMessage> relay);

	void setDownstreamMessageRelay(Consumer<OOCStreamMessage> relay);

	void addUpstreamMessageRelay(Consumer<OOCStreamMessage> relay);

	void addDownstreamMessageRelay(Consumer<OOCStreamMessage> relay);

	void clearUpstreamMessageRelays();

	void clearDownstreamMessageRelays();

	void setIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform);
}

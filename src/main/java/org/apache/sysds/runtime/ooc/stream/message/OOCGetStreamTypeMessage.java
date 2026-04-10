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

package org.apache.sysds.runtime.ooc.stream.message;

import org.apache.sysds.runtime.util.IndexRange;

import java.util.function.BiFunction;

public class OOCGetStreamTypeMessage implements OOCStreamMessage {
	public static final byte STREAM_TYPE_UNKNOWN = 0;
	public static final byte STREAM_TYPE_CACHED = 1;
	public static final byte STREAM_TYPE_IN_MEMORY = 2;

	private byte _streamType;

	public OOCGetStreamTypeMessage() {
		_streamType = 0;
	}

	@Override
	public void addIXTransform(BiFunction<Boolean, IndexRange, IndexRange> transform) {}

	public void setUnknownType() {
		_streamType = STREAM_TYPE_UNKNOWN;
	}

	public void setCachedType() {
		_streamType = STREAM_TYPE_CACHED;
	}

	public void setInMemoryType() {
		_streamType = STREAM_TYPE_IN_MEMORY;
	}

	public byte getStreamType() {
		return _streamType;
	}

	public boolean isRequestable() {
		return _streamType == STREAM_TYPE_CACHED || _streamType == STREAM_TYPE_IN_MEMORY;
	}
}

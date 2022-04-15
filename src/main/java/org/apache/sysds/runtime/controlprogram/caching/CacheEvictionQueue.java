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

package org.apache.sysds.runtime.controlprogram.caching;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class CacheEvictionQueue extends LinkedHashMap<String, ByteBuffer>
{
	/**
	 * Extended LinkedHashMap with convenience methods for adding and removing
	 * last/first entries.
	 *
	 */
	private static final long serialVersionUID = -5208333402581364859L;

	public void addLast( String fname, ByteBuffer bbuff ) {
		//put entry into eviction queue w/ 'addLast' semantics
		put(fname, bbuff);
	}

	public Map.Entry<String, ByteBuffer> removeFirst()
	{
		//move iterator to first entry
		Iterator<Map.Entry<String, ByteBuffer>> iter = entrySet().iterator();
		Map.Entry<String, ByteBuffer> entry = iter.next();

		//remove current iterator entry
		iter.remove();

		return entry;
	}

	public Map.Entry<String, ByteBuffer> removeFirstUnpinned(List<String> pinnedList) {
		//move iterator to first entry
		Iterator<Map.Entry<String, ByteBuffer>> iter = entrySet().iterator();
		var entry = iter.next();
		while (pinnedList.contains(entry.getKey()))
			entry = iter.next();

		//remove current iterator entry
		iter.remove();
		return entry;
	}
}

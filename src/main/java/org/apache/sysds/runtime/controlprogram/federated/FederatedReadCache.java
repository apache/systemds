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

package org.apache.sysds.runtime.controlprogram.federated;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

import org.apache.log4j.Logger;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;

public class FederatedReadCache {
	private static final Logger LOG = Logger.getLogger(FederatedReadCache.class);

	private Map<String, ReadCacheEntry> _rmap = new ConcurrentHashMap<>();

	/**
	 * Get the data from the ReadCacheEntry corresponding to the specified
	 * filename, if the data from this filename has already been read.
	 * Otherwise, create a new ReadCacheEntry for the filename and return null
	 * to indicate that the data is not cached yet.
	 *
	 * @param fname the filename of the read data
	 * @param putPlaceholder whether to put a placeholder if there is no mapping for the filename
	 * @return the CacheableData object if it is cached, otherwise null
	 */
	public CacheableData<?> get(String fname, boolean putPlaceholder) {
		ReadCacheEntry tmp = putPlaceholder ?
			_rmap.putIfAbsent(fname, new ReadCacheEntry()) : _rmap.get(fname);
		return (tmp != null) ? tmp.get() : null;
	}

	/**
	 * Set the data for the ReadCacheEntry with specified filename.
	 *
	 * @param fname the filename of the read data
	 * @param data the CacheableData object for setting the ReadCacheEntry
	 */
	public void setData(String fname, CacheableData<?> data) {
		LOG.trace("Setting the data for the ReadCacheEntry of file " + fname);
		ReadCacheEntry rce = _rmap.get(fname);
		if(rce == null)
			throw new DMLRuntimeException("Tried to set the data for an unregistered ReadCacheEntry.");
		rce.setValue(data);
	}

	/**
	 * Set the ReadCacheEntry of a given filename to invalid. Usually done after a
	 * failing read attempt so that the threads waiting for the data can continue.
	 *
	 * @param fname the filename of the read data
	 */
	public void setInvalid(String fname) {
		LOG.debug("Read of file " + fname + " failed. Setting the corresponding ReadCacheEntry to invalid.");
		ReadCacheEntry rce = _rmap.get(fname);
		if(rce == null)
			throw new DMLRuntimeException("Tried to set an unexisting ReadCacheEntry to invalid.");
		rce.setInvalid();
	}

	/**
	 * Class representing an entry of the federated read cache.
	 */
	public static class ReadCacheEntry {
		protected CacheableData<?> _data = null;
		private boolean _is_valid = true;

		public synchronized CacheableData<?> get() {
			try {
				//wait until other thread completes operation
				//in order to avoid redundant computation
				while(_data == null && _is_valid) {
					wait();
				}
				if(!_is_valid) { // previous thread failed when trying to read the data
					_is_valid = true;
					return null; // trying to read the data with the current thread
				}
			}
			catch( InterruptedException ex ) {
				throw new DMLRuntimeException(ex);
			}

			if(DMLScript.STATISTICS) {
				FederatedStatistics.incFedReuseReadHitCount();
				FederatedStatistics.incFedReuseReadBytesCount(_data);
			}

			//comes here if data is placed or the entry is removed by the running thread
			return _data;
		}

		public synchronized void setValue(CacheableData<?> val) {
			if(_data != null)
				throw new DMLRuntimeException("Tried to set the value of a ReadCacheEntry twice. "
					+ "Should only be performed once.");

			_data = val;
			//resume all threads waiting for _data
			notifyAll();
		}

		public synchronized void setInvalid() {
			_is_valid = false;
			notify(); // resume one waiting thread so it can try reading the data
		}
	}
}


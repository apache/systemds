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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.LocalFileUtils;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CacheMaintenanceService
{
	protected ExecutorService _pool = null;

	public CacheMaintenanceService() {
		//create new threadpool for async cleanup
		if( isAsync() )
			_pool = Executors.newCachedThreadPool();
	}

	public void deleteFile(String fname) {
		//sync or async file delete
		if( CacheableData.CACHING_ASYNC_FILECLEANUP )
			_pool.submit(new CacheMaintenanceService.FileCleanerTask(fname));
		else
			LocalFileUtils.deleteFileIfExists(fname, true);
	}

	public void serializeData(ByteBuffer bbuff, CacheBlock<?> cb) {
		//sync or async file delete
		if( CacheableData.CACHING_ASYNC_SERIALIZE )
			_pool.submit(new CacheMaintenanceService.DataSerializerTask(bbuff, cb));
		else {
			try {
				bbuff.serializeBlock(cb);
			}
			catch(IOException ex) {
				throw new DMLRuntimeException(ex);
			}
		}
	}

	public void close() {
		//execute pending tasks and shutdown pool
		if( isAsync() )
			_pool.shutdown();
	}

	@SuppressWarnings("unused")
	public boolean isAsync() {
		return CacheableData.CACHING_ASYNC_FILECLEANUP
			|| CacheableData.CACHING_ASYNC_SERIALIZE;
	}

	private static class FileCleanerTask implements Runnable {
		private String _fname = null;

		public FileCleanerTask( String fname ) {
			_fname = fname;
		}

		@Override
		public void run() {
			LocalFileUtils.deleteFileIfExists(_fname, true);
		}
	}

	private static class DataSerializerTask implements Runnable {
		private ByteBuffer _bbuff = null;
		private CacheBlock<?> _cb = null;

		public DataSerializerTask(ByteBuffer bbuff, CacheBlock<?> cb) {
			_bbuff = bbuff;
			_cb = cb;
		}

		@Override
		public void run() {
			try {
				_bbuff.serializeBlock(_cb);
			}
			catch(IOException ex) {
				throw new DMLRuntimeException(ex);
			}
		}
	}
}

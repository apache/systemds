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

package org.apache.sysds.runtime.util;

import java.lang.ref.WeakReference;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;

/**
 * This memory monitor periodically calls garbage collection
 * in order to obtain the actual memory consumption during
 * runtime.
 */
public class MemoryMonitor implements Runnable {

	@Override
	public void run() {
		while( true ) {
			try {
				//wait for one second
				Thread.sleep(1000);
				
				//call garbage collection (just a hint) until garbage collection 
				//was actually trigger as indicated by a cleaned weak reference
				WeakReference<int[]> wr = new WeakReference<int[]>(new int[1024]);
				while(wr.get() != null) {
					System.gc();
				}
				
				long mem = Runtime.getRuntime().maxMemory() - Runtime.getRuntime().freeMemory();
				System.out.println("MemoryMonitor: "+ OptimizerUtils.toMB(mem)+" MB used.");
			}
			catch (InterruptedException e) {
				throw new DMLRuntimeException(e);
			}
		}
	}
}

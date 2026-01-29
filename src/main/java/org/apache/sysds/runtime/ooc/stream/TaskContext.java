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

import java.util.ArrayDeque;

public class TaskContext {
	private static final ThreadLocal<TaskContext> CTX = new ThreadLocal<>();

	private ArrayDeque<Runnable> _deferred;

	public static TaskContext getContext() {
		return CTX.get();
	}

	public static void setContext(TaskContext context) {
		if(CTX.get() != null)
			throw new IllegalStateException();
		CTX.set(context);
	}

	public static void clearContext() {
		CTX.remove();
	}

	public static void defer(Runnable deferred) {
		TaskContext ctx = CTX.get();
		if(ctx == null) {
			deferred.run();
			return;
		}
		if(ctx._deferred == null)
			ctx._deferred = new ArrayDeque<>();
		ctx._deferred.add(deferred);
		if(ctx._deferred.size() > 3)
			System.out.println("[WARN] Defer size bigger than 3");
	}

	public static boolean runDeferred() {
		TaskContext ctx = CTX.get();
		if(ctx == null || ctx._deferred == null || ctx._deferred.isEmpty())
			return false;
		Runnable deferred;
		while((deferred = ctx._deferred.poll()) != null)
			deferred.run();
		return true;
	}
}

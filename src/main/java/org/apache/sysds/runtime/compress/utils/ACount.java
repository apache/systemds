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

package org.apache.sysds.runtime.compress.utils;

public abstract class ACount {
	public int count;
	public int id;

	protected abstract Object K();

	@Override
	public final String toString() {
		return "[K:" + K() + ", <ID:" + id + ",C:" + count + ">]";
	}

	public static class DArrCounts extends ACount {
		public final DblArray key;

		public DArrCounts(DblArray key, int id) {
			this.key = key;
			this.id = id;
			count = 1;
		}

		public void inc() {
			count++;
		}

		protected Object K() {
			return key;
		}
	}

	public static class DCounts extends ACount {
		final public double key;

		public DCounts(double key, int id) {
			this.key = key;
			this.id = id;
			count = 1;
		}

		public void inc() {
			count++;
		}

		protected Object K() {
			return key;
		}

	}
}

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

package org.apache.sysds.runtime.compress.colgroup.mapping;

import java.io.DataOutput;
import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public abstract class AMapToData {

	protected static final Log LOG = LogFactory.getLog(AMapToData.class.getName());

	public abstract int getIndex(int n);

	public abstract void set(int n, int v);

	public abstract void fill(int v);

	public abstract long getInMemorySize();

	public abstract long getExactSizeOnDisk();

	public abstract int size();

	public abstract void write(DataOutput out) throws IOException;

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" [");
		for(int i = 0; i < size() - 1; i++)
			sb.append(getIndex(i) + ", ");
		sb.append(getIndex(size() - 1));
		sb.append("]");
		return sb.toString();
	}
}

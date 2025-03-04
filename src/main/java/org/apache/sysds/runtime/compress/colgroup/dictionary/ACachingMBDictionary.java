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
 * O
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import java.lang.ref.SoftReference;

public abstract class ACachingMBDictionary extends ADictionary {

	private static final long serialVersionUID = 7035552219254994595L;
	/** A Cache to contain a materialized version of the identity matrix. */
	protected volatile SoftReference<MatrixBlockDictionary> cache = null;

	@Override
	public final MatrixBlockDictionary getMBDict(int nCol) {
		if(cache != null) {
			MatrixBlockDictionary r = cache.get();
			if(r != null)
				return r;
		}
		MatrixBlockDictionary ret = createMBDict(nCol);
		cache = new SoftReference<>(ret);
		return ret;
	}

	public abstract MatrixBlockDictionary createMBDict(int nCol);
}

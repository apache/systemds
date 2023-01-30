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

package org.apache.sysds.runtime.compress.colgroup.indexes;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public abstract class AColIndex implements IColIndex {

	protected static final Log LOG = LogFactory.getLog(AColIndex.class.getName());

	@Override
	public int hashCode() {
		return hashCode(iterator());
	}

	@Override
	public boolean equals(Object other) {
		return other instanceof IColIndex && this.equals((IColIndex) other);
	}

	@Override
	public boolean contains(IColIndex a, IColIndex b) {
		return a != null && b != null && findIndex(a.get(0)) >= 0 && findIndex(b.get(0)) >= 0;
	}

	@Override
	public boolean containsStrict(IColIndex a, IColIndex b) {
		if(a != null && b != null && a.size() + b.size() == size()) {
			IIterate ia = a.iterator();
			while(ia.hasNext()) {
				if(!(findIndex(ia.next()) >= 0))
					return false;
			}

			IIterate ib = b.iterator();
			while(ib.hasNext()) {
				if(!(findIndex(ib.next()) >= 0))
					return false;
			}
			return true;
		}
		return false;
	}

	private static int hashCode(IIterate it) {
		int res = 1;
		while(it.hasNext())
			res = 31 * res + it.next();
		return res;
	}
}

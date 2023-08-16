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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;

public abstract class DDCScheme extends ACLAScheme {

	// TODO make it into a soft reference
	protected ADictionary lastDict;

	protected DDCScheme(IColIndex cols) {
		super(cols);
	}

	/**
	 * Create a scheme for the DDC compression given
	 * 
	 * @param g A DDC Column group
	 * @return A DDC Compression scheme
	 */
	public static DDCScheme create(ColGroupDDC g) {
		return g.getNumCols() == 1 ? new DDCSchemeSC(g) : new DDCSchemeMC(g);
	}

	/**
	 * Create a scheme for the DDC compression given a list of columns.
	 * 
	 * @param cols The columns to compress
	 * @return A DDC Compression scheme
	 */
	public static DDCScheme create(IColIndex cols) {
		return cols.size() == 1 ? new DDCSchemeSC(cols) : new DDCSchemeMC(cols);
	}

	@Override
	protected final IColIndex getColIndices() {
		return cols;
	}

	protected abstract Object getMap();

	@Override
	public final String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\nCols: ");
		sb.append(cols);
		sb.append("\nMap:  ");
		sb.append(getMap());
		return sb.toString();
	}

}

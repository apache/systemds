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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ASDC;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCFOR;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public abstract class SDCScheme extends ACLAScheme {

	// TODO make it into a soft reference
	protected IDictionary lastDict;

	protected SDCScheme(IColIndex cols) {
		super(cols);
	}

	public static SDCScheme create(ASDC g) {
		if(g instanceof ColGroupSDCFOR)
			throw new NotImplementedException();
		if(g.getColIndices().size() == 1)
			return new SDCSchemeSC(g);
		else
			return new SDCSchemeMC(g);
	}

	public static SDCScheme create(ASDCZero g) {
		if(g.getColIndices().size() == 1)
			return new SDCSchemeSC(g);
		else
			return new SDCSchemeMC(g);
	}

	@Override
	public AColGroup encode(MatrixBlock data, IColIndex columns) {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException("Unimplemented method 'encode'");
	}

	@Override
	public ICLAScheme update(MatrixBlock data, IColIndex columns) {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException("Unimplemented method 'update'");
	}

	protected abstract Object getDef();

	protected abstract Object getMap();

	@Override
	public final String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\nCols: ");
		sb.append(cols);
		sb.append("\nDef:  ");
		sb.append(getDef());
		sb.append("\nMap:  ");
		sb.append(getMap());
		return sb.toString();
	}

}

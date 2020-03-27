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

package org.apache.sysds.hops.cost;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class VarStats 
{
	DataCharacteristics _dc;
	boolean _inmem = false;
	
	public VarStats( long rlen, long clen, int blen, long nnz, boolean inmem ) {
		_dc = new MatrixCharacteristics(rlen, clen, blen, nnz);
		_inmem = inmem;
	}
	
	public long getRows() {
		return _dc.getRows();
	}
	
	public long getCols() {
		return _dc.getCols();
	}
	
	public double getSparsity() {
		return OptimizerUtils.getSparsity(_dc);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("VarStats: [");
		sb.append("rlen = ");
		sb.append(_dc.getRows());
		sb.append(", clen = ");
		sb.append(_dc.getCols());
		sb.append(", nnz = ");
		sb.append(_dc.getNonZeros());
		sb.append(", inmem = ");
		sb.append(_inmem);
		sb.append("]");
		return sb.toString();
	}
	
	@Override
	public Object clone() {
		return new VarStats(_dc.getRows(), _dc.getCols(),
			_dc.getBlocksize(), _dc.getNonZeros(), _inmem );
	}
}

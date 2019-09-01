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

package org.tugraz.sysds.hops.cost;

import org.tugraz.sysds.hops.OptimizerUtils;

public class VarStats 
{
	long _rlen = -1;
	long _clen = -1;
	int _blen = -1;
	long _nnz = -1;
	boolean _inmem = false;
	
	public VarStats( long rlen, long clen, int blen, long nnz, boolean inmem ) {
		_rlen = rlen;
		_clen = clen;
		_blen = blen;
		_nnz = nnz;
		_inmem = inmem;
	}
	
	public double getSparsity() {
		return OptimizerUtils.getSparsity(_rlen, _clen, _nnz);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("VarStats: [");
		sb.append("rlen = ");
		sb.append(_rlen);
		sb.append(", clen = ");
		sb.append(_clen);
		sb.append(", nnz = ");
		sb.append(_nnz);
		sb.append(", inmem = ");
		sb.append(_inmem);
		sb.append("]");
		return sb.toString();
	}
	
	@Override
	public Object clone() {
		return new VarStats(_rlen, _clen, _blen, (long)_nnz, _inmem );
	}
}

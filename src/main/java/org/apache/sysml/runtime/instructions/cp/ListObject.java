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

package org.apache.sysml.runtime.instructions.cp;

import java.util.Arrays;
import java.util.List;

import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;

public class ListObject extends Data {
	private static final long serialVersionUID = 3652422061598967358L;

	private final List<String> _names;
	private final List<Data> _data;
	private boolean[] _dataState = null;
	
	public ListObject(List<Data> data) {
		super(DataType.LIST, ValueType.UNKNOWN);
		_data = data;
		_names = null;
	}

	public ListObject(List<Data> data, List<String> names) {
		super(DataType.LIST, ValueType.UNKNOWN);
		_data = data;
		_names = names;
	}
	
	public void setStatus(boolean[] status) {
		_dataState = status;
	}
	
	public boolean[] getStatus() {
		return _dataState;
	}
	
	public int getLength() {
		return _data.size();
	}
	
	public Data slice(int ix) {
		return _data.get(ix);
	}
	
	public ListObject slice(int ix1, int ix2) {
		ListObject ret = new ListObject(_data.subList(ix1, ix2 + 1),
			(_names != null) ? _names.subList(ix1, ix2 + 1) : null);
		ret.setStatus(Arrays.copyOfRange(_dataState, ix2, ix2 + 1));
		return ret;
	}
	
	public Data slice(String name) {
		//check for existing named list
		if (_names == null)
			throw new DMLRuntimeException("Invalid lookup by name" + " in unnamed list: " + name + ".");

		//find position and check for existing entry
		int pos = _names.indexOf(name);
		if (pos < 0 || pos >= _data.size())
			throw new DMLRuntimeException("List lookup returned no entry for name='" + name + "'");

		//return existing entry
		return slice(pos);
	}
	
	public ListObject slice(String name1, String name2) {
		//check for existing named list
		if (_names == null)
			throw new DMLRuntimeException("Invalid lookup by name" + " in unnamed list: " + name1 + ", " + name2 + ".");

		//find position and check for existing entry
		int pos1 = _names.indexOf(name1);
		int pos2 = _names.indexOf(name2);
		if (pos1 < 0 || pos1 >= _data.size())
			throw new DMLRuntimeException("List lookup returned no entry for name='" + name1 + "'");
		if (pos2 < 0 || pos2 >= _data.size())
			throw new DMLRuntimeException("List lookup returned no entry for name='" + name2 + "'");

		//return list object
		return slice(pos1, pos2);
	}

	public List<String> getNames() {
		return _names;
	}

	public String getName(int ix) {
		return (_names == null) ? null : _names.get(ix);
	}

	public boolean isNamedList() {
		return _names != null;
	}

	public List<Data> getData() {
		return _data;
	}

	public long getDataSize() {
		return _data.stream().filter(data -> data instanceof CacheableData)
			.mapToLong(data -> ((CacheableData<?>) data).getDataSize()).sum();
	}

	@Override
	public String getDebugName() {
		return toString();
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("List (");
		for (int i = 0; i < _data.size(); i++) {
			if (i > 0)
				sb.append(", ");
			if (_names != null) {
				sb.append(_names.get(i));
				sb.append("=");
			}
			sb.append(_data.get(i).toString());
		}
		sb.append(")");
		return sb.toString();
	}
}

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

package org.apache.sysds.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.lineage.LineageItem;

public class ListObject extends Data {
	private static final long serialVersionUID = 3652422061598967358L;

	private final List<Data> _data;
	private boolean[] _dataState = null;
	private List<String> _names = null;
	private int _nCacheable;
	private List<LineageItem> _lineage = null;
	
	public ListObject(List<Data> data) {
		this(data, null, null);
	}

	public ListObject(List<Data> data, List<String> names) {
		this(data, names, null);
	}

	public ListObject(List<Data> data, List<String> names, List<LineageItem> lineage) {
		super(DataType.LIST, ValueType.UNKNOWN);
		_data = data;
		_names = names;
		_lineage = lineage;
		_nCacheable = (int) data.stream().filter(
			d -> d instanceof CacheableData).count();
	}
	
	public ListObject(ListObject that) {
		this(new ArrayList<>(that._data), (that._names != null) ?
			new ArrayList<>(that._names) : null, (that._lineage != null) ?
			new ArrayList<>(that._lineage) : null);
		if( that._dataState != null )
			_dataState = Arrays.copyOf(that._dataState, getLength());
	}
	
	public void deriveAndSetStatusFromData() {
		_dataState = new boolean[_data.size()];
		for(int i=0; i<_data.size(); i++) {
			Data dat = _data.get(i);
			if( dat instanceof CacheableData<?> )
			_dataState[i] = ((CacheableData<?>) dat).isCleanupEnabled();
		}
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
	
	public int getNumCacheableData() {
		return _nCacheable;
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
	
	public Data getData(int ix) {
		return _data.get(ix);
	}
	
	public Data getData(String name) {
		return slice(name);
	}
	
	public List<LineageItem> getLineageItems() {
		return _lineage;
	}

	public long getDataSize() {
		return _data.stream().filter(data -> data instanceof CacheableData)
			.mapToLong(data -> ((CacheableData<?>) data).getDataSize()).sum();
	}
	
	public boolean checkAllDataTypes(DataType dt) {
		return _data.stream().allMatch(d -> d.getDataType()==dt);
	}
	
	public Data slice(int ix) {
		return _data.get(ix);
	}

	public LineageItem getLineageItem(int ix) {
		LineageItem li = _lineage != null ? _lineage.get(ix):null;
		return li;
	}
	
	public ListObject slice(int ix1, int ix2) {
		ListObject ret = new ListObject(_data.subList(ix1, ix2 + 1),
			(_names != null) ? _names.subList(ix1, ix2 + 1) : null,
			(_lineage != null) ? _lineage.subList(ix1, ix2 + 1) : null);
		if( _dataState != null )
			ret.setStatus(Arrays.copyOfRange(_dataState, ix2, ix2 + 1));
		return ret;
	}
	
	public Data slice(String name) {
		//lookup position by name, incl error handling
		int pos = getPosForName(name);
		
		//return existing entry
		return slice(pos);
	}
	
	public ListObject slice(String name1, String name2) {
		//lookup positions by name, incl error handling
		int pos1 = getPosForName(name1);
		int pos2 = getPosForName(name2);
		
		//return list object
		return slice(pos1, pos2);
	}
	
	public ListObject copy() {
		List<String> names = isNamedList() ? new ArrayList<>(getNames()) : null;
		List<LineageItem> LineageItems = getLineageItems() != null ? new ArrayList<>(getLineageItems()) : null;
		ListObject ret = new ListObject(new ArrayList<>(getData()), names, LineageItems);
		if( getStatus() != null )
			ret.setStatus(Arrays.copyOf(getStatus(), getLength()));
		return ret;
	}
	
	public ListObject set(int ix, Data data) {
		_data.set(ix, data);
		return this;
	}
	
	public ListObject set(int ix1, int ix2, ListObject data) {
		int range = ix2 - ix1 + 1;
		if( range != data.getLength() || range > getLength() ) {
			throw new DMLRuntimeException("List leftindexing size mismatch: length(lhs)="
				+getLength()+", range=["+ix1+":"+ix2+"], legnth(rhs)="+data.getLength());
		}
		
		//copy rhs list object including meta data
		if( range == getLength() ) {
			//overwrite all entries in left hand side
			_data.clear(); _data.addAll(data.getData());
			System.arraycopy(data.getStatus(), 0, _dataState, 0, range);
			if( data.isNamedList() )
				_names = new ArrayList<>(data.getNames());
		}
		else {
			//overwrite entries of subrange in left hand side
			for( int i=ix1; i<=ix2; i++ ) {
				set(i, data.slice(i-ix1));
				_dataState[i] = data._dataState[i-ix1];
				if( isNamedList() && data.isNamedList() )
					_names.set(i, data.getName(i-ix1));
			}
		}
		return this;
	}
	
	public Data set(String name, Data data) {
		//lookup position by name, incl error handling
		int pos = getPosForName(name);
		
		//set entry into position
		return set(pos, data);
	}
	
	public ListObject set(String name1, String name2, ListObject data) {
		//lookup positions by name, incl error handling
		int pos1 = getPosForName(name1);
		int pos2 = getPosForName(name2);
		
		//set list into position range
		return set(pos1, pos2, data);
	}
	
	public ListObject add(Data dat, LineageItem li) {
		add(null, dat, li);
		return this;
	}
	
	public ListObject add(String name, Data dat, LineageItem li) {
		if( _names != null && name == null )
			throw new DMLRuntimeException("Cannot add to a named list");
		//otherwise append and ignore name
		if( _names != null )
			_names.add(name);
		_data.add(dat);
		if (_lineage == null && li!= null) 
			_lineage = new ArrayList<>();
		if (li != null)
			_lineage.add(li);
		return this;
	}
	
	public ListObject remove(int pos) {
		ListObject ret = new ListObject(Arrays.asList(_data.get(pos)),
				null, _lineage != null?Arrays.asList(_lineage.get(pos)):null);
		_data.remove(pos);
		if (_lineage != null)
			_lineage.remove(pos);
		if( _names != null )
			_names.remove(pos);
		return ret;
	}
	
	private int getPosForName(String name) {
		//check for existing named list
		if ( _names == null )
			throw new DMLRuntimeException("Invalid indexing by name" + " in unnamed list: " + name + ".");
		
		//find position and check for existing entry
		int pos = _names.indexOf(name);
		if (pos < 0 || pos >= _data.size())
			throw new DMLRuntimeException("List indexing returned no entry for name='" + name + "'");
		return pos;
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

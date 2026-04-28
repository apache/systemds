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

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;

public class ListObject extends Data implements Externalizable {
	private static final long serialVersionUID = 3652422061598967358L;

	private final List<Data> _data;
	private boolean[] _dataState = null;
	private List<String> _names = null;
	private int _nCacheable;
	private List<LineageItem> _lineage = null;

	/*
	 * No op constructor for Externalizable interface
	 */
	public ListObject() {
		super(DataType.LIST, ValueType.UNKNOWN);
		_data = new ArrayList<>();
	}
	
	public ListObject(List<Data> data) {
		this(data, null, null);
	}
	
	public ListObject(Data[] data) {
		this(Arrays.asList(data), null, null);
	}

	public ListObject(List<Data> data, List<String> names) {
		this(data, names, null);
	}
	
	public ListObject(Data[] data, String[] names) {
		this(Arrays.asList(data), Arrays.asList(names), null);
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

	public void setNames(List<String> names) {
		_names = names;
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
	
	public LineageItem getLineageItem(String name) {
		//lookup position by name, incl error handling
		int pos = getPosForName(name);
		return getLineageItem(pos);
	}
	
	public List<LineageItem> getLineageItems() {
		return _lineage;
	}

	public boolean contains(Data d) {
		return _data.stream().anyMatch(lo -> lo instanceof ListObject ?
			(lo == d || ((ListObject)lo).contains(d)) : lo == d);
	}
	
	public boolean contains(String name) {
		return _names != null
			&& _names.contains(name);
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

	public ListObject set(int ix, Data data, LineageItem li) {
		_data.set(ix, data);
		if (li != null) _lineage.set(ix, li);
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
	
	public Data set(String name, Data data, LineageItem li) {
		//lookup position by name, incl error handling
		int pos = getPosForName(name);
		//set entry into position
		return set(pos, data, li);
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

	/**
	 * Removes the element at the specified position from the list
	 * and returns that element as the only element in a new ListObject.
	 * @param pos position of element in the list
	 * @return new ListObject with the specified element
	 */
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
		return toString(false);
	}
	
	@Override
	public String toString(boolean metaOnly) {
		StringBuilder sb = new StringBuilder("List (");
		for (int i = 0; i < _data.size(); i++) {
			if (i > 0)
				sb.append(", ");
			if (_names != null) {
				sb.append(_names.get(i));
				sb.append("=");
			}
			if(metaOnly)
				sb.append(_data.get(i).getClass().getSimpleName());
			else
				sb.append(_data.get(i).toString());
		}
		sb.append(")");
		return sb.toString();
	}

	/**
	 * Redirects the default java serialization via externalizable to our default
	 * hadoop writable serialization for efficient broadcast/rdd serialization.
	 *
	 * @param out object output
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		// write out length
		out.writeInt(getLength());
		// write out num cacheable
		out.writeInt(_nCacheable);

		// write out names for named list
		out.writeBoolean(getNames() != null);
		if(getNames() != null) {
			for (int i = 0; i < getLength(); i++) {
				out.writeObject(_names.get(i));
			}
		}

		// write out data
		for(int i = 0; i < getLength(); i++) {
			Data d = getData(i);
			out.writeObject(d.getDataType());
			out.writeObject(d.getValueType());
			switch(d.getDataType()) {
				case LIST:
					ListObject lo = (ListObject) d;
					out.writeObject(lo);
					break;
				case MATRIX:
					MatrixObject mo = (MatrixObject) d;
					MetaDataFormat md = (MetaDataFormat) mo.getMetaData();
					DataCharacteristics dc = md.getDataCharacteristics();

					out.writeObject(dc.getRows());
					out.writeObject(dc.getCols());
					out.writeObject(dc.getBlocksize());
					out.writeObject(dc.getNonZeros());
					out.writeObject(md.getFileFormat());
					out.writeObject(mo.acquireReadAndRelease());
					break;
				case SCALAR:
					ScalarObject so = (ScalarObject) d;
					out.writeObject(so.getStringValue());
					break;
				case ENCRYPTED_CIPHER:
				case ENCRYPTED_PLAIN:
					Encrypted e = (Encrypted) d;
					int[] dims = e.getDims();
					dc = e.getDataCharacteristics();
					out.writeObject(dims);
					out.writeObject(dc.getRows());
					out.writeObject(dc.getCols());
					out.writeObject(dc.getBlocksize());
					out.writeObject(dc.getNonZeros());
					out.writeObject(e.getData());
					break;
				default:
					throw new DMLRuntimeException("Unable to serialize datatype " + dataType);
			}
		}
	}

	/**
	 * Redirects the default java serialization via externalizable to our default
	 * hadoop writable serialization for efficient broadcast/rdd deserialization.
	 *
	 * @param in object input
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
		// read in length
		int length = in.readInt();
		// read in num cacheable
		_nCacheable = in.readInt();

		// read in names
		Boolean names = in.readBoolean();
		if(names) {
			_names = new ArrayList<>();
			for (int i = 0; i < length; i++) {
				_names.add((String) in.readObject());
			}
		}

		// read in data
		for(int i = 0; i < length; i++) {
			DataType dataType = (DataType) in.readObject();
			ValueType valueType = (ValueType) in.readObject();
			Data d;
			switch(dataType) {
				case LIST:
					d = (ListObject) in.readObject();
					break;
				case MATRIX:
					long rows = (long) in.readObject();
					long cols = (long) in.readObject();
					int blockSize = (int) in.readObject();
					long nonZeros = (long) in.readObject();
					Types.FileFormat fileFormat = (Types.FileFormat) in.readObject();

					// construct objects and set meta data
					MatrixCharacteristics matrixCharacteristics = new MatrixCharacteristics(rows, cols, blockSize, nonZeros);
					MetaDataFormat metaDataFormat = new MetaDataFormat(matrixCharacteristics, fileFormat);
					MatrixBlock matrixBlock = (MatrixBlock) in.readObject();

					d = new MatrixObject(valueType, Dag.getNextUniqueVarname(Types.DataType.MATRIX), metaDataFormat, matrixBlock);
					break;
				case SCALAR:
					String value = (String) in.readObject();
					ScalarObject so;
					switch (valueType) {
						case INT64:     so = new IntObject(Long.parseLong(value)); break;
						case FP64:  	so = new DoubleObject(Double.parseDouble(value)); break;
						case BOOLEAN: 	so = new BooleanObject(Boolean.parseBoolean(value)); break;
						case STRING:  	so = new StringObject(value); break;
						default:
							throw new DMLRuntimeException("Unable to parse valuetype " + valueType);
					}
					d = so;
					break;
				case ENCRYPTED_CIPHER:
				case ENCRYPTED_PLAIN:
					int[] dims = (int[]) in.readObject();
					rows = (long) in.readObject();
					cols = (long) in.readObject();
					blockSize = (int) in.readObject();
					nonZeros = (long) in.readObject();
					byte[] data = (byte[])in.readObject();
					DataCharacteristics dc = new MatrixCharacteristics(rows, cols, blockSize, nonZeros);
					if (dataType == DataType.ENCRYPTED_CIPHER) {
						d = new CiphertextMatrix(dims, dc, data);
					} else {
						d = new PlaintextMatrix(dims, dc, data);
					}
					break;
				default:
					throw new DMLRuntimeException("Unable to deserialize datatype " + dataType);
			}
			_data.add(d);
		}
	}

	/**
	 * Gets list of current cleanupFlag values recursively for every element
	 * in the list and in its sublists of type CacheableData. The order is
	 * as CacheableData elements are discovered during DFS. Elements that
	 * are not of type CacheableData are skipped.
	 *
	 * @return list of booleans containing the _cleanupFlag values.
	 */
	public List<Boolean> getCleanupStates() {
		List<Boolean> varsState = new LinkedList<>();
		for (Data dat : this.getData()) {
			if (dat instanceof CacheableData<?>)
				varsState.add(((CacheableData<?>)dat).isCleanupEnabled());
			else if (dat instanceof ListObject)
				varsState.addAll(((ListObject)dat).getCleanupStates());
		}
		return varsState;
	}

	/**
	 * Sets the cleanupFlag values recursively for every element of type
	 * CacheableData in the list and in its sublists to the provided flag
	 * value.
	 *
	 * @param flag New value for every CacheableData element.
	 */
	public void enableCleanup(boolean flag) {
		for (Data dat : this.getData()) {
			if (dat instanceof CacheableData<?>)
				((CacheableData<?>)dat).enableCleanup(flag);
			if (dat instanceof ListObject)
				((ListObject)dat).enableCleanup(flag);
		}
	}

	/**
	 * Sets the cleanupFlag values recursively for every element of type
	 * CacheableData in the list and in its sublists to the provided values
	 * in flags. The cleanupFlag value of the i-th CacheableData element
	 * in the list (counted in the order of DFS) is set to the i-th value
	 * in flags.
	 *
	 * @param flags Queue of values in the same order as its corresponding
	 *              elements occur in DFS.
	 */
	public void enableCleanup(Queue<Boolean> flags) {
		for (Data dat : this.getData()) {
			if (dat instanceof CacheableData<?>)
				((CacheableData<?>)dat).enableCleanup(Boolean.TRUE.equals(flags.poll()));
			else if (dat instanceof ListObject)
				((ListObject)dat).enableCleanup(flags);
		}
	}
}

/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.data;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;

/**
 * This represent the indexes to the blocks of the tensor.
 * Please note that these indexes are 1-based, whereas the data in the block are zero-based (as they are double arrays).
 */
public class TensorIndexes implements WritableComparable<TensorIndexes>, Externalizable {
	private static final long serialVersionUID = -8596795142899904117L;

	private long[] _ix;

	///////////////////////////
	// constructors

	public TensorIndexes() {
		//do nothing
	}

	public TensorIndexes(long[] ix) {
		setIndexes(ix);
	}

	public TensorIndexes(TensorIndexes indexes) {
		setIndexes(indexes._ix);
	}

	///////////////////////////
	// get/set methods

	public long getIndex(int dim) {
		return _ix[dim];
	}

	public long[] getIndexes() {
		return _ix;
	}

	public int getNumDims() {
		return _ix.length;
	}

	public TensorIndexes setIndexes(long[] ix) {
		// copy
		_ix = ix.clone();
		return this;
	}

	public TensorIndexes setIndexes(TensorIndexes that) {
		setIndexes(that._ix);
		return this;
	}

	@Override
	public int compareTo(TensorIndexes other) {
		for (int i = 0; i < _ix.length; i++) {
			if (_ix[i] != other._ix[i])
				return _ix[i] < other._ix[i] ? -1 : 1;
		}
		return 0;
	}

	@Override
	public boolean equals(Object other) {
		if (!(other instanceof TensorIndexes))
			return false;

		TensorIndexes tother = (TensorIndexes) other;
		return compareTo(tother) == 0;
	}

	@Override
	public int hashCode() {
		return Arrays.hashCode(_ix);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder("(");
		for (int i = 0; i < _ix.length - 1; i++)
			sb.append(_ix[i]).append(", ");
		sb.append(_ix[_ix.length - 1]).append(")");
		return sb.toString();
	}

	////////////////////////////////////////////////////
	// implementation of Writable read/write

	@Override
	public void readFields(DataInput in)
			throws IOException {
		_ix = new long[in.readInt()];
		for (int i = 0; i < _ix.length; i++) {
			_ix[i] = in.readLong();
		}
	}

	@Override
	public void write(DataOutput out)
			throws IOException {
		out.writeInt(_ix.length);
		for (long ix : _ix) {
			out.writeLong(ix);
		}
	}


	////////////////////////////////////////////////////
	// implementation of Externalizable read/write

	/**
	 * Redirects the default java serialization via externalizable to our default
	 * hadoop writable serialization for consistency/maintainability.
	 *
	 * @param is object input
	 * @throws IOException if IOException occurs
	 */
	public void readExternal(ObjectInput is)
			throws IOException {
		//default deserialize (general case)
		readFields(is);
	}

	/**
	 * Redirects the default java serialization via externalizable to our default
	 * hadoop writable serialization for consistency/maintainability.
	 *
	 * @param os object output
	 * @throws IOException if IOException occurs
	 */
	public void writeExternal(ObjectOutput os)
			throws IOException {
		//default serialize (general case)
		write(os);
	}
}

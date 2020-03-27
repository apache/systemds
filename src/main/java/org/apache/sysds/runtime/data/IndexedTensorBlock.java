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

package org.apache.sysds.runtime.data;

import java.io.Serializable;

public class IndexedTensorBlock implements Serializable {
	private static final long serialVersionUID = -6227311506740546446L;

	private TensorIndexes _indexes;
	private TensorBlock _block = null;

	public IndexedTensorBlock()
	{
		_indexes = new TensorIndexes();
	}

	public IndexedTensorBlock(TensorIndexes ind, TensorBlock b)
	{
		this();

		_indexes.setIndexes(ind);
		_block = b;
	}

	public IndexedTensorBlock(IndexedTensorBlock that)
	{
		this(that._indexes, that._block);
	}


	public TensorIndexes getIndexes()
	{
		return _indexes;
	}

	public TensorBlock getValue()
	{
		return _block;
	}

	public void set(TensorIndexes indexes2, TensorBlock block2) {
		_indexes.setIndexes(indexes2);
		_block = block2;
	}

	@Override
	public String toString() {
		return _indexes.toString() + ": \n"+_block;
	}
}

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

package org.apache.sysds.hops.estim;

import org.apache.sysds.hops.estim.SparsityEstimator.OpCode;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

/**
 * Helper class to represent matrix multiply operators in a DAG
 * along with references to its abstract data handles.
 */
public class MMNode 
{
	private final MMNode _m1;
	private final MMNode _m2;
	private final MatrixBlock _data;
	private final DataCharacteristics _mc;
	private Object _synops = null;
	private final OpCode _op;
	private final long[] _misc;
	
	public MMNode(MatrixBlock in) {
		_m1 = null;
		_m2 = null;
		_data = in;
		_mc = in.getDataCharacteristics();
		_op = null;
		_misc = null;
	}
	
	public MMNode(MMNode left, MMNode right, OpCode op, long[] misc) {
		_m1 = left;
		_m2 = right;
		_data = null;
		_mc = new MatrixCharacteristics(-1, -1, -1, -1);
		_op = op;
		_misc = misc;
	}
	
	public MMNode(MMNode left, MMNode right, OpCode op) {
		this(left, right, op, null);
	}
	
	public MMNode(MMNode left, OpCode op) {
		this(left, null, op);
	}
	
	public MMNode(MMNode left, OpCode op, long[] misc) {
		this(left, null, op, misc);
	}
	
	public void reset() {
		if( _m1 != null )
			_m1.reset();
		if( _m2 != null )
			_m2.reset();
		_synops = null;
	}
	
	public int getRows() {
		return (int)_mc.getRows();
	}
	
	public int getCols() {
		return (int)_mc.getCols();
	}
	
	public long[] getMisc() {
		return _misc;
	}
	
	public long getMisc(int pos) {
		if( _misc == null )
			throw new DMLRuntimeException("Extra meta data not available.");
		return _misc[pos];
	}
	
	public DataCharacteristics getDataCharacteristics() {
		return _mc;
	}
	
	public DataCharacteristics setDataCharacteristics(DataCharacteristics mc) {
		return _mc.set(mc); //implicit copy
	}
	
	public MMNode getLeft() {
		return _m1;
	}
	
	public MMNode getRight() {
		return _m2;
	}
	
	public boolean isLeaf() {
		return _data != null;
	}
	
	public MatrixBlock getData() {
		return _data;
	}
	
	public void setSynopsis(Object obj) {
		_synops = obj;
	}
	
	public Object getSynopsis() {
		return _synops;
	}
	
	public OpCode getOp() {
		return _op;
	}
}

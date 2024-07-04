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

package org.apache.sysds.resource.cost;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

public class VarStats 
{
	MatrixCharacteristics _mc;
	/**
	 * Size in memory estimate
	 * <li>-1 if not in memory yet</li>
	 * <li>0 if scalar</li>
	 */
	long _memory;
	/**
	 * true if object modified since last saved, or
	 * if HDFS file still doesn't exist
	 */
	boolean _dirty = false;

	RDDStats _rdd = null;

	Object[] _fileInfo = null;

	public VarStats(DataCharacteristics dc) {
		this(dc, -1);
	}

	public VarStats(DataCharacteristics dc, long sizeEstimate) {
		if (dc == null) {
			_mc = null;
		}
		else if (dc instanceof MatrixCharacteristics) {
			_mc = (MatrixCharacteristics) dc;
		} else {
			throw new RuntimeException("VarStats: expecting MatrixCharacteristics or null");
		}
		_memory = sizeEstimate;
	}

	public long getM() {
		return _mc.getRows();
	}

	public long getN() {
		return _mc.getCols();
	}

	public double getS() {
		return _mc == null? 1.0 : OptimizerUtils.getSparsity(_mc);
	}

	public long getCells() {
		return _mc.getRows() * _mc.getCols();
	}

	public double getCellsWithSparsity() {
		if (isSparse())
			return getCells() * getS();
		return (double) getCells();
	}

	public boolean isSparse() {
		return MatrixBlock.evalSparseFormatInMemory(_mc);
	}

	// clone() needed?
}

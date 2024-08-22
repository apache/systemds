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
	/**
	 * <li>null if scalar</li>
	 * <li>initialized if Matrix or Frame</li>
	 */
	MatrixCharacteristics characteristics;
	/**
	 * estimated size in memory
	 * <li>-1 if not in memory yet</li>
	 * <li>0 if scalar</li>
	 * <li>=>1 estimated loaded size in Bytes</li>
	 */
	long allocatedMemory;
	/**
	 * true if object modified since last saved, or
	 * if HDFS file still doesn't exist
	 */
	boolean isDirty = false;

	// needed for the cases of 'çpvar', 'fcall' or reblock
	int refCount;

	RDDStats rddStats = null;

	Object[] fileInfo = null;

	public VarStats(DataCharacteristics dc) {
		if (dc == null) {
			characteristics = null; // for scalar
			allocatedMemory = 0;
		} else if (dc instanceof MatrixCharacteristics) {
			characteristics = (MatrixCharacteristics) dc;
			allocatedMemory = -1;
		} else {
			throw new RuntimeException("Unexpected error: expecting MatrixCharacteristics or null");
		}
		refCount = 1;
	}

	public boolean isScalar() {
		return characteristics == null;
	}

	public long getM() {
		return isScalar()? 1 : characteristics.getRows();
	}

	public long getN() {
		return isScalar()? 1 : characteristics.getCols();
	}

	public double getS() {
		return isScalar()? 1.0 : OptimizerUtils.getSparsity(characteristics);
	}

	public long getCells() {
		return isScalar()? 1 : (characteristics.getRows() * characteristics.getCols());
	}

	public long getCellsWithSparsity() {
		if (isScalar()) return 1;
		if (isSparse())
			return (long) (getCells() * getS());
		return getCells();
	}

	public boolean isSparse() {
		return (!isScalar() && MatrixBlock.evalSparseFormatInMemory(characteristics));
	}

}

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
	// helps for debugging + carries value for scalar literals
	String varName;
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
	// refCount/selfRefCount cases of variables copying (operations 'çpvar' or 'fcall')
	// increase/decrease only one of them at a time (selfRefCount is not a refCount)
	int refCount;
	int selfRefCount;
	/**
	 * Always contains 2 elements:
	 * first elements: {@code String} with the source type (hdfs, s3 or local)
	 * second element: {@code Types.FileFormat} value
	 */
	Object[] fileInfo = null;
	RDDStats rddStats = null;

	public VarStats(String name, DataCharacteristics dc) {
		varName = name;
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
		selfRefCount = 1;
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

	public long getNNZ() { return characteristics.getNonZeros(); }

	public double getSparsity() {
		return isScalar()? 1.0 : OptimizerUtils.getSparsity(characteristics);
	}

	public long getCells() {
		return isScalar()? 1 : !characteristics.dimsKnown()? -1 :
				(characteristics.getRows() * characteristics.getCols());
	}

	public long getCellsWithSparsity() {
		if (isScalar()) return 1;
		return (long) (getCells() * getSparsity());
	}

	public boolean isSparse() {
		return (!isScalar() && MatrixBlock.evalSparseFormatInMemory(characteristics));
	}

	/**
	 * Meant to be used at testing
	 * @return corresponding RDD statistics
	 */
	public RDDStats getRddStats() {
		return rddStats;
	}

	/**
	 * Meant to be used at testing
	 * @param rddStats corresponding RDD statistics
	 */
	public void setRddStats(RDDStats rddStats) {
		this.rddStats = rddStats;
	}
}

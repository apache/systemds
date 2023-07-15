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

package org.apache.sysds.runtime.controlprogram.federated;

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.meta.DataCharacteristics;


/**
 * Class to represent the relation between a MatrixObject and the respective
 * LineageItem.
 * This class couples the MatrixObject with its LineageItem, where the
 * LineageItem is treated like meta information and the most important methods
 * of the MatrixObject are included as a simple forwarding to the MatrixObject.
 */
public class MatrixLineagePair extends MutablePair<MatrixObject, LineageItem> {
	private static final long serialVersionUID = 1L;

	public MatrixLineagePair() {
		super();
	}

	public MatrixLineagePair(MatrixObject mo, LineageItem li) {
		super(mo, li);
	}

	public static MatrixLineagePair of(MatrixObject mo, LineageItem li) {
		return new MatrixLineagePair(mo, li);
	}

	public MatrixObject getMO() {
		return left;
	}

	public LineageItem getLI() {
		return right;
	}

	public boolean isFederated() {
		return left.isFederated();
	}

	public boolean isFederated(FType ftype) {
		return left.isFederated(ftype);
	}

	public boolean isFederatedExcept(FType ftype) {
		return left.isFederatedExcept(ftype);
	}

	public FederationMap getFedMapping() {
		return left.getFedMapping();
	}

	public DataCharacteristics getDataCharacteristics() {
		return left.getDataCharacteristics();
	}

	public long getNumRows() {
		return left.getNumRows();
	}

	public long getNumColumns() {
		return left.getNumColumns();
	}

	public long getBlocksize() {
		return left.getBlocksize();
	}

	public long getNnz() {
		return left.getNnz();
	}

	public DataType getDataType() {
		return left.getDataType();
	}

	public long getDim(int dim) {
		return left.getDim(dim);
	}
}

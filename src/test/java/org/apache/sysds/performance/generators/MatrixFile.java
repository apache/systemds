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

package org.apache.sysds.performance.generators;

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataAll;

public class MatrixFile extends ConstMatrix {

	final private String path;

	private MatrixFile(String path, MatrixBlock mb) {
		super(mb);
		this.path = path;
	}

	public static MatrixFile create(String path) throws Exception {

		MetaDataAll mba = new MetaDataAll(path + ".mtd", false, true);
		DataCharacteristics ds = mba.getDataCharacteristics();
		FileFormat f = FileFormat.valueOf(mba.getFormatTypeString().toUpperCase());

		MatrixReader r = MatrixReaderFactory.createMatrixReader(f);
		MatrixBlock mb = r.readMatrixFromHDFS(path, ds.getRows(), ds.getCols(), ds.getBlocksize(), ds.getNonZeros());
		return new MatrixFile(path, mb);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" From file: ");
		sb.append(path);
		return sb.toString();
	}

}

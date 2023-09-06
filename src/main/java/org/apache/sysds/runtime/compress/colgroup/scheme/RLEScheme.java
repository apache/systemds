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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupRLE;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class RLEScheme extends ACLAScheme {

	private static boolean messagePrinted = false;
	// private final DoubleCountHashMap map;
	// private final DblArrayCountHashMap map;

	public RLEScheme(IColIndex cols) {
		super(cols);
		if(!messagePrinted)
			LOG.error("Not Implemented RLE Scheme yet");
		messagePrinted = true;
		throw new NotImplementedException();
	}

	public static ICLAScheme create(ColGroupRLE g) {
		return new RLEScheme(g.getColIndices());
	}

	@Override
	protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
		throw new NotImplementedException();
	}

	@Override
	protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {
		throw new NotImplementedException();
	}

	@Override
	protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
		throw new NotImplementedException();
	}

	@Override
	protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
		throw new NotImplementedException();
	}

	@Override
	public ACLAScheme clone() {
		throw new NotImplementedException();
	}

}

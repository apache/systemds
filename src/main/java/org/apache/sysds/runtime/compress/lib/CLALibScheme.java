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

package org.apache.sysds.runtime.compress.lib;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.indexes.SingleIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.CompressionScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.SchemeFactory;

public class CLALibScheme {

	public static CompressionScheme getScheme(CompressedMatrixBlock cmb) {
		return CompressionScheme.getScheme(cmb);
	}

	/**
	 * Generate a scheme with the given type of columnGroup and number of columns in each group
	 * 
	 * @param type  The type of encoding to use
	 * @param nCols The number of columns
	 * @return A scheme to generate.
	 */
	public static CompressionScheme genScheme(CompressionType type, int nCols) {
		ICLAScheme[] encodings = new ICLAScheme[nCols];
		for(int i = 0; i < nCols; i++)
			encodings[i] = SchemeFactory.create(new SingleIndex(i), type);
		return new CompressionScheme(encodings);
	}
}

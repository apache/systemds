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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import java.io.DataInput;
import java.io.IOException;

public class DictionaryFactory {

	public static ADictionary read(DataInput in) throws IOException {
		boolean lossy = in.readBoolean();
		if(lossy) {

			double scale = in.readDouble();
			int numVals = in.readInt();
			// read distinct values
			byte[] values = numVals == 0 ? null : new byte[numVals];
			for(int i = 0; i < numVals; i++)
				values[i] = in.readByte();
			return new QDictionary(values, scale);
		}
		else {
			int numVals = in.readInt();
			// read distinct values
			double[] values = new double[numVals];
			for(int i = 0; i < numVals; i++)
				values[i] = in.readDouble();
			return new Dictionary(values);
		}
	}

	public static long getInMemorySize(int nrValues, int nrColumns, boolean lossy) {
		if(lossy)
			return QDictionary.getInMemorySize(nrValues * nrColumns);
		else
			return Dictionary.getInMemorySize(nrValues * nrColumns);
	}
}

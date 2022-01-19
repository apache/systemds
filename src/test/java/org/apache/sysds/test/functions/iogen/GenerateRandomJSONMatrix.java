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

package org.apache.sysds.test.functions.iogen;

import org.apache.sysds.runtime.iogen.ReaderMappingJSON;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.functions.iogen.objects.NumericObject1;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;

public class GenerateRandomJSONMatrix extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";

	@Override public void setUp() {
	}

	@Test
	@Ignore
	public void generateDataset(){
		int nrows = 1;
		NumericObject1 ot = new NumericObject1();
		ArrayList<Object> olt = ot.getJSONFlatValues();
		int ncols = olt.size();
		double[][] data = new double[nrows][ncols];
		StringBuilder sampleRaw = new StringBuilder();
		for(int r = 0; r < nrows; r++) {
			NumericObject1 o = new NumericObject1();
			ArrayList<Object> ol = o.getJSONFlatValues();
			int index = 0;
			for(Object oi : ol) {
				if(oi != null)
					data[r][index++] = UtilFunctions.getDouble(oi);
				else
					data[r][index++] = 0;
			}
			sampleRaw.append(o.getJSON());
			if(r!=nrows-1)
				sampleRaw.append("\n");
		}

		MatrixBlock sampleMB = DataConverter.convertToMatrixBlock(data);
		ReaderMappingJSON.MatrixReaderMapping mappingJSON = new ReaderMappingJSON.MatrixReaderMapping(sampleRaw.toString(), sampleMB);
	}
}

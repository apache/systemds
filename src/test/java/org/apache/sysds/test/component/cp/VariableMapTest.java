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

package org.apache.sysds.test.component.cp;

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.junit.Assert;
import org.junit.Test;

public class VariableMapTest {

	@Test
	public void testPinnedMethods() {
		LocalVariableMap vars = createSymbolTable();
		Assert.assertTrue(vars.getPinnedDataSize() > 2e5);
		vars.releaseAcquiredData(); //no impact on pinned status
		Assert.assertTrue(vars.getPinnedDataSize() > 2e5);
		vars.removeAll();
	}
	
	@Test
	public void testSerializeDeserialize() {
		LocalVariableMap vars = createSymbolTable();
		LocalVariableMap vars2 = LocalVariableMap.deserialize(vars.serialize());
		vars2.setID(1);
		Assert.assertEquals(vars.toString(), vars2.toString());
		LocalVariableMap vars3 = (LocalVariableMap) vars2.clone();
		vars3.setID(1);
		Assert.assertEquals(vars.toString(), vars3.toString());
	}
	
	private LocalVariableMap createSymbolTable() {
		LocalVariableMap vars = new LocalVariableMap();
		vars.put("a", createPinnedMatrixObject(1));
		vars.put("b", createPinnedMatrixObject(2));
		return vars;
	}
	
	private MatrixObject createPinnedMatrixObject(int seed) {
		MatrixBlock mb1 = MatrixBlock.randOperations(150, 167, 0.3, 1, 1, "uniform", seed);
		MatrixObject mo = new MatrixObject(ValueType.FP64, "./tmp", 
			new MetaDataFormat(new MatrixCharacteristics(), FileFormat.BINARY));
		mo.acquireModify(mb1);
		mo.release();
		mo.enableCleanup(false);
		mo.setDirty(false);
		return mo;
	}
}

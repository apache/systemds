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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.junit.Test;

public class FrameApplySchema {


	@Test
	public void testApplySchema(){
		try{

			FrameBlock fb = genBoolean(10, 2);
			ValueType[] schema = new ValueType[]{ValueType.BOOLEAN,ValueType.BOOLEAN};
			FrameBlock ret = fb.applySchema(schema);
			assertTrue(ret.getColumn(0) instanceof BooleanArray);
			assertTrue(ret.getColumn(1) instanceof BooleanArray);
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private FrameBlock genBoolean(int row, int col){
		FrameBlock ret = new FrameBlock();
		Random r = new Random(31);
		for(int c = 0; c < col; c ++){
			String[] column = new String[row];
			for(int i = 0; i < row; i ++)
				column[i] = "" + r.nextBoolean();
			
			ret.appendColumn(column);
		}
		return ret;
	}
}

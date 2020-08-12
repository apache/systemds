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

package org.apache.sysds.test.functions.data.rand;

import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;


@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class RandRuntimePlatformUniformTest extends RandRuntimePlatformBase 
{

	private final static String TEST_CLASS_DIR = TEST_DIR + RandRuntimePlatformUniformTest.class.getSimpleName() + "/";
	
	public RandRuntimePlatformUniformTest(int r, int c, double sp, long sd) {
		super(r,c,sp,sd,"uniform");
	}
	
	@Override
	protected String getClassDir() {
		return TEST_CLASS_DIR;
	}
	
}

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

package org.apache.sysds.test.applications.nn;

import java.util.ArrayList;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class NNMaxPool2dComponentTest extends TestFolder {

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		for(int h = 0; h < 4; h++)
			for(int w = 0; w < 4; w++)
				tests.add(new Object[] {h, w});

		return tests;
	}

	@Parameterized.Parameter
	public int h;

	@Parameterized.Parameter(1)
	public int w;

	final static String[] argNames = new String[] {"$h", "$w"};

	@Test
	public void max_pool2d_padh_padw() {
		run("max_pool2d.dml", argNames, new Object[] {h, w});
	}

	@Override
	protected void run(String name, String[] var, Object[] val) {
		super.run("component/" + name, var, val);
	}
}

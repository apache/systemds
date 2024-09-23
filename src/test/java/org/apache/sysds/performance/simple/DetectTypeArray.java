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

package org.apache.sysds.performance.simple;

import java.util.Random;

import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.utils.stats.Timing;

public class DetectTypeArray {

	public static void main(String[] args) {
		Array<?> a = ArrayFactory.create(generateRandomFloatString(1000, 134));

		Timing t = new Timing();
		t.start();
		int N = 10000;
		for(int i = 0; i < N; i++)
			a.analyzeValueType();

		System.out.println(t.stop() / N);

	}

	public static String[] generateRandomFloatString(int size, int seed) {
		Random r = new Random(seed);
		String[] ret = new String[size];
		for(int i = 0; i < size; i++) {
			int e = r.nextInt(999);
			int a = r.nextInt(999);

			ret[i] = String.format("%d.%03d", e, a);
		}

		return ret;
	}

}

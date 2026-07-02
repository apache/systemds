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

package org.apache.sysds.test.component.format;

/**
 * Temporary probe used to verify the Java Format CI check: this file is written to comply with
 * dev/CodeStyle_eclipse.xml and should therefore NOT be flagged.
 */
public class FormatProbeClean {

	public int add(int a, int b) {
		int result = a + b;
		return result;
	}

	public String describe(int a, int b) {
		return "sum=" + add(a, b);
	}

	public int sub(int a, int b) {
		return a - b;
	}

	public int  div( int a,int b ){
		return a/b ;
	}
}

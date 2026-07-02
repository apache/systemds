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

/** Temporary probe with several distinct formatting mistakes to be flagged. */
public class FormatProbeErrors {

	// missing spaces around operators
	public int operators(int a, int b) {
		return a+b*2;
	}

	// Allman-style brace and extra spaces in the signature
	public int  brace( int a )
	{
		return a;
	}

	// over-indented body
	public int indent(int a) {
				return a;
	}

	// missing space after comma in parameter list
	public int comma(int a,int b) {
		return a + b;
	}
}

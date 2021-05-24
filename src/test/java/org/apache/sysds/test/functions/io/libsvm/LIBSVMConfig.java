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

package org.apache.sysds.test.functions.io.libsvm;

public class LIBSVMConfig {

	private String inSep;
	private String inIndSep;
	private int colCount;
	private String outSep;
	private String outIndSep;

	public LIBSVMConfig(String inSep, String inIndSep, int colCount, String outSep, String outIndSep) {
		this.inSep = inSep;
		this.inIndSep = inIndSep;
		this.colCount = colCount;
		this.outSep = outSep;
		this.outIndSep = outIndSep;
	}

	public String getInSep() {
		return inSep;
	}

	public void setInSep(String inSep) {
		this.inSep = inSep;
	}

	public String getInIndSep() {
		return inIndSep;
	}

	public void setInIndSep(String inIndSep) {
		this.inIndSep = inIndSep;
	}

	public int getColCount() {
		return colCount;
	}

	public void setColCount(int colCount) {
		this.colCount = colCount;
	}

	public String getOutSep() {
		return outSep;
	}

	public void setOutSep(String outSep) {
		this.outSep = outSep;
	}

	public String getOutIndSep() {
		return outIndSep;
	}

	public void setOutIndSep(String outIndSep) {
		this.outIndSep = outIndSep;
	}
}

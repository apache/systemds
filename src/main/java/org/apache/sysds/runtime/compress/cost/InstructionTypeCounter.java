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

package org.apache.sysds.runtime.compress.cost;

import java.io.Serializable;

public final class InstructionTypeCounter implements Serializable {

	private static final long serialVersionUID = 115L;

	protected int scans = 0;
	protected int decompressions = 0;
	protected int overlappingDecompressions = 0;
	protected int leftMultiplications = 0;
	protected int rightMultiplications = 0;
	protected int compressedMultiplications = 0;
	protected int dictionaryOps = 0; // base cost is one pass of dictionary
	protected int indexing = 0;
	protected boolean isDensifying = false;

	protected InstructionTypeCounter() {
	}

	public int getScans() {
		return scans;
	}

	public int getDecompressions() {
		return decompressions;
	}

	public int getOverlappingDecompressions() {
		return overlappingDecompressions;
	}

	public int getLeftMultipications() {
		return leftMultiplications;
	}

	public int getRightMultiplications() {
		return rightMultiplications;
	}

	public int getCompressedMultiplications() {
		return compressedMultiplications;
	}

	public int getDictionaryOps() {
		return dictionaryOps;
	}

	public int getIndexing() {
		return indexing;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("\nscans                     :%4d", scans));
		sb.append(String.format("\ndecompressions            :%4d", decompressions));
		sb.append(String.format("\noverlappingDecompressions :%4d", overlappingDecompressions));
		sb.append(String.format("\nleftMultiplications       :%4d", leftMultiplications));
		sb.append(String.format("\nrightMultiplications      :%4d", rightMultiplications));
		sb.append(String.format("\ncompressedMultiplications :%4d", compressedMultiplications));
		sb.append(String.format("\ndictionaryOps             :%4d", dictionaryOps));
		sb.append(String.format("\nindexing                  :%4d", indexing));
		return sb.toString();
	}
}

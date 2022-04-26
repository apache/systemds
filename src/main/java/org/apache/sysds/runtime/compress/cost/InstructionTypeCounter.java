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

	public InstructionTypeCounter() {
		// default no count.
	}

	public InstructionTypeCounter(int scans, int decompressions, int overlappingDecompressions, int leftMultiplications,
		int rightMultiplications, int compressedMultiplications, int dictionaryOps, int indexing, boolean isDensifying) {
		this.scans = scans;
		this.decompressions = decompressions;
		this.overlappingDecompressions = overlappingDecompressions;
		this.leftMultiplications = leftMultiplications;
		this.rightMultiplications = rightMultiplications;
		this.compressedMultiplications = compressedMultiplications;
		this.dictionaryOps = dictionaryOps;
		this.indexing = indexing;
		this.isDensifying = isDensifying;
	}

	public int getScans() {
		return scans;
	}

	public void incScans() {
		scans++;
	}

	public int getDecompressions() {
		return decompressions;
	}

	public void incDecompressions() {
		decompressions++;
	}

	public int getOverlappingDecompressions() {
		return overlappingDecompressions;
	}

	public void incOverlappingDecompressions() {
		overlappingDecompressions++;
	}

	public int getLeftMultiplications() {
		return leftMultiplications;
	}

	public void incLMM() {
		leftMultiplications++;
	}

	public void incLMM(int c) {
		leftMultiplications += c;
	}

	public int getRightMultiplications() {
		return rightMultiplications;
	}

	public void incRMM() {
		rightMultiplications++;
	}

	public void incRMM(int c) {
		rightMultiplications += c;
	}

	public int getCompressedMultiplications() {
		return compressedMultiplications;
	}

	public void incCMM() {
		compressedMultiplications++;
	}

	public int getDictionaryOps() {
		return dictionaryOps;
	}

	public void incDictOps() {
		dictionaryOps++;
	}

	public int getIndexing() {
		return indexing;
	}

	public void incIndexOp() {
		indexing++;
	}

	public static InstructionTypeCounter MMR(int count) {
		return new InstructionTypeCounter(0, 0, 0, 0, count, 0, 0, 0, false);
	}

	public static InstructionTypeCounter MML(int count) {
		return new InstructionTypeCounter(0, 0, 0, count, 0, 0, 0, 0, false);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		if(scans > 0)
			sb.append(String.format("Sca:%d;", scans));
		if(decompressions > 0)
			sb.append(String.format("DeC:%d;", decompressions));
		if(overlappingDecompressions > 0)
			sb.append(String.format("OvD:%d;", overlappingDecompressions));
		if(leftMultiplications > 0)
			sb.append(String.format("LMM:%d;", leftMultiplications));
		if(rightMultiplications > 0)
			sb.append(String.format("RMM:%d;", rightMultiplications));
		if(compressedMultiplications > 0)
			sb.append(String.format("CMM:%d;", compressedMultiplications));
		if(dictionaryOps > 0)
			sb.append(String.format("dic:%d;", dictionaryOps));
		if(indexing > 0)
			sb.append(String.format("ind:%d;", indexing));
		if(sb.length() > 1)
			sb.setLength(sb.length() - 1); // remove last semicolon
		else
			sb.append("Empty");
		return sb.toString();
	}
}

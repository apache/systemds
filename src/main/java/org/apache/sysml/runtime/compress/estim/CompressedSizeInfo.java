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

package org.apache.sysml.runtime.compress.estim;

/**
 * 
 * A helper reusable object for maintaining bitmap sizes
 */
public class CompressedSizeInfo 
{
	private int _estCard = -1;
	private long _rleSize = -1; 
	private long _oleSize = -1;

	public CompressedSizeInfo() {
		
	}

	public CompressedSizeInfo(int estCard, long rleSize, long oleSize) {
		_estCard = estCard;
		_rleSize = rleSize;
		_oleSize = oleSize;
	}

	public void setRLESize(long rleSize) {
		_rleSize = rleSize;
	}
	
	public long getRLESize() {
		return _rleSize;
	}
	
	public void setOLESize(long oleSize) {
		_oleSize = oleSize;
	}

	public long getOLESize() {
		return _oleSize;
	}

	public long getMinSize() {
		return Math.min(_rleSize, _oleSize);
	}

	public void setEstCardinality(int estCard) {
		_estCard = estCard;
	}

	public int getEstCarinality() {
		return _estCard;
	}
}

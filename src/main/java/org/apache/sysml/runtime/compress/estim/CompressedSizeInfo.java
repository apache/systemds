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

import org.apache.sysml.runtime.compress.CompressedMatrixBlock;

/**
 * 
 * A helper reusable object for maintaining bitmap sizes
 */
public class CompressedSizeInfo 
{
	private final int _estCard;
	private final int _estNnz;
	private final long _rleSize; 
	private final long _oleSize;
	private final long _ddcSize;

	public CompressedSizeInfo(int estCard, int estNnz, long rleSize, long oleSize, long ddcSize) {
		_estCard = estCard;
		_estNnz = estNnz;
		_rleSize = rleSize;
		_oleSize = oleSize;
		_ddcSize = ddcSize;
	}

	public long getRLESize() {
		return _rleSize;
	}

	public long getOLESize() {
		return _oleSize;
	}
	
	public long getDDCSize() {
		return CompressedMatrixBlock.ALLOW_DDC_ENCODING ? 
			_ddcSize : Long.MAX_VALUE; 
	}

	public long getMinSize() {
		return Math.min(Math.min(
			getRLESize(), 
			getOLESize()),
			getDDCSize());
	}

	public int getEstCard() {
		return _estCard;
	}
	
	public int getEstNnz() {
		return _estNnz;
	}
}

/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */


package org.apache.sysml.runtime.matrix.data;

public enum RuntimeDataFormat 
{
	Invalid(true), 
	TextCell(true), 
	BinaryCell(true), 
	BinaryBlock(true), 
	SortInput(true), 
	SortOutput(true), 
	WeightedPair(true), 
	MatrixMarket(false);
	
	
	private boolean nativeFormat = false;
	private RuntimeDataFormat(boolean nf) {
		nativeFormat = nf;
	}
	
	public static RuntimeDataFormat parseFormat(String fmt) {
		if(fmt.equalsIgnoreCase("textcell")) 
			return TextCell;
		else if(fmt.equalsIgnoreCase("binarycell"))
			return BinaryCell;
		else if(fmt.equalsIgnoreCase("binaryblock"))
			return BinaryBlock;
		else if(fmt.equalsIgnoreCase("sort_input"))
			return SortInput;
		else if(fmt.equalsIgnoreCase("sort_output"))
			return SortOutput;
		else if(fmt.equalsIgnoreCase("weightedpair"))
			return WeightedPair;
		else if(fmt.equalsIgnoreCase("matrixmarket"))
			return MatrixMarket;
		else 
			return Invalid;
	}

	public boolean isNative() {
		return nativeFormat;
	}

}

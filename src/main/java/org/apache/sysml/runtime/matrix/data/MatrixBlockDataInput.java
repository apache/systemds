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

import java.io.IOException;

/**
 * Any data input that is intended to support fast deserialization / read
 * of entire blocks should implement this interface. On read of a matrix block
 * we check if the input stream is an implementation of this interface, if
 * yes we let the implementation directly pass the entire block instead of value-by-value.
 * 
 * Known implementation classes:
 *    - FastBufferedDataInputStream
 *    - CacheDataInput
 *    
 */
public interface MatrixBlockDataInput 
{	
	/**
	 * Reads the double array from the data input into the given dense block
	 * and returns the number of non-zeros. 
	 * 
	 * @param len
	 * @param varr
	 * @return
	 * @throws IOException
	 */
	public long readDoubleArray(int len, double[] varr) 
		throws IOException;
	
	/**
	 * Reads the sparse rows array from the data input into a sparse block
	 * and returns the number of non-zeros.
	 * 
	 * @param rlen
	 * @param rows
	 * @throws IOException
	 */
	public long readSparseRows(int rlen, SparseRow[] rows) 
		throws IOException;
}

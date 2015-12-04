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


package org.apache.sysml.runtime.matrix.data;

import java.io.IOException;

/**
 * Any data output that is intended to support fast serialization / write
 * of entire blocks should implement this interface. On write of a matrix block
 * we check if the output stream is an implementation of this interface, if
 * yes we directly pass the entire block instead of value-by-value.
 * 
 * Known implementation classes:
 *    - CacheDataOutput (cache serialization into in-memory write buffer)
 *    - FastBufferedDataOutputStream (cache eviction to local file system)
 * 
 */
public interface MatrixBlockDataOutput 
{
	/**
	 * Writes the double array of a dense block to the data output. 
	 * 
	 * @param len
	 * @param varr
	 * @throws IOException
	 */
	public void writeDoubleArray(int len, double[] varr) 
		throws IOException;
	
	/**
	 * Writes the sparse rows array of a sparse block to the data output.
	 * 
	 * @param rlen
	 * @param rows
	 * @throws IOException
	 */
	public void writeSparseRows(int rlen, SparseRow[] rows) 
		throws IOException;
}

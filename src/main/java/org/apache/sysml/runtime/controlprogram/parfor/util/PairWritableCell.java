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

package org.apache.sysml.runtime.controlprogram.parfor.util;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.Writable;

import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

/**
 * Custom writable for a pair of matrix indexes and matrix cell
 * as required for binarycell in remote data partitioning.
 * 
 */
public class PairWritableCell implements Writable, Serializable
{

	private static final long serialVersionUID = 6835655576808999370L;
	
	public MatrixIndexes indexes;
	public MatrixCell cell;
	
	@Override
	public void readFields(DataInput in) throws IOException 
	{
		indexes = new MatrixIndexes();
		indexes.readFields(in);
		
		cell = new MatrixCell();
		cell.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException 
	{
		indexes.write(out);
		cell.write(out);
	}
}

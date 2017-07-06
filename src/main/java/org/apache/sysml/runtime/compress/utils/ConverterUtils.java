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

package org.apache.sysml.runtime.compress.utils;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.sysml.runtime.compress.ColGroup;
import org.apache.sysml.runtime.compress.ColGroupOLE;
import org.apache.sysml.runtime.compress.ColGroupRLE;
import org.apache.sysml.runtime.compress.ColGroupUncompressed;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;

public class ConverterUtils 
{
	/**
	 * Copy col group instance with deep copy of column indices but
	 * shallow copy of actual contents;
	 * 
	 * @param group column group
	 * @return column group (deep copy of indices but shallow copy of contents)
	 */
	public static ColGroup copyColGroup(ColGroup group)
	{
		ColGroup ret = null;
		
		//deep copy col indices
		int[] colIndices = Arrays.copyOf(group.getColIndices(), group.getNumCols());
		
		//create copy of column group
		if( group instanceof ColGroupUncompressed ) {
			ColGroupUncompressed in = (ColGroupUncompressed)group;
			ret = new ColGroupUncompressed(colIndices, in.getNumRows(), in.getData());
		}
		else if( group instanceof ColGroupRLE ) {
			ColGroupRLE in = (ColGroupRLE)group;
			ret = new ColGroupRLE(colIndices, in.getNumRows(), in.hasZeros(), 
					in.getValues(), in.getBitmaps(), in.getBitmapOffsets());
		}
		else if( group instanceof ColGroupOLE ) {
			ColGroupOLE in = (ColGroupOLE) group;
			ret = new ColGroupOLE(colIndices, in.getNumRows(), in.hasZeros(),
					in.getValues(), in.getBitmaps(), in.getBitmapOffsets());
		}
		
		return ret;
	}

	public static double[] getDenseVector( MatrixBlock vector ) {
		return DataConverter.convertToDoubleVector(vector, false);
	}

	public static MatrixBlock getUncompressedColBlock( ColGroup group )
	{
		MatrixBlock ret = null;
		if( group instanceof ColGroupUncompressed ) {
			ret = ((ColGroupUncompressed) group).getData();
		}
		else {
			ArrayList<ColGroup> tmpGroup = new ArrayList<ColGroup>(Arrays.asList(group));
			ColGroupUncompressed decompressedCols = new ColGroupUncompressed(tmpGroup);
			ret = decompressedCols.getData();
		}
		
		return ret;
	}
}

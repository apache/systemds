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

package org.apache.sysml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import org.apache.sysml.runtime.matrix.CSVReblockMR;
import org.apache.sysml.runtime.matrix.CSVReblockMR.OffsetCount;

public class CSVAssignRowIDReducer extends MapReduceBase implements Reducer<ByteWritable, OffsetCount, ByteWritable, OffsetCount>
{
	
	
	private ArrayList<OffsetCount> list = new ArrayList<OffsetCount>();
	
	@Override
	@SuppressWarnings("unchecked")
	public void reduce(ByteWritable key, Iterator<OffsetCount> values,
			OutputCollector<ByteWritable, OffsetCount> out, Reporter report)
			throws IOException 
	{	
		//need to sort the values by filename and fileoffset
		while(values.hasNext())
			list.add(new OffsetCount(values.next()));
		Collections.sort(list);
		
		long lineOffset=0;
		for(OffsetCount oc: list)
		{
			long count=oc.count;
			oc.count=lineOffset;
			out.collect(key, oc);
			lineOffset+=count;
		}
		report.incrCounter(CSVReblockMR.NUM_ROWS_IN_MATRIX, key.toString(), lineOffset);
		list.clear();
	}

}

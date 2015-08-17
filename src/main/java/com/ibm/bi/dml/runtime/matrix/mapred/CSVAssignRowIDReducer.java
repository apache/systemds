/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.mapred;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.apache.hadoop.io.ByteWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.ibm.bi.dml.runtime.matrix.CSVReblockMR;
import com.ibm.bi.dml.runtime.matrix.CSVReblockMR.OffsetCount;

public class CSVAssignRowIDReducer extends MapReduceBase implements Reducer<ByteWritable, OffsetCount, ByteWritable, OffsetCount>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
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

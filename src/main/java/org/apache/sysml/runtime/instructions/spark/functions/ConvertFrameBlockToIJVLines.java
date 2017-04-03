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
package org.apache.sysml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.Iterator;

import org.apache.spark.api.java.function.FlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.FrameBlock;

public class ConvertFrameBlockToIJVLines implements FlatMapFunction<Tuple2<Long,FrameBlock>, String> 
{
	private static final long serialVersionUID = 1803516615963340115L;

	@Override
	public Iterator<String> call(Tuple2<Long, FrameBlock> kv) 
		throws Exception 
	{
		long rowoffset = kv._1;
		FrameBlock block = kv._2;
		
		ArrayList<String> cells = new ArrayList<String>();
		
		//write frame meta data
		if( rowoffset == 1 ) {
			for( int j=0; j<block.getNumColumns(); j++ )
				if( !block.isColumnMetadataDefault(j) ) {
					cells.add("-1 " + (j+1) + " " + block.getColumnMetadata(j).getNumDistinct());
					cells.add("-2 " + (j+1) + " " + block.getColumnMetadata(j).getMvValue());
				}
		}
		
		//convert frame block to list of ijv cell triples
		StringBuilder sb = new StringBuilder();
		Iterator<String[]> iter = block.getStringRowIterator();
		for( int i=0; iter.hasNext(); i++ ) { //for all rows
			String rowIndex = Long.toString(rowoffset + i);
			String[] row = iter.next();
			for( int j=0; j<row.length; j++ ) {
				if( row[j] != null ) {
					sb.append( rowIndex );
					sb.append(' ');
					sb.append( j+1 );
					sb.append(' ');
					sb.append( row[j] );
					cells.add( sb.toString() );
					sb.setLength(0); 
				}
			}
		}
		
		return cells.iterator();
	}
}

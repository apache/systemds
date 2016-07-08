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

package org.apache.sysml.runtime.transform.encode;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.transform.DistinctValue;
import org.apache.sysml.runtime.transform.TfUtils;

/**
 * Simple composite encoder that applies a list of encoders 
 * in specified order. By implementing the default encoder API
 * it can be used as a drop-in replacement for any other encoder. 
 * 
 */
public class EncoderComposite extends Encoder
{
	private static final long serialVersionUID = -8473768154646831882L;
	
	private List<Encoder> _encoders = null;
	private FrameBlock _meta = null;
	
	protected EncoderComposite(List<Encoder> encoders) {
		super(null, -1);
		_encoders = encoders;
	}
	
	protected EncoderComposite(Encoder[] encoders) {
		super(null, -1);
		_encoders = Arrays.asList(encoders);
	}
	
	@Override
	public int getNumCols() {
		int clen = 0;
		for( Encoder encoder : _encoders )
			clen = Math.max(clen, encoder.getNumCols());
		return clen;
	}

	public List<Encoder> getEncoders() {
		return _encoders;
	}
	
	@Override
	public MatrixBlock encode(FrameBlock in, MatrixBlock out) {
		//build meta data first (for all encoders)
		for( Encoder encoder : _encoders )
			encoder.build(in);
		
		//propagate meta data 
		_meta = new FrameBlock(in.getNumColumns(), ValueType.STRING);
		for( Encoder encoder : _encoders )
			_meta = encoder.getMetaData(_meta);
		for( Encoder encoder : _encoders )
			encoder.initMetaData(_meta);
		
		//apply meta data
		for( Encoder encoder : _encoders )
			out = encoder.apply(in, out);
			
		return out;
	}

	@Override
	public void build(FrameBlock in) {
		for( Encoder encoder : _encoders )
			encoder.build(in);
	}


	@Override
	public String[] apply(String[] in) {
		for( Encoder encoder : _encoders )
			encoder.apply(in);
		return in;
	}
	
	@Override 
	public MatrixBlock apply(FrameBlock in, MatrixBlock out) {
		for( Encoder encoder : _encoders )
			out = encoder.apply(in, out);
		return out;
	}
	
	@Override
	public FrameBlock getMetaData(FrameBlock out) {
		if( _meta != null )
			return _meta;
		for( Encoder encoder : _encoders )
			encoder.getMetaData(out);
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock out) {
		for( Encoder encoder : _encoders )
			encoder.initMetaData(out);
	}

	@Override
	public void mapOutputTransformationMetadata(OutputCollector<IntWritable, DistinctValue> out, int taskID, TfUtils agents) throws IOException {
		throw new RuntimeException("File-based api not supported.");
	}

	@Override
	public void mergeAndOutputTransformationMetadata(Iterator<DistinctValue> values, String outputDir, int colID, FileSystem fs, TfUtils agents) throws IOException {
		throw new RuntimeException("File-based api not supported.");	
	}

	@Override
	public void loadTxMtd(JobConf job, FileSystem fs, Path txMtdDir, TfUtils agents) throws IOException {
		throw new RuntimeException("File-based api not supported.");
	}
}

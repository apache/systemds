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

package org.apache.sysml.runtime.transform.decode;

import java.util.Arrays;
import java.util.List;

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Simple composite decoder that applies a list of decoders 
 * in specified order. By implementing the default decoder API
 * it can be used as a drop-in replacement for any other decoder. 
 * 
 */
public class DecoderComposite extends Decoder
{
	private static final long serialVersionUID = 5790600547144743716L;
	
	private List<Decoder> _decoders = null;
	
	protected DecoderComposite(List<ValueType> schema, List<Decoder> decoders) {
		super(schema, null);
		_decoders = decoders;
	}
	
	protected DecoderComposite(List<ValueType> schema, Decoder[] decoders) {
		super(schema, null);
		_decoders = Arrays.asList(decoders);
	}

	@Override
	public FrameBlock decode(MatrixBlock in, FrameBlock out) {
		for( Decoder decoder : _decoders )
			out = decoder.decode(in, out);	
		return out;
	}
	
	@Override
	public void initMetaData(FrameBlock meta) {
		for( Decoder decoder : _decoders )
			decoder.initMetaData(meta);	
	}
}

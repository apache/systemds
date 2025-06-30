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

package org.apache.sysds.runtime.transform.decode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.wink.json4j.JSONObject;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import static org.apache.sysds.runtime.util.CollectionUtils.except;
import static org.apache.sysds.runtime.util.CollectionUtils.unionDistinct;

public class DecoderFactory 
{
	public enum DecoderType {
		Bin,
		Dummycode, 
		PassThrough,
		Recode,
	}
	
	public static Decoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta) {
		return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), -1, -1);
	}
	
	public static Decoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int clen) {
		return createDecoder(spec, colnames, schema, meta, clen, -1, -1);
	}

	public static Decoder createDecoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta, int minCol,
		int maxCol) {
		return createDecoder(spec, colnames, schema, meta, meta.getNumColumns(), minCol, maxCol);
	}

	public static Decoder createDecoder(String spec, String[] colnames, ValueType[] schema,
		FrameBlock meta, int clen, int minCol, int maxCol)
	{
		Decoder decoder = null;
		
		try {
			//parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			
			//create decoders 'bin', 'recode', 'hash', 'dummy', and 'pass-through'
			List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
			List<Integer> rcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.RECODE.toString(), minCol, maxCol)));
			List<Integer> hcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.HASH.toString(), minCol, maxCol)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils.toObject(
					TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.DUMMYCODE.toString(), minCol, maxCol)));
			// only specially treat the columns with both recode and dictionary
			rcIDs = unionDistinct(rcIDs, dcIDs);
			// remove hash recoded. // todo potentially wrong and remove?
			rcIDs = except(rcIDs, hcIDs);

			int len = dcIDs.isEmpty() ? Math.min(meta.getNumColumns(), clen) : meta.getNumColumns();

			// set the remaining columns to passthrough.
			List<Integer> ptIDs = UtilFunctions.getSeqList(1, len, 1);
			// except recoded columns
			ptIDs = except(ptIDs, rcIDs);
			// binned columns
			ptIDs = except(ptIDs, binIDs);
			// hashed columns
			ptIDs = except(ptIDs, hcIDs); // remove hashed columns

			//create default schema if unspecified (with double columns for pass-through)
			if( schema == null ) {
				schema = UtilFunctions.nCopies(len, ValueType.STRING);
				for( Integer col : ptIDs )
					schema[col-1] = ValueType.FP64;
			}

			// collect all the decoders in one list.
			List<Decoder> ldecoders = new ArrayList<>();
			
			if( !binIDs.isEmpty() ) {
				ldecoders.add(new DecoderBin(schema, 
					ArrayUtils.toPrimitive(binIDs.toArray(new Integer[0])),
					ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0]))));
			}
			if( !dcIDs.isEmpty() ) {
				ldecoders.add(new DecoderDummycode(schema, 
					ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0]))));
			}
			if( !rcIDs.isEmpty() ) {
				// todo figure out if we need to handle rc columns with regards to dictionary offsets.
				ldecoders.add(new DecoderRecode(schema, !dcIDs.isEmpty(),
					ArrayUtils.toPrimitive(rcIDs.toArray(new Integer[0]))));
			}
			if( !ptIDs.isEmpty() ) {
				ldecoders.add(new DecoderPassThrough(schema, 
					ArrayUtils.toPrimitive(ptIDs.toArray(new Integer[0])),
					ArrayUtils.toPrimitive(dcIDs.toArray(new Integer[0]))));
			}
			
			//create composite decoder of all created decoders
			//and initialize with given meta data (recode, dummy, bin)
			decoder = new DecoderComposite(schema, ldecoders);
			decoder.setColnames(colnames);
			decoder.initMetaData(meta);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return decoder;
	}
	
	public static int getDecoderType(Decoder decoder) {
		if( decoder instanceof DecoderDummycode )
			return DecoderType.Dummycode.ordinal();
		else if( decoder instanceof DecoderRecode )
			return DecoderType.Recode.ordinal();
		else if( decoder instanceof DecoderPassThrough )
			return DecoderType.PassThrough.ordinal();
		throw new DMLRuntimeException("Unsupported decoder type: "
			+ decoder.getClass().getCanonicalName());
	}
	
	public static Decoder createInstance(int type) {
		DecoderType dtype = DecoderType.values()[type];
		
		// create instance
		switch(dtype) {
			case Dummycode:   return new DecoderDummycode(null, null);
			case PassThrough: return new DecoderPassThrough(null, null, null);
			case Recode:      return new DecoderRecode(null, false, null);
			default:
				throw new DMLRuntimeException("Unsupported Encoder Type used:  " + dtype);
		}
	}
}

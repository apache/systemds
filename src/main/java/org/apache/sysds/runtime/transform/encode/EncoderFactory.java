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

package org.apache.sysds.runtime.transform.encode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.TfUtils.TfMethod;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder.EncoderType;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.TransformStatistics;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONObject;

import static org.apache.sysds.runtime.util.CollectionUtils.except;
import static org.apache.sysds.runtime.util.CollectionUtils.intersect;
import static org.apache.sysds.runtime.util.CollectionUtils.unionDistinct;
import static org.apache.sysds.runtime.util.CollectionUtils.naryUnionDistinct;

public interface EncoderFactory {
	final static Log LOG = LogFactory.getLog(EncoderFactory.class.getName());

	public static MultiColumnEncoder createEncoder(String spec, int clen) {
		return createEncoder(spec, null, clen, null, null, -1, -1);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta) {
		return createEncoder(spec, colnames, clen, meta, null, -1, -1);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta,
		int minCol, int maxCol) {
		return createEncoder(spec, colnames, clen, meta, null, minCol, maxCol);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema, int clen,
		FrameBlock meta) {
		return createEncoder(spec, colnames, clen, meta);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema, int clen,
												   FrameBlock meta, MatrixBlock embeddings) {
		ValueType[] lschema = (schema == null) ? UtilFunctions.nCopies(clen, ValueType.STRING) : schema;
		return createEncoder(spec, colnames, lschema, meta, embeddings);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema,
		FrameBlock meta) {
		return createEncoder(spec, colnames, schema, meta, -1, -1);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema, FrameBlock meta,
		int minCol, int maxCol) {
		return createEncoder(spec, colnames, schema.length, meta, null, minCol, maxCol);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta, MatrixBlock embeddings) {
		return createEncoder(spec, colnames, UtilFunctions.nCopies(clen, ValueType.STRING), meta, embeddings);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, ValueType[] schema,
												   FrameBlock meta, MatrixBlock embeddings) {
		return createEncoder(spec, colnames, schema.length, meta, embeddings, -1, -1);
	}

	public static MultiColumnEncoder createEncoder(String spec, String[] colnames, int clen, FrameBlock meta,
		MatrixBlock embeddings, int minCol, int maxCol) {


		MultiColumnEncoder encoder;
		// int clen = schema.length;

		try {
			// parse transform specification
			JSONObject jSpec = new JSONObject(spec);
			List<ColumnEncoderComposite> lencoders = new ArrayList<>();
			HashMap<Integer, List<ColumnEncoder>> colEncoders = new HashMap<>();
			boolean ids = jSpec.containsKey("ids") && jSpec.getBoolean("ids");
			TfMetaUtils.checkValidEncoders(jSpec);

			// prepare basic id lists (recode, feature hash, dummycode, pass-through)
			List<Integer> rcIDs = Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.RECODE.toString(), minCol, maxCol)));
			List<Integer> haIDs = Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.HASH.toString(), minCol, maxCol)));
			List<Integer> dcIDs = Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.DUMMYCODE.toString(), minCol, maxCol)));
			List<Integer> binIDs = TfMetaUtils.parseBinningColIDs(jSpec, colnames, minCol, maxCol);
			List<Integer> weIDs = Arrays.asList(ArrayUtils
					.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.WORD_EMBEDDING.toString(), minCol, maxCol)));
			List<Integer> bowIDs = Arrays.asList(ArrayUtils
					.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.BAG_OF_WORDS.toString(), minCol, maxCol)));
			List<Integer> ragIDs = Arrays.asList(ArrayUtils
					.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.RAGGED.toString(), minCol, maxCol)));

			// NOTE: any dummycode column requires recode as preparation, unless the dummycode
			// column follows binning or feature hashing
			rcIDs = unionDistinct(rcIDs, except(except(dcIDs, binIDs), haIDs));
			// Error out if the first level encoders have overlaps
			if (intersect(rcIDs, binIDs, haIDs, weIDs, bowIDs, ragIDs))
				throw new DMLRuntimeException("More than one encoders (recode, binning, hashing, word_embedding, bag_of_words, ragIDs) on one column is not allowed:\n" + spec);

			List<Integer> ptIDs = except(UtilFunctions.getSeqList(1, clen, 1), naryUnionDistinct(rcIDs, haIDs, binIDs, weIDs, bowIDs));
			List<Integer> oIDs = new ArrayList<>(Arrays.asList(ArrayUtils
				.toObject(TfMetaUtils.parseJsonIDList(jSpec, colnames, TfMethod.OMIT.toString(), minCol, maxCol))));
			List<Integer> mvIDs = new ArrayList<>(Arrays.asList(ArrayUtils.toObject(
				TfMetaUtils.parseJsonObjectIDList(jSpec, colnames, TfMethod.IMPUTE.toString(), minCol, maxCol))));
			List<Integer> udfIDs = TfMetaUtils.parseUDFColIDs(jSpec, colnames, minCol, maxCol);

			// robustness for transformencode specs w/ non-existing columns (so far, endless loops)
			rcIDs.removeIf(i -> i > clen);
			ptIDs.removeIf(i -> i > clen);
			oIDs.removeIf(i -> i > clen);
			mvIDs.removeIf(i -> i > clen);
			udfIDs.removeIf(i -> i > clen);
			binIDs.removeIf(i -> i > clen);
			weIDs.removeIf(i -> i > clen);
			bowIDs.removeIf(i -> i > clen);
			
			// create individual encoders
			if(!rcIDs.isEmpty())
				for(Integer id : rcIDs)
					addEncoderToMap(new ColumnEncoderRecode(id), colEncoders);
			if(!haIDs.isEmpty())
				for(Integer id : haIDs)
					addEncoderToMap(new ColumnEncoderFeatureHash(id, TfMetaUtils.getK(jSpec)), colEncoders);
			if(!ptIDs.isEmpty())
				for(Integer id : ptIDs)
					addEncoderToMap(new ColumnEncoderPassThrough(id), colEncoders);
			if(!weIDs.isEmpty())
				for(Integer id : weIDs)
					addEncoderToMap(new ColumnEncoderWordEmbedding(id), colEncoders);
			if(!ragIDs.isEmpty())
    			for(Integer id : ragIDs)
        			addEncoderToMap(new ColumnEncoderRagged(id), colEncoders);
			if(!bowIDs.isEmpty())
				for(Integer id : bowIDs)
					addEncoderToMap(new ColumnEncoderBagOfWords(id), colEncoders);
			if(!binIDs.isEmpty())
				for(Object o : (JSONArray) jSpec.get(TfMethod.BIN.toString())) {
					JSONObject colspec = (JSONObject) o;
					int numBins = colspec.containsKey("numbins") ? colspec.getInt("numbins") : 1;
					int id = TfMetaUtils.parseJsonObjectID(colspec, colnames, minCol, maxCol, ids);
					if(id <= 0)
						continue;
					String method = colspec.get("method").toString().toUpperCase();
					ColumnEncoderBin.BinMethod binMethod;
					if ("EQUI-WIDTH".equals(method))
						binMethod = ColumnEncoderBin.BinMethod.EQUI_WIDTH;
					else if ("EQUI-HEIGHT".equals(method))
						binMethod = ColumnEncoderBin.BinMethod.EQUI_HEIGHT;
					else if ("EQUI-HEIGHT-APPROX".equals(method))
						binMethod = ColumnEncoderBin.BinMethod.EQUI_HEIGHT_APPROX;
					else
						throw new DMLRuntimeException("Unsupported binning method: " + method);
					ColumnEncoderBin bin = new ColumnEncoderBin(id, numBins, binMethod);
					addEncoderToMap(bin, colEncoders);
				}
			if(!dcIDs.isEmpty())
				for(Integer id : dcIDs)
					addEncoderToMap(new ColumnEncoderDummycode(id), colEncoders);
			if(!udfIDs.isEmpty()) {
				String name = jSpec.getJSONObject("udf").getString("name");
				for(Integer id : udfIDs)
					addEncoderToMap(new ColumnEncoderUDF(id, name), colEncoders);
			}
			
			// create composite decoder of all created encoders
			for(Entry<Integer, List<ColumnEncoder>> listEntry : colEncoders.entrySet()) {
				if(DMLScript.STATISTICS)
					TransformStatistics.incEncoderCount(listEntry.getValue().size());
				lencoders.add(new ColumnEncoderComposite(listEntry.getValue()));
			}
			encoder = new MultiColumnEncoder(lencoders);
			if(!oIDs.isEmpty()) {
				encoder.addReplaceLegacyEncoder(new EncoderOmit(jSpec, colnames, clen, minCol, maxCol));
				if(DMLScript.STATISTICS)
					TransformStatistics.incEncoderCount(1);
			}
			if(!mvIDs.isEmpty()) {
				EncoderMVImpute ma = new EncoderMVImpute(jSpec, colnames, clen, minCol, maxCol);
				ma.initRecodeIDList(rcIDs);
				encoder.addReplaceLegacyEncoder(ma);
				if(DMLScript.STATISTICS)
					TransformStatistics.incEncoderCount(1);
			}

			// initialize meta data w/ robustness for superset of cols
			if(meta != null) {
				String[] colnames2 = meta.getColumnNames();

				if(!TfMetaUtils.isIDSpec(jSpec) && colnames != null && colnames2 != null &&
					!Objects.deepEquals(colnames, colnames2)) {
					HashMap<String, Integer> colPos = getColumnPositions(colnames2);
					// create temporary meta frame block w/ shallow column copy
					FrameBlock meta2 = new FrameBlock(meta.getSchema(), colnames2);

					for(int i = 0; i < colnames.length; i++) {
						if(!colPos.containsKey(colnames[i])) {
							throw new DMLRuntimeException("Column name not found in meta data: " + colnames[i]
								+ " (meta: " + Arrays.toString(colnames2) + ")");
						}
						int pos = colPos.get(colnames[i]);
						meta2.setColumn(i, meta.getColumn(pos));
						meta2.setColumnMetadata(i, meta.getColumnMetadata(pos));
					}
					meta = meta2;
				}
				encoder.initMetaData(meta);
			}
			//initialize embeddings matrix block in the encoders in case word embedding transform is used
			if(!weIDs.isEmpty())
				encoder.initEmbeddings(embeddings);
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		return encoder;
	}

	private static void addEncoderToMap(ColumnEncoder encoder, HashMap<Integer, List<ColumnEncoder>> map) {
		if(!map.containsKey(encoder._colID)) {
			map.put(encoder._colID, new ArrayList<>());
		}
		map.get(encoder._colID).add(encoder);
	}

	public static int getEncoderType(ColumnEncoder columnEncoder) {
		//TODO replace with columnEncoder.getType().ordinal
		//(which requires a cleanup of all type handling)
		if(columnEncoder instanceof ColumnEncoderBin)
			return EncoderType.Bin.ordinal();
		else if(columnEncoder instanceof ColumnEncoderDummycode)
			return EncoderType.Dummycode.ordinal();
		else if(columnEncoder instanceof ColumnEncoderFeatureHash)
			return EncoderType.FeatureHash.ordinal();
		else if(columnEncoder instanceof ColumnEncoderPassThrough)
			return EncoderType.PassThrough.ordinal();
		else if(columnEncoder instanceof ColumnEncoderRecode)
			return EncoderType.Recode.ordinal();
		else if(columnEncoder instanceof ColumnEncoderWordEmbedding)
			return EncoderType.WordEmbedding.ordinal();
		else if(columnEncoder instanceof ColumnEncoderBagOfWords)
			return EncoderType.BagOfWords.ordinal();
		throw new DMLRuntimeException("Unsupported encoder type: " + columnEncoder.getClass().getCanonicalName());
	}

	public static ColumnEncoder createInstance(int type) {
		EncoderType etype = EncoderType.values()[type];
		switch(etype) {
			case Bin:
				return new ColumnEncoderBin();
			case Dummycode:
				return new ColumnEncoderDummycode();
			case FeatureHash:
				return new ColumnEncoderFeatureHash();
			case PassThrough:
				return new ColumnEncoderPassThrough();
			case Recode:
				return new ColumnEncoderRecode();
			case WordEmbedding:
				return new ColumnEncoderWordEmbedding();
			case BagOfWords:
				return new ColumnEncoderBagOfWords();
			case Ragged:
    			return new ColumnEncoderRagged();
			default:
				throw new DMLRuntimeException("Unsupported encoder type: " + etype);
		}
	}

	private static HashMap<String, Integer> getColumnPositions(String[] colnames) {
		HashMap<String, Integer> ret = new HashMap<>();
		for(int i = 0; i < colnames.length; i++)
			ret.put(colnames[i], i);
		return ret;
	}
}

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

package org.apache.sysds.runtime.instructions.spark;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.AccumulatorV2;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.instructions.spark.ParameterizedBuiltinSPInstruction.RDDTransformApplyFunction;
import org.apache.sysds.runtime.instructions.spark.ParameterizedBuiltinSPInstruction.RDDTransformApplyOffsetFunction;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysds.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBagOfWords;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBin;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderComposite;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderMVImpute;
import org.apache.sysds.runtime.transform.encode.EncoderMVImpute.MVMethod;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.transform.meta.TfMetaUtils;
import org.apache.sysds.runtime.transform.meta.TfOffsetMap;

import scala.Tuple2;

public class MultiReturnParameterizedBuiltinSPInstruction extends ComputationSPInstruction {
	protected ArrayList<CPOperand> _outputs;
	protected final boolean _metaReturn;

	private MultiReturnParameterizedBuiltinSPInstruction(Operator op, CPOperand input1, CPOperand input2,
		boolean metaReturn, ArrayList<CPOperand> outputs, String opcode, String istr) {
		super(SPType.MultiReturnBuiltin, op, input1, input2, outputs.get(0), opcode, istr);
		_metaReturn = metaReturn;
		_outputs = outputs;
	}

	public static MultiReturnParameterizedBuiltinSPInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		ArrayList<CPOperand> outputs = new ArrayList<>();
		String opcode = parts[0];

		if(opcode.equalsIgnoreCase("transformencode")) {
			// one input and two outputs
			CPOperand in1 = new CPOperand(parts[1]);
			CPOperand in2 = new CPOperand(parts[2]);
			int pos = 3;
			boolean metaReturn = true;
			if( parts.length == 6 ) //no need for meta data
				metaReturn = new CPOperand(parts[pos++]).getLiteral().getBooleanValue();
			outputs.add(new CPOperand(parts[pos], ValueType.FP64, DataType.MATRIX));
			outputs.add(new CPOperand(parts[pos+1], ValueType.STRING, DataType.FRAME));
			return new MultiReturnParameterizedBuiltinSPInstruction(null, in1, in2, metaReturn, outputs, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Invalid opcode in MultiReturnBuiltin instruction: " + opcode);
		}
	}

	@Override
	@SuppressWarnings("unchecked")
	public void processInstruction(ExecutionContext ec) {
		SparkExecutionContext sec = (SparkExecutionContext) ec;

		try {
			// get input RDD and meta data
			FrameObject fo = sec.getFrameObject(input1.getName());
			FrameObject fometa = sec.getFrameObject(_outputs.get(1).getName());
			JavaPairRDD<Long, FrameBlock> in = (JavaPairRDD<Long, FrameBlock>) 
				sec.getRDDHandleForFrameObject(fo, FileFormat.BINARY);
			String spec = ec.getScalarInput(input2).getStringValue();
			DataCharacteristics mcIn = sec.getDataCharacteristics(input1.getName());
			DataCharacteristics mcOut = sec.getDataCharacteristics(output.getName());
			String[] colnames = !TfMetaUtils.isIDSpec(spec) ? in.lookup(1L).get(0).getColumnNames() : null;

			// step 1: build transform meta data
			MultiColumnEncoder encoderBuild = EncoderFactory
				.createEncoder(spec, colnames, fo.getSchema(), (int) fo.getNumColumns(), null);

			MaxLongAccumulator accMax = registerMaxLongAccumulator(sec.getSparkContext());
			JavaRDD<String> rcMaps = in.mapPartitionsToPair(new TransformEncodeBuildFunction(encoderBuild)).distinct()
				.groupByKey().flatMap(new TransformEncodeGroupFunction(encoderBuild, accMax));
			if(containsMVImputeEncoder(encoderBuild)) {
				EncoderMVImpute mva = encoderBuild.getLegacyEncoder(EncoderMVImpute.class);
				rcMaps = rcMaps.union(in.mapPartitionsToPair(new TransformEncodeBuild2Function(mva)).groupByKey()
					.flatMap(new TransformEncodeGroup2Function(mva)));
			}
			rcMaps.saveAsTextFile(fometa.getFileName()); // trigger eval

			// consolidate meta data frame (reuse multi-threaded reader, special handling missing values)
			FrameReader reader = FrameReaderFactory.createFrameReader(FileFormat.TEXT);
			FrameBlock meta = reader.readFrameFromHDFS(fometa.getFileName(), accMax.value(), fo.getNumColumns());
			meta.recomputeColumnCardinality(); // recompute num distinct items per column
			meta.setColumnNames((colnames != null) ? colnames : meta.getColumnNames());
			meta.mapInplace(TfUtils::desanitizeSpaces); // due to format TEXT

			// step 2: transform apply (similar to spark transformapply)
			// compute omit offset map for block shifts
			TfOffsetMap omap = null;
			if(TfMetaUtils.containsOmitSpec(spec, colnames)) {
				omap = new TfOffsetMap(SparkUtils
					.toIndexedLong(in.mapToPair(new RDDTransformApplyOffsetFunction(spec, colnames)).collect()));
			}

			// create encoder broadcast (avoiding replication per task)
			MultiColumnEncoder encoder = EncoderFactory
				.createEncoder(spec, colnames, fo.getSchema(), (int) fo.getNumColumns(), meta);
			mcOut.setDimension(mcIn.getRows() - ((omap != null) ? omap.getNumRmRows() : 0), encoder.getNumOutCols());
			Broadcast<MultiColumnEncoder> bmeta = sec.getSparkContext().broadcast(encoder);
			Broadcast<TfOffsetMap> bomap = (omap != null) ? sec.getSparkContext().broadcast(omap) : null;

			// execute transform apply
			JavaPairRDD<Long, FrameBlock> tmp = in.mapToPair(new RDDTransformApplyFunction(bmeta, bomap));
			JavaPairRDD<MatrixIndexes, MatrixBlock> out = FrameRDDConverterUtils
				.binaryBlockToMatrixBlock(tmp, mcOut, mcOut).cache(); // best effort cache as reblock not at hop level

			// set output and maintain lineage/output characteristics
			sec.setRDDHandleForVariable(_outputs.get(0).getName(), out);
			sec.addLineageRDD(_outputs.get(0).getName(), input1.getName());
			if( _metaReturn )
				sec.setFrameOutput(_outputs.get(1).getName(), meta);
			else
				sec.setFrameOutput(_outputs.get(1).getName(), new FrameBlock());
		}
		catch(IOException ex) {
			throw new RuntimeException(ex);
		}
	}

	private static boolean containsMVImputeEncoder(Encoder encoder) {
		if(encoder instanceof ColumnEncoderComposite) {
			throw new DMLRuntimeException("CompositeEncoders cannot contain legacy encoder MVImpute");
		}
		else if(encoder instanceof MultiColumnEncoder) {
			return ((MultiColumnEncoder) encoder).hasLegacyEncoder(EncoderMVImpute.class);
		}
		return false;
	}

	private static MaxLongAccumulator registerMaxLongAccumulator(JavaSparkContext sc) {
		MaxLongAccumulator acc = new MaxLongAccumulator(Long.MIN_VALUE);
		sc.sc().register(acc, "max");
		return acc;
	}

	public List<CPOperand> getOutputs() {
		return _outputs;
	}

	private static class MaxLongAccumulator extends AccumulatorV2<Long, Long> {
		private static final long serialVersionUID = -3739727823287550826L;

		private long _value = Long.MIN_VALUE;

		public MaxLongAccumulator(long value) {
			_value = value;
		}

		@Override
		public void add(Long arg0) {
			_value = Math.max(_value, arg0);
		}

		@Override
		public AccumulatorV2<Long, Long> copy() {
			return new MaxLongAccumulator(_value);
		}

		@Override
		public boolean isZero() {
			return _value == Long.MIN_VALUE;
		}

		@Override
		public void merge(AccumulatorV2<Long, Long> arg0) {
			_value = Math.max(_value, arg0.value());
		}

		@Override
		public void reset() {
			_value = Long.MIN_VALUE;
		}

		@Override
		public Long value() {
			return _value;
		}
	}

	/**
	 * This function pre-aggregates distinct values of recoded columns per partition (part of distributed recode map
	 * construction, used for recoding, binning and dummy coding). We operate directly over schema-specific objects to
	 * avoid unnecessary string conversion, as well as reduce memory overhead and shuffle.
	 */
	public static class TransformEncodeBuildFunction
		implements PairFlatMapFunction<Iterator<Tuple2<Long, FrameBlock>>, Integer, Object> {
		private static final long serialVersionUID = 6336375833412029279L;

		private MultiColumnEncoder _encoder;

		public TransformEncodeBuildFunction(MultiColumnEncoder encoder) {
			_encoder = encoder;
		}

		@Override
		public Iterator<Tuple2<Integer, Object>> call(Iterator<Tuple2<Long, FrameBlock>> iter) throws Exception {
			// build meta data (e.g., recoding recode maps and binning min/max)
			_encoder.prepareBuildPartial();
			while(iter.hasNext())
				_encoder.buildPartial(iter.next()._2());

			// encoder-specific outputs
			List<ColumnEncoderRecode> raEncoders = _encoder.getColumnEncoders(ColumnEncoderRecode.class);
			List<ColumnEncoderBin> baEncoders = _encoder.getColumnEncoders(ColumnEncoderBin.class);
			List<ColumnEncoderBagOfWords> bowEncoders = _encoder.getColumnEncoders(ColumnEncoderBagOfWords.class);
			ArrayList<Tuple2<Integer, Object>> ret = new ArrayList<>();

			// output recode maps as columnID - token pairs
			if(!raEncoders.isEmpty()) {
				// TODO check in debbuger if correct
				Map<Integer, HashSet<Object>> tmp = raEncoders.stream()
					.collect(Collectors.toMap(ColumnEncoder::getColID, ColumnEncoderRecode::getCPRecodeMapsPartial));
				for(Entry<Integer, HashSet<Object>> e1 : tmp.entrySet())
					for(Object token : e1.getValue())
						ret.add(new Tuple2<>(e1.getKey(), token));
				raEncoders.forEach(columnEncoderRecode -> columnEncoderRecode.getCPRecodeMapsPartial().clear());
			}

			if(!bowEncoders.isEmpty()){
				for (ColumnEncoderBagOfWords bowEnc : bowEncoders)
					for (Object token : bowEnc.getPartialTokenDictionary())
						ret.add(new Tuple2<>(bowEnc.getColID(), token));
				bowEncoders.forEach(enc -> enc.getPartialTokenDictionary().clear());
			}

			// output binning column min/max as columnID - min/max pairs
			if(!baEncoders.isEmpty()) {
				int[] colIDs = _encoder.getFromAllIntArray(ColumnEncoderBin.class, ColumnEncoder::getColID);
				double[] colMins = _encoder.getFromAllDoubleArray(ColumnEncoderBin.class, ColumnEncoderBin::getColMins);
				double[] colMaxs = _encoder.getFromAllDoubleArray(ColumnEncoderBin.class, ColumnEncoderBin::getColMaxs);
				for(int j = 0; j < colIDs.length; j++) {
					ret.add(new Tuple2<>(colIDs[j], String.valueOf(colMins[j])));
					ret.add(new Tuple2<>(colIDs[j], String.valueOf(colMaxs[j])));
				}
			}

			return ret.iterator();
		}
	}

	/**
	 * This function assigns codes to globally distinct values of recoded columns and writes the resulting column map in
	 * textcell (IJV) format to the output. (part of distributed recode map construction, used for recoding, binning and
	 * dummy coding). We operate directly over schema-specific objects to avoid unnecessary string conversion, as well
	 * as reduce memory overhead and shuffle.
	 */
	public static class TransformEncodeGroupFunction
		implements FlatMapFunction<Tuple2<Integer, Iterable<Object>>, String> {
		private static final long serialVersionUID = -1034187226023517119L;

		private final MultiColumnEncoder _encoder;
		private final MaxLongAccumulator _accMax;

		public TransformEncodeGroupFunction(MultiColumnEncoder encoder, MaxLongAccumulator accMax) {
			_encoder = encoder;
			_accMax = accMax;
		}

		@Override
		public Iterator<String> call(Tuple2<Integer, Iterable<Object>> arg0) throws Exception {
			String scolID = String.valueOf(arg0._1());
			int colID = Integer.parseInt(scolID);
			Iterator<Object> iter = arg0._2().iterator();
			ArrayList<String> ret = new ArrayList<>();

			int rowID = 1;
			StringBuilder sb = new StringBuilder();

			// handle recode maps
			if(_encoder.containsEncoderForID(colID, ColumnEncoderRecode.class) ||
					_encoder.containsEncoderForID(colID, ColumnEncoderBagOfWords.class)) {
				while(iter.hasNext()) {
					String token = TfUtils.sanitizeSpaces(iter.next().toString());
					sb.append(rowID).append(' ').append(scolID).append(' ');
					sb.append(ColumnEncoderRecode.constructRecodeMapEntry(token, rowID));
					ret.add(sb.toString());
					sb.setLength(0);
					rowID++;
				}
			}
			// handle bin boundaries
			else if(_encoder.containsEncoderForID(colID, ColumnEncoderBin.class)) {
				ColumnEncoderBin baEncoder = _encoder.getColumnEncoder(colID, ColumnEncoderBin.class);
				if (baEncoder.getBinMethod() == ColumnEncoderBin.BinMethod.EQUI_WIDTH) {
					double min = Double.MAX_VALUE;
					double max = -Double.MAX_VALUE;
					while(iter.hasNext()) {
						double value = Double.parseDouble(iter.next().toString());
						min = Math.min(min, value);
						max = Math.max(max, value);
					}
					assert baEncoder != null;
					baEncoder.computeBins(min, max);
				}
				else //TODO: support equi-height
					throw new DMLRuntimeException("Binning method "+baEncoder.getBinMethod().toString()
						+" is not support for Spark");

				double[] binMins = baEncoder.getBinMins();
				double[] binMaxs = baEncoder.getBinMaxs();
				for(int i = 0; i < binMins.length; i++) {
					sb.append(rowID).append(' ').append(scolID).append(' ');
					sb.append(binMins[i]).append(Lop.DATATYPE_PREFIX).append(binMaxs[i]);
					ret.add(sb.toString());
					sb.setLength(0);
					rowID++;
				}
			}
			else {
				throw new DMLRuntimeException("Unsupported metadata output for encoder: \n" + _encoder);
			}
			_accMax.add(rowID - 1L);

			return ret.iterator();
		}
	}

	public static class TransformEncodeBuild2Function
		implements PairFlatMapFunction<Iterator<Tuple2<Long, FrameBlock>>, Integer, ColumnMetadata> {
		private static final long serialVersionUID = 6336375833412029279L;

		private EncoderMVImpute _encoder;

		public TransformEncodeBuild2Function(EncoderMVImpute encoder) {
			_encoder = encoder;
		}

		@Override
		public Iterator<Tuple2<Integer, ColumnMetadata>> call(Iterator<Tuple2<Long, FrameBlock>> iter)
			throws Exception {
			// build meta data (e.g., histograms and means)
			while(iter.hasNext()) {
				FrameBlock block = iter.next()._2();
				_encoder.build(block);
			}

			// extract meta data
			ArrayList<Tuple2<Integer, ColumnMetadata>> ret = new ArrayList<>();
			int[] collist = _encoder.getColList();
			for(int j = 0; j < collist.length; j++) {
				if(_encoder.getMethod(collist[j]) == MVMethod.GLOBAL_MODE) {
					HashMap<String, Long> hist = _encoder.getHistogram(collist[j]);
					for(Entry<String, Long> e : hist.entrySet())
						ret.add(new Tuple2<>(collist[j], new ColumnMetadata(e.getValue(), e.getKey())));
				}
				else if(_encoder.getMethod(collist[j]) == MVMethod.GLOBAL_MEAN) {
					ret.add(new Tuple2<>(collist[j], new ColumnMetadata(_encoder.getNonMVCount(collist[j]),
						String.valueOf(_encoder.getMeans()[j]._sum))));
				}
				else if(_encoder.getMethod(collist[j]) == MVMethod.CONSTANT) {
					ret.add(new Tuple2<>(collist[j], new ColumnMetadata(0, _encoder.getReplacement(collist[j]))));
				}
			}

			return ret.iterator();
		}
	}

	public static class TransformEncodeGroup2Function
		implements FlatMapFunction<Tuple2<Integer, Iterable<ColumnMetadata>>, String> {
		private static final long serialVersionUID = 702100641492347459L;

		private EncoderMVImpute _encoder;

		public TransformEncodeGroup2Function(EncoderMVImpute encoder) {
			_encoder = encoder;
		}

		@Override
		public Iterator<String> call(Tuple2<Integer, Iterable<ColumnMetadata>> arg0) throws Exception {
			int colix = arg0._1();
			Iterator<ColumnMetadata> iter = arg0._2().iterator();
			ArrayList<String> ret = new ArrayList<>();

			// compute global mode of categorical feature, i.e., value with highest frequency
			if(_encoder.getMethod(colix) == MVMethod.GLOBAL_MODE) {
				HashMap<String, Long> hist = new HashMap<>();
				while(iter.hasNext() ) {
					ColumnMetadata cmeta = iter.next();
					if(!cmeta.isDefault()){
						Long tmp = hist.get(cmeta.getMvValue());
						hist.put(cmeta.getMvValue(), cmeta.getNumDistinct() + ((tmp != null) ? tmp : 0));
					}
				}
				long max = Long.MIN_VALUE;
				String mode = null;
				for(Entry<String, Long> e : hist.entrySet())
					if(e.getValue() > max) {
						mode = e.getKey();
						max = e.getValue();
					}
				ret.add("-2 " + colix + " " + TfUtils.sanitizeSpaces(mode));
			}
			// compute global mean of categorical feature
			else if(_encoder.getMethod(colix) == MVMethod.GLOBAL_MEAN) {
				KahanObject kbuff = new KahanObject(0, 0);
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				int count = 0;
				while(iter.hasNext()) {
					ColumnMetadata cmeta = iter.next();
					if(!cmeta.isDefault()){
						kplus.execute2(kbuff, Double.parseDouble(cmeta.getMvValue()));
						count += cmeta.getNumDistinct();
					}
				}
				if(count > 0)
					ret.add("-2 " + colix + " " + kbuff._sum / count);
			}
			// pass-through constant label
			else if(_encoder.getMethod(colix) == MVMethod.CONSTANT) {
				if(iter.hasNext())
					ret.add("-2 " + colix + " " + TfUtils.sanitizeSpaces(iter.next().getMvValue()));
			}

			return ret.iterator();
		}
	}

	public boolean getMetaReturn() {
		return _metaReturn;
	}
}

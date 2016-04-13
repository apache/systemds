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

package org.apache.sysml.runtime.instructions.flink.utils;

import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.accumulators.LongCounter;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.functions.RuntimeContext;
import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.FlinkExecutionContext;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixCell;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.mapred.IndexedMatrixValue;
import org.apache.sysml.runtime.matrix.mapred.ReblockBuffer;
import org.apache.sysml.runtime.util.FastStringTokenizer;
import org.apache.sysml.runtime.util.UtilFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class DataSetConverterUtils {

	/**
	 * @param env
	 * @param input
	 * @param mcOut
	 * @param outputEmptyBlocks
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static DataSet<Tuple2<MatrixIndexes, MatrixBlock>> textCellToBinaryBlock(ExecutionEnvironment env,
																					DataSet<Tuple2<LongWritable, Text>> input,
																					MatrixCharacteristics mcOut,
																					boolean outputEmptyBlocks)
			throws DMLRuntimeException {
		//convert textcell dataset to binary block dataset (w/ partial blocks)
		DataSet<Text> temp = input.map(new ExtractElement(1)).returns(Text.class);
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out = temp.mapPartition(new TextToBinaryBlockFunction(mcOut));


		//inject empty blocks (if necessary)
		if (outputEmptyBlocks && mcOut.mightHaveEmptyBlocks()) {
			out = out.union(
					FlinkUtils.getEmptyBlockDataSet(env, mcOut));
		}

		//aggregate partial matrix blocks
		out = DataSetAggregateUtils.mergeByKey(out);

		return out;
	}


	/**
	 * @param env
	 * @param input
	 * @param mcOut
	 * @param outputEmptyBlocks
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static DataSet<Tuple2<MatrixIndexes, MatrixBlock>> binaryCellToBinaryBlock(ExecutionEnvironment env,
																					  DataSet<Tuple2<MatrixIndexes, MatrixCell>> input,
																					  MatrixCharacteristics mcOut,
																					  boolean outputEmptyBlocks)
			throws DMLRuntimeException {
		//convert binarycell dataset to binary block dataset (w/ partial blocks)
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out = input
				.mapPartition(new BinaryCellToBinaryBlockFunction(mcOut));

		//inject empty blocks (if necessary)
		if (outputEmptyBlocks && mcOut.mightHaveEmptyBlocks()) {
			out = out.union(
					FlinkUtils.getEmptyBlockDataSet(env, mcOut));
		}

		//aggregate partial matrix blocks
		out = DataSetAggregateUtils.mergeByKey(out);

		return out;
	}


	public static DataSet<Tuple2<MatrixIndexes, MatrixBlock>> csvToBinaryBlock(ExecutionEnvironment env,
																			   DataSet<Tuple2<Integer, String>> input,
																			   MatrixCharacteristics mcOut,
																			   boolean hasHeader,
																			   String delim,
																			   boolean fill,
																			   double fillValue) throws DMLRuntimeException {

		//determine unknown dimensions and sparsity if required
		if (!mcOut.dimsKnown(true)) {
			try {
				List<String> row = input.map(new CSVAnalysisFunction(delim)).first(1).collect();
				JobExecutionResult result = env.getLastJobExecutionResult();
				long numRows = result.getAccumulatorResult(CSVAnalysisFunction.NUM_ROWS);
				numRows = numRows - (hasHeader ? 1 : 0);
				long numCols = row.get(0).split(delim).length;
				long nonZeroValues = result.getAccumulatorResult(CSVAnalysisFunction.NON_ZERO_VALUES);

				mcOut.set(numRows, numCols, mcOut.getRowsPerBlock(), mcOut.getColsPerBlock(), nonZeroValues);
			} catch (Exception e) {
				throw new DMLRuntimeException("Could not get metadata for input data set: ", e);
			}
		}

		// zip with row id
		DataSet<Tuple2<Long, String>> indexed = IndexUtils.zipWithRowIndex(input);
		//convert to binary block dataset (w/ partial blocks)
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> out = indexed.mapPartition(
				new CSVToBinaryBlockFunction(mcOut, delim, fill, fillValue));
		//aggregate partial matrix blocks
		out = DataSetAggregateUtils.mergeByKey(out);

		return out;
	}


	/**
	 * This functions allows to map dataset partitions of csv rows into a set of partial binary blocks.
	 * <p>
	 * NOTE: For this csv to binary block function, we need to hold all output blocks per partition
	 * in-memory. Hence, we keep state of all column blocks and aggregate row segments into these blocks.
	 * In terms of memory consumption this is better than creating partial blocks of row segments.
	 */
	private static class CSVToBinaryBlockFunction
			implements MapPartitionFunction<Tuple2<Long, String>, Tuple2<MatrixIndexes, MatrixBlock>> {
		private static final long serialVersionUID = -4948430402942717043L;

		private long _rlen = -1;
		private long _clen = -1;
		private int _brlen = -1;
		private int _bclen = -1;
		private String _delim = null;
		private boolean _fill = false;
		private double _fillValue = 0;

		public CSVToBinaryBlockFunction(MatrixCharacteristics mc, String delim, boolean fill, double fillValue) {
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();
			_delim = delim;
			_fill = fill;
			_fillValue = fillValue;
		}

		// Creates new state of empty column blocks for current global row index.
		private void createBlocks(long rowix, int lrlen, MatrixIndexes[] ix, MatrixBlock[] mb) {
			//compute row block index and number of column blocks
			long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
			int ncblks = (int) Math.ceil((double) _clen / _bclen);

			//create all column blocks (assume dense since csv is dense text format)
			for (int cix = 1; cix <= ncblks; cix++) {
				int lclen = (int) UtilFunctions.computeBlockSize(_clen, cix, _bclen);
				ix[cix - 1] = new MatrixIndexes(rix, cix);
				mb[cix - 1] = new MatrixBlock(lrlen, lclen, false);
			}
		}

		// Flushes current state of filled column blocks to output list.
		private void flushBlocksToList(MatrixIndexes[] ix, MatrixBlock[] mb,
									   Collector<Tuple2<MatrixIndexes, MatrixBlock>> out)
				throws DMLRuntimeException {
			int len = ix.length;
			for (int i = 0; i < len; i++)
				if (mb[i] != null) {
					out.collect(new Tuple2<MatrixIndexes, MatrixBlock>(ix[i], mb[i]));
					mb[i].examSparsity(); //ensure right representation
				}
		}

		@Override
		public void mapPartition(Iterable<Tuple2<Long, String>> values,
								 Collector<Tuple2<MatrixIndexes, MatrixBlock>> out) throws Exception {

			int ncblks = (int) Math.ceil((double) _clen / _bclen);
			MatrixIndexes[] ix = new MatrixIndexes[ncblks];
			MatrixBlock[] mb = new MatrixBlock[ncblks];
			Iterator<Tuple2<Long, String>> arg0 = values.iterator();

			while (arg0.hasNext()) {
				Tuple2<Long, String> tmp = arg0.next();
				String row = tmp.f1;
				long rowix = tmp.f0 + 1;

				long rix = UtilFunctions.computeBlockIndex(rowix, _brlen);
				int pos = UtilFunctions.computeCellInBlock(rowix, _brlen);

				//create new blocks for entire row
				if (ix[0] == null || ix[0].getRowIndex() != rix) {
					if (ix[0] != null)
						flushBlocksToList(ix, mb, out);
					long len = UtilFunctions.computeBlockSize(_rlen, rix, _brlen);
					createBlocks(rowix, (int) len, ix, mb);
				}

				//process row data
				String[] parts = IOUtilFunctions.split(row, _delim);
				boolean emptyFound = false;
				for (int cix = 1, pix = 0; cix <= ncblks; cix++) {
					int lclen = (int) UtilFunctions.computeBlockSize(_clen, cix, _bclen);
					for (int j = 0; j < lclen; j++) {
						String part = parts[pix++];
						emptyFound |= part.isEmpty() && !_fill;
						double val = (part.isEmpty() && _fill) ?
								_fillValue : Double.parseDouble(part);
						mb[cix - 1].appendValue(pos, j, val);
					}
				}

				//sanity check empty cells filled w/ values
				IOUtilFunctions.checkAndRaiseErrorCSVEmptyField(row, _fill, emptyFound);
			}

			//flush last blocks
			flushBlocksToList(ix, mb, out);
		}
	}

	public static DataSet<String> binaryBlockToCsv(DataSet<Tuple2<MatrixIndexes, MatrixBlock>> inBlock,
												   MatrixCharacteristics mcIn, CSVFileFormatProperties csvProps,
												   boolean strict) {
		DataSet<Tuple2<MatrixIndexes, MatrixBlock>> input = inBlock;
		if (mcIn.getCols() > mcIn.getColsPerBlock()) {
			input = input.groupBy(new BinaryBlockGroupByRow()).reduceGroup(new ConcatenateBlocks(mcIn));
		}

		if (strict) {
			input = input.sortPartition(0, Order.ASCENDING);
		}
		DataSet<String> out = input.flatMap(new BinaryBlockToCsvFlatMapper(csvProps));
		return out;
	}

	public static class BinaryBlockToCsvFlatMapper
			implements FlatMapFunction<Tuple2<MatrixIndexes, MatrixBlock>, String> {
		private final CSVFileFormatProperties csvProps;

		public BinaryBlockToCsvFlatMapper(CSVFileFormatProperties csvProps) {
			this.csvProps = csvProps;
		}

		@Override
		public void flatMap(Tuple2<MatrixIndexes, MatrixBlock> value, Collector<String> out) throws Exception {
			MatrixBlock block = value.f1;
			int numRows = block.getNumRows();
			int numCols = block.getNumColumns();
			StringBuilder strBld = new StringBuilder(numCols * numCols - 1);
			;
			for (int r = 0; r < numRows; r++) {
				strBld.setLength(0);
				for (int c = 0; c < numCols; c++) {
					double v = block.getValue(r, c);
					strBld.append(v);
					if (c < numCols - 1) {
						strBld.append(csvProps.getDelim());
					}
				}
				out.collect(strBld.toString());
			}

		}
	}

	public static class BinaryBlockGroupByRow implements KeySelector<Tuple2<MatrixIndexes, MatrixBlock>, Long> {

		@Override
		public Long getKey(Tuple2<MatrixIndexes, MatrixBlock> value) throws Exception {
			return value.f0.getRowIndex();
		}
	}

	public static class ConcatenateBlocks
			implements GroupReduceFunction<Tuple2<MatrixIndexes, MatrixBlock>, Tuple2<MatrixIndexes, MatrixBlock>> {

		private final MatrixCharacteristics matrixCharacteristics;

		public ConcatenateBlocks(MatrixCharacteristics matrixCharacteristics) {
			this.matrixCharacteristics = matrixCharacteristics;
		}

		@Override
		public void reduce(Iterable<Tuple2<MatrixIndexes, MatrixBlock>> values,
						   Collector<Tuple2<MatrixIndexes, MatrixBlock>> out) throws Exception {
			long numCols = matrixCharacteristics.getCols();

			long currentRow = -1;
			MatrixBlock block = null;
			for (Tuple2<MatrixIndexes, MatrixBlock> indexedBlock : values) {
				if (block == null) {
					block = new MatrixBlock(indexedBlock.f1.getNumRows(), (int) numCols,
							indexedBlock.f1.isInSparseFormat());
					currentRow = indexedBlock.f0.getRowIndex();
				}
				int cl = ((int) indexedBlock.f0.getColumnIndex() - 1) * indexedBlock.f1.getNumColumns();
				int cu = (int) indexedBlock.f0.getColumnIndex() * indexedBlock.f1.getNumColumns() - 1;
				block.copy(0, 0, cl, cu, indexedBlock.f1, false);
			}
			block.recomputeNonZeros();
			MatrixIndexes index = new MatrixIndexes(currentRow, 1);
			out.collect(new Tuple2<MatrixIndexes, MatrixBlock>(index, block));
		}
	}

	private static class CSVAnalysisFunction extends RichMapFunction<Tuple2<Integer, String>, String> {

		public static final String NUM_ROWS = "numRows";
		public static final String NON_ZERO_VALUES = "nonZeroValues";

		private final LongCounter numValues = new LongCounter();
		private final LongCounter nonZeroValues = new LongCounter();

		private final String delimiter;

		public CSVAnalysisFunction(String delim) {
			this.delimiter = delim;
		}

		@Override
		public void open(Configuration parameters) throws Exception {
			getRuntimeContext().addAccumulator(NUM_ROWS, this.numValues);
			getRuntimeContext().addAccumulator(NON_ZERO_VALUES, this.nonZeroValues);
		}

		@Override
		public String map(Tuple2<Integer, String> tuple) throws Exception {
			//parse input line
			String[] cols = IOUtilFunctions.split(tuple.f1, delimiter);
			//determine number of non-zeros of row (w/o string parsing)
			long lnnz = 0;
			for (String col : cols) {
				if (!col.isEmpty() && !col.equals("0") && !col.equals("0.0")) {
					lnnz++;
				}
			}
			//update counters
			this.nonZeroValues.add(lnnz);
			this.numValues.add(1);

			return tuple.f1;
		}
	}

	public static final class ExtractElement<T extends Tuple, R> implements MapFunction<T, R> {
		private int id;

		public ExtractElement(int id) {
			this.id = id;
		}

		@Override
		public R map(T value) {
			return (R) value.getField(id);

		}
	}

	private static class TextToBinaryBlockFunction extends CellToBinaryBlockFunction<Text> {
		protected TextToBinaryBlockFunction(MatrixCharacteristics mc) {
			super(mc);
		}

		public TextToBinaryBlockFunction() {
			super();
		}

		@Override
		public void mapPartition(Iterable<Text> textCollection, Collector<Tuple2<MatrixIndexes, MatrixBlock>> out)
				throws Exception {
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _brlen, _bclen);
			FastStringTokenizer st = new FastStringTokenizer(' ');

			long row;
			long col;
			double val;
			//get input string (ignore matrix market comments)
			for (Text text : textCollection) {
				String strVal = text.toString();
				if (strVal.startsWith("%")) {
					continue;
				}

				//parse input ijv triple
				st.reset(strVal);
				row = st.nextLong();
				col = st.nextLong();
				val = st.nextDouble();

				//flush buffer if necessary
				if (rbuff.getSize() >= rbuff.getCapacity()) {
					flushBufferToList(out, rbuff);
				}

				//add value to reblock buffer
				rbuff.appendCell(row, col, val);
			}

			//final flush buffer
			flushBufferToList(out, rbuff);
		}
	}

	/////////////////////////////////
	// TEXTCELL-SPECIFIC FUNCTIONS

	public static abstract class CellToBinaryBlockFunction<R>
			extends RichMapPartitionFunction<R, Tuple2<MatrixIndexes, MatrixBlock>> {
		//internal buffer size (aligned w/ default matrix block size)
		protected static final int BUFFER_SIZE = 4 * 1000 * 1000; //4M elements (32MB)
		protected int _bufflen = -1;

		protected long _rlen = -1;
		protected long _clen = -1;
		protected int _brlen = -1;
		protected int _bclen = -1;

		MatrixCharacteristics mc = null;
		RuntimeContext ctx = null;

		public CellToBinaryBlockFunction() {

		}

		public CellToBinaryBlockFunction(MatrixCharacteristics mc) {
			this.mc = mc;
		}

		@Override
		public void open(Configuration parameters) throws Exception {
			_rlen = mc.getRows();
			_clen = mc.getCols();
			_brlen = mc.getRowsPerBlock();
			_bclen = mc.getColsPerBlock();

			int numSlots = getRuntimeContext().getNumberOfParallelSubtasks();
			long freeMemory = FlinkExecutionContext.getUDFMemoryBudget();

			// in flink, each slot only has limited memory - we have to dynamically set the buffer size so that we don't run out of memory
			// udf memory budget / number of slots / longs / 3 dimensional buffer
			int numElems = (int) ((freeMemory / numSlots) / 8) / 3;

			//determine upper bounded buffer len
			_bufflen = (int) Math.min(_rlen * _clen, Math.min(BUFFER_SIZE, numElems));
		}

		/**
		 * @param out
		 * @param rbuff
		 * @throws IOException
		 * @throws DMLRuntimeException
		 */
		public void flushBufferToList(Collector<Tuple2<MatrixIndexes, MatrixBlock>> out, ReblockBuffer rbuff)
				throws IOException, DMLRuntimeException {
			//temporary list of indexed matrix values to prevent library dependencies
			ArrayList<IndexedMatrixValue> rettmp = new ArrayList<IndexedMatrixValue>();
			rbuff.flushBufferToBinaryBlocks(rettmp);

			for (Tuple2<MatrixIndexes, MatrixBlock> block : fromIndexedMatrixBlock(rettmp)) {
				out.collect(block);
			}
		}


	}

	/////////////////////////////////
	// BINARYCELL-SPECIFIC FUNCTIONS

	private static class BinaryCellToBinaryBlockFunction
			extends CellToBinaryBlockFunction<Tuple2<MatrixIndexes, MatrixCell>> {
		protected BinaryCellToBinaryBlockFunction(MatrixCharacteristics mc) {
			super(mc);
		}

		public BinaryCellToBinaryBlockFunction() {
			super();
		}

		@Override
		public void mapPartition(Iterable<Tuple2<MatrixIndexes, MatrixCell>> binCollection,
								 Collector<Tuple2<MatrixIndexes, MatrixBlock>> out)
				throws Exception {
			ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
			ReblockBuffer rbuff = new ReblockBuffer(_bufflen, _rlen, _clen, _brlen, _bclen);

			for (Tuple2<MatrixIndexes, MatrixCell> bin : binCollection) {
				//unpack the binary cell input
				Tuple2<MatrixIndexes, MatrixCell> tmp = bin;

				//parse input ijv triple
				long row = tmp.f0.getRowIndex();
				long col = tmp.f0.getColumnIndex();
				double val = tmp.f1.getValue();

				//flush buffer if necessary
				if (rbuff.getSize() >= rbuff.getCapacity())
					flushBufferToList(out, rbuff);

				//add value to reblock buffer
				rbuff.appendCell(row, col, val);
			}

			//final flush buffer
			flushBufferToList(out, rbuff);
		}
	}


	/**
	 * @param in
	 * @return
	 */
	public static ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> fromIndexedMatrixBlock(
			ArrayList<IndexedMatrixValue> in) {
		ArrayList<Tuple2<MatrixIndexes, MatrixBlock>> ret = new ArrayList<Tuple2<MatrixIndexes, MatrixBlock>>();
		for (IndexedMatrixValue imv : in)
			ret.add(fromIndexedMatrixBlock(imv));

		return ret;
	}

	/**
	 * @param in
	 * @return
	 */
	public static Tuple2<MatrixIndexes, MatrixBlock> fromIndexedMatrixBlock(IndexedMatrixValue in) {
		return new Tuple2<MatrixIndexes, MatrixBlock>(in.getIndexes(), (MatrixBlock) in.getValue());
	}
}

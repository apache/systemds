/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.mapred.KeyValueTextInputFormat;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.cp.CPOperand;
import com.ibm.bi.dml.runtime.instructions.spark.ReblockSPInstruction.DropEmptyBinaryCells;
import com.ibm.bi.dml.runtime.instructions.spark.functions.ConvertCSVLinesToMatrixBlocks;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixCell;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;
import com.ibm.bi.dml.runtime.matrix.operators.Operator;

public class CSVReblockSPInstruction extends UnarySPInstruction {

	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n"
			+ "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private int brlen;
	private int bclen;
	private boolean hasHeader;
	private String delim;
	private boolean fill;
	private double missingValue;

	public CSVReblockSPInstruction(Operator op, CPOperand in, CPOperand out,
			int br, int bc, boolean hasHeader, String delim, boolean fill,
			double missingValue, String opcode, String instr) {
		super(op, in, out, opcode, instr);
		brlen = br;
		bclen = bc;
		this.hasHeader = hasHeader;
		this.delim = delim;
		this.fill = fill;
		this.missingValue = missingValue;
	}

	public static Instruction parseInstruction(String str)
			throws DMLRuntimeException {
		String opcode = InstructionUtils.getOpCode(str);
		if (opcode.compareTo("csvrblk") != 0) {
			throw new DMLRuntimeException(
					"Incorrect opcode for ReblockSPInstruction:" + opcode);
		}

		// Example parts of CSVReblockSPInstruction:
		// [csvrblk, pREADmissing_val_maps·MATRIX·DOUBLE, _mVar37·MATRIX·DOUBLE,
		// 1000, 1000, false, ,, true, 0.0]
		String parts[] = InstructionUtils.getInstructionPartsWithValueType(str);

		CPOperand in = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		CPOperand out = new CPOperand("", ValueType.UNKNOWN, DataType.UNKNOWN);
		in.split(parts[1]);
		out.split(parts[2]);

		int brlen = Integer.parseInt(parts[3]);
		int bclen = Integer.parseInt(parts[4]);

		boolean hasHeader = Boolean.parseBoolean(parts[5]);
		String delim = parts[6];
		boolean fill = Boolean.parseBoolean(parts[7]);
		double missingValue = Double.parseDouble(parts[8]);

		Operator op = null; // no operator for ReblockSPInstruction
		return new CSVReblockSPInstruction(op, in, out, brlen, bclen,
				hasHeader, delim, fill, missingValue, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec)
			throws DMLRuntimeException, DMLUnsupportedOperationException {
		SparkExecutionContext sec = (SparkExecutionContext) ec;
		String opcode = getOpcode();

		if (opcode.equalsIgnoreCase("csvrblk")) {
			MatrixObject mo = sec.getMatrixObject(input1.getName());
			MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
			if (iimd.getInputInfo() != InputInfo.CSVInputInfo) {
				throw new DMLRuntimeException(
						"The given InputInfo is not implemented for ReblockSPInstruction:"
								+ iimd.getInputInfo());
			}
			String fileName = mo.getFileName();

			JavaRDD<String> csvLines = sec.getSparkContext().textFile(fileName);

			// Compute (if not already computed) the start offset of each
			// partition of our input,
			// RDD, so that we can parse all the partitions in parallel and send
			// each chunk of
			// the matrix to the appropriate block.
			Broadcast<HashMap<Integer, Long>> offsetsBroadcast = sec
					.getSparkContext().broadcast(getRowOffsets(csvLines));
			JavaPairRDD<MatrixIndexes, MatrixBlock> chunks = JavaPairRDD
					.fromJavaRDD(csvLines.mapPartitionsWithIndex(
							new ConvertCSVLinesToMatrixBlocks(offsetsBroadcast,
									brlen, bclen, hasHeader, fileName, fill,
									missingValue), true));

			// Merge chunks according to their block index
			// Each entry in chunks is a tuple
			// ((block row number, block column number), (row within block, list
			// of values))
			// We need to group by the block row number and block column number.
			// The nested lists are already set up to help us do this.
			JavaPairRDD<MatrixIndexes, MatrixBlock> blocksRDD = chunks
					.reduceByKey(new MergeBlocks());

			// put output RDD handle into symbol table
			// TODO: Handle inputs with unknown sizes
			MatrixCharacteristics mcOut = sec.getMatrixCharacteristics(output.getName());
			if(!mcOut.dimsKnown()) {
				throw new DMLRuntimeException("TODO: Handle csvreblock with unknown sizes");
			}
			sec.setRDDHandleForVariable(output.getName(), blocksRDD);
		} else {
			throw new DMLRuntimeException(
					"In CSVReblockSPInstruction,  Unknown opcode in Instruction: "
							+ toString());
		}

	}

	HashMap<Integer, Long> rowOffsets = null;

	private HashMap<Integer, Long> getRowOffsets(JavaRDD<String> csvLines) {
		// Initialize the row offsets if necessary
		if (null == rowOffsets) {

			// Start by counting the number of lines in each partition.
			JavaRDD<Tuple2<Integer, Long>> lineCounts = csvLines
					.mapPartitionsWithIndex(new CountLines(), true);

			// Not sure if the sort here is necessary.
			List<Tuple2<Integer, Long>> linesPerPartition = JavaPairRDD
					.fromJavaRDD(lineCounts).sortByKey().collect();
			// lineCounts.sortBy((p: (Int, Long)) => p._1, true, 1).collect()

			// Compute the offset of the first line in the each partition.
			// This code assumes that partitions are numbered in order, but does
			// not assume that
			// partition numbers are contiguous
			rowOffsets = new HashMap<Integer, Long>();
			rowOffsets.put(linesPerPartition.get(0)._1, 0L);

			int prevPartNo = linesPerPartition.get(0)._1;
			for (int i = 1; i < linesPerPartition.size(); i++) {
				Integer partNo = linesPerPartition.get(i)._1;
				Long prevOffset = rowOffsets.get(prevPartNo);
				long curOffset = prevOffset + linesPerPartition.get(i - 1)._2;

				rowOffsets.put(partNo, curOffset);
				prevPartNo = partNo;
			}
		}

		return rowOffsets;
	}

	public static class CountLines
			implements
			Function2<Integer, Iterator<String>, Iterator<Tuple2<Integer, Long>>> {
		private static final long serialVersionUID = -2611946238807543849L;

		@Override
		public Iterator<Tuple2<Integer, Long>> call(Integer partNo,
				Iterator<String> lines) throws Exception {
			long nline = 0;
			while (lines.hasNext()) {
				lines.next();
				nline = nline + 1;
			}

			// Package up the result in a format that Spark understands
			ArrayList<Tuple2<Integer, Long>> retVal = new ArrayList<Tuple2<Integer, Long>>();
			retVal.add(new Tuple2<Integer, Long>(partNo, nline));
			return retVal.iterator();
		}
	}

	public static class MergeBlocks implements
			Function2<MatrixBlock, MatrixBlock, MatrixBlock> {

		private static final long serialVersionUID = -8881019027250258850L;

		@Override
		public MatrixBlock call(MatrixBlock b1, MatrixBlock b2)
				throws Exception {
			MatrixBlock ret = null;
			if (b1.getNumRows() != b2.getNumRows()
					|| b1.getNumColumns() != b2.getNumColumns()) {
				throw new DMLRuntimeException("Mismatched block sizes: "
						+ b1.getNumRows() + " " + b1.getNumColumns() + " "
						+ b2.getNumRows() + " " + b2.getNumColumns());
			}

			boolean isB1Empty = b1.isEmpty();
			boolean isB2Empty = b2.isEmpty();

			if (isB1Empty && !isB2Empty) {
				return b2; // b2.clone();
			} else if (!isB1Empty && isB2Empty) {
				return b1;
			} else if (isB1Empty && isB2Empty) {
				return b1;
			}

			// ret = b1;
			// ret.merge(b2, false);

			if (b1.isInSparseFormat() && b2.isInSparseFormat()) {
				ret = mergeSparseBlocks(b1, b2);
			} else if (false == b1.isInSparseFormat()) {
				// b1 dense --> Merge b2 directly into a copy of b1, regardless
				// of whether it's dense or sparse
				ret = mergeIntoDenseBlock(b1, b2);
			} else {
				// b1 is not dense, b2 is dense --> Merge b1 into a copy of b2
				ret = mergeIntoDenseBlock(b2, b1);
			}

			// Sanity check
			if (ret.getNonZeros() != b1.getNonZeros() + b2.getNonZeros()) {
				throw new DMLRuntimeException(
						"Number of non-zeros dont match: " + ret.getNonZeros()
								+ " " + b1.getNonZeros() + " "
								+ b2.getNonZeros());
			}

			return ret;
		}

		private MatrixBlock mergeSparseBlocks(MatrixBlock b1, MatrixBlock b2)
				throws DMLRuntimeException {

			// Validate inputs
			if (false == b1.isInSparseFormat()) {
				throw new DMLRuntimeException(
						"First block is not sparse in mergeSparseBlocks");
			}
			if (false == b2.isInSparseFormat()) {
				throw new DMLRuntimeException(
						"Second block is not sparse in mergeSparseBlocks");
			}

			if (b1.isEmpty()) {
				throw new DMLRuntimeException(
						"Empty block passed as first argument in mergeSparseBlocks");
			}
			if (b2.isEmpty()) {
				throw new DMLRuntimeException(
						"Empty block passed as second argument in mergeSparseBlocks");
			}

			// Construct merged output. Note shallow copy of rows.
			MatrixBlock ret = new MatrixBlock(b1.getNumRows(),
					b1.getNumColumns(), true);

			for (int r = 0; r < ret.getNumRows(); r++) {
				// Read directly from the internal representation
				SparseRow row1 = b1.getSparseRows()[r];

				SparseRow row2 = b2.getSparseRows()[r];

				if (null != row1 && null != row2) {
					// Both inputs have content for this row.
					SparseRow mergedRow = new SparseRow(row1);

					// TODO: Should we check for conflicting cells (O(nlogn)
					// overhead)?

					int[] indexes = row2.getIndexContainer();
					double[] values = row2.getValueContainer();

					for (int i = 0; i < indexes.length; i++) {
						mergedRow.append(indexes[i], values[i]);
					}

					mergedRow.sort();
					ret.appendRow(r, mergedRow);

					// throw new SystemMLException ("CONFLICTING_ROWS", r);
				} else if (null != row1) {
					// Input 1 has this row, input 2 does not
					ret.appendRow(r, row1);
				} else if (null != row2) {
					// Input 2 has this row, input 1 does not
					ret.appendRow(r, row2);
				} else {
					// Neither input has this row; do nothing
				}
			}

			return ret;
		}

		private MatrixBlock mergeIntoDenseBlock(MatrixBlock denseBlock,
				MatrixBlock otherBlock) throws DMLRuntimeException {
			if (denseBlock.isInSparseFormat()) {
				throw new DMLRuntimeException(
						"First block is not dens in mergeIntoDenseBlock");
			}

			// Start with the contents of the dense input
			MatrixBlock ret = new MatrixBlock(denseBlock.getNumRows(),
					denseBlock.getNumColumns(), false);
			ret.copy(denseBlock);

			// Add the contents of the other block.
			int numNonzerosAdded = 0;

			if (otherBlock.isInSparseFormat()) {
				// Other block is sparse, so we can directly access the nonzero
				// values.
				SparseRowsIterator itr = otherBlock.getSparseRowsIterator();
				while (itr.hasNext()) {
					IJV ijv = itr.next();

					// Sanity-check the previous value; the inputs to this
					// function shouldn't overlap
					double prevValue = ret.getValue(ijv.i, ijv.j);
					if (0.0D != prevValue) {
						throw new DMLRuntimeException(
								"NONZERO_VALUE_SHOULD_BE_ZERO");
						// throw new SystemMLException
						// ("NONZERO_VALUE_SHOULD_BE_ZERO", prevValue, ijv.i,
						// ijv.j, otherBlock, denseBlock); }
					}

					ret.setValue(ijv.i, ijv.j, ijv.v);
					numNonzerosAdded++;
				}
			} else {
				// Other block is dense; iterate over all values, adding
				// nonzeros.
				for (int r = 0; r < ret.getNumRows(); r++) {
					for (int c = 0; c < ret.getNumColumns(); c++) {
						double prevValue = ret.getValue(r, c);
						double otherValue = otherBlock.getValue(r, c);

						if (0.0D != otherValue) {
							if (0.0D != prevValue) {
								throw new DMLRuntimeException(
										"NONZERO_VALUE_SHOULD_BE_ZERO");
								// throw new SystemMLException
								// ("NONZERO_VALUE_SHOULD_BE_ZERO", prevValue,
								// ijv.i, ijv.j, otherBlock, denseBlock); }
							}

							// Use the "safe" accessor method, which also
							// updates sparsity information
							ret.setValue(r, c, otherValue);

							numNonzerosAdded++;
						}

					}
				}

			}

			// Sanity check
			if (numNonzerosAdded != otherBlock.getNonZeros()) {
				throw new DMLRuntimeException("Incorrect number of non-zeros");
				// throw new SystemMLException ("WRONG_NONZERO_COUNT",
				// numNonzerosAdded, otherBlock.getNonZeros (), otherBlock,
				// denseBlock); }
			}
			return ret;
		}

	}
}

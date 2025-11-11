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

package org.apache.sysds.runtime.instructions.ooc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;

public class CSVReblockOOCInstruction extends ComputationOOCInstruction {
	private static final int MAX_BLOCKS_IN_CACHE = 40;

	private final int blen;

	private CSVReblockOOCInstruction(Operator op, CPOperand in, CPOperand out, int blocklength, String opcode,
		String instr) {
		super(OOCType.Reblock, op, in, out, opcode, instr);
		blen = blocklength;
	}

	public static CSVReblockOOCInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		String opcode = parts[0];
		if(!opcode.equals(Opcodes.CSVRBLK.toString()))
			throw new DMLRuntimeException("Incorrect opcode for CSVReblockOOCInstruction:" + opcode);

		CPOperand in = new CPOperand(parts[1]);
		CPOperand out = new CPOperand(parts[2]);
		int blen = Integer.parseInt(parts[3]);
		return new CSVReblockOOCInstruction(null, in, out, blen, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		MatrixObject min = ec.getMatrixObject(input1);
		DataCharacteristics mc = ec.getDataCharacteristics(input1.getName());
		DataCharacteristics mcOut = ec.getDataCharacteristics(output.getName());
		mcOut.set(mc.getRows(), mc.getCols(), blen, mc.getNonZeros());

		OOCStream<IndexedMatrixValue> qOut = createWritableStream();
		addOutStream(qOut);

		FileFormatProperties props = min.getFileFormatProperties();
		FileFormatPropertiesCSV csvProps = props instanceof FileFormatPropertiesCSV ? (FileFormatPropertiesCSV) props : new FileFormatPropertiesCSV();

		final Path path = new Path(min.getFileName());
		final JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());

		try {
			final FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			MatrixReader.checkValidInputFile(fs, path);

			final List<Path> files = collectInputFiles(fs, path);

			if(files.size() == 1) {
				submitOOCTask(() -> {
					readCSVBlock(qOut, files.get(0), csvProps);
					qOut.closeInput();
				}, qOut);
			}
			else {
				submitOOCTask(() -> {
					try(MultiFileBufferedSeekableInput in = new MultiFileBufferedSeekableInput(fs, files)) {
						readCSVBlock(qOut, in, csvProps);
					}
					catch(IOException ioe) {
						throw new DMLRuntimeException(ioe);
					}
					qOut.closeInput();
				}, qOut);
			}

			MatrixObject mout = ec.getMatrixObject(output);
			mout.setStreamHandle(qOut);
		}
		catch(IOException e) {
			throw new DMLRuntimeException(e);
		}
	}

	private static List<Path> collectInputFiles(FileSystem fs, Path path) throws IOException {
		if(!fs.getFileStatus(path).isDirectory())
			return Collections.singletonList(path);

		final List<Path> files = new ArrayList<>();
		for(FileStatus stat : fs.listStatus(path, IOUtilFunctions.hiddenFileFilter))
			files.add(stat.getPath());
		Collections.sort(files);
		return files;
	}

	private void readCSVBlock(OOCStream<IndexedMatrixValue> qOut, Path path, FileFormatPropertiesCSV props) {
		final JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());

		try {
			final FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			MatrixReader.checkValidInputFile(fs, path);

			if(props.getDelim().length() != 1)
				throw new DMLRuntimeException("Can only read CSVs with single char delimiters");

			try(FSDataInputStream rawIn = fs.open(path); BufferedSeekableInput in = new BufferedSeekableInput(rawIn)) {
				readCSVBlock(qOut, in, props);
			}
		}
		catch(IOException ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private void readCSVBlock(OOCStream<IndexedMatrixValue> qOut, SeekableInput in, FileFormatPropertiesCSV props)
		throws IOException {
		final int delim = props.getDelim().charAt(0);
		final ColumnInfo columnInfo = detectColumnInfo(in, delim, props.hasHeader());
		final int ncols = columnInfo.ncols;

		if(ncols <= 0)
			return;

		in.seek(columnInfo.dataStart);

		final int segLenMax = Math.min(MAX_BLOCKS_IN_CACHE * blen, ncols);
		final int segLen = Math.min(segLenMax, ncols);
		final boolean fill = props.isFill();
		final double fillValue = props.getFillValue();
		final Set<String> naStrings = props.getNAStrings();

		final List<Long> segStartList = new ArrayList<>();
		MatrixBlock[] firstBlocks = null;
		DenseBlock[] firstDense = null;
		int rowsInBand = 0;
		int brow = 0;
		int rowOffset = 0;

		while(true) {
			int next = peek(in);
			if(next == -1)
				break;

			if(rowsInBand == 0) {
				firstBlocks = allocateBlocks(blen, 0, segLen, ncols);
				firstDense = extractDense(firstBlocks);
			}

			final long nextSegPos = fillFirstSegmentRow(in, firstDense, rowsInBand, segLen, ncols, delim, fill,
				fillValue, naStrings);
			segStartList.add(nextSegPos);

			rowsInBand++;
			if(rowsInBand == blen) {
				emitBlocks(qOut, firstBlocks, rowsInBand, brow, 0);
				firstBlocks = null;
				firstDense = null;

				fillFollowingSegments(qOut, in, segStartList.subList(rowOffset, rowOffset + rowsInBand), segLen, ncols,
					brow, rowsInBand, blen, delim, fill, fillValue, naStrings);

				rowsInBand = 0;
				brow++;
				rowOffset += blen;
			}
		}

		// Process remainder
		if(rowsInBand > 0) {
			emitBlocks(qOut, firstBlocks, rowsInBand, brow, 0);
			fillFollowingSegments(qOut, in, segStartList.subList(rowOffset, rowOffset + rowsInBand), segLen, ncols,
				brow, rowsInBand, blen, delim, fill, fillValue, naStrings);
		}
	}

	private void fillFollowingSegments(OOCStream<IndexedMatrixValue> qOut, SeekableInput in, List<Long> segStartList,
		int segLen, int ncols, int brow, int rowsInBand, int blen, int delim, boolean fill, double fillValue,
		Set<String> naStrings) throws IOException {
		for(int c0 = segLen; c0 < ncols; c0 += segLen) {
			final int seg = Math.min(segLen, ncols - c0);
			final int firstBlockCol = c0 / blen;

			//final int rows = rowsInBand;
			final MatrixBlock[] blocks = allocateBlocks(rowsInBand, firstBlockCol, seg, ncols);
			final DenseBlock[] dense = extractDense(blocks);

			for(int i = 0; i < rowsInBand; i++) {
				in.seek(segStartList.get(i));
				int col = c0;
				int read = 0;

				while(read < seg) {
					final Token token = parseToken(in, delim, fill, fillValue, naStrings);
					final int bci = (col / blen) - firstBlockCol;
					final int within = col % blen;
					dense[bci].set(i, within, token.value);
					if(token.term == delim)
						in.read();
					read++;
					col++;
				}

				segStartList.set(i, in.getPos());
				skipRestOfLineFast(in);
			}

			emitBlocks(qOut, blocks, rowsInBand, brow, firstBlockCol);
		}
	}

	private static ColumnInfo detectColumnInfo(SeekableInput in, int delim, boolean hasHeader) throws IOException {
		in.seek(0);
		if(hasHeader)
			skipRestOfLineFast(in);

		final long dataStart = in.getPos();
		int ch;
		int ncols = 0;
		boolean seenToken = false;
		while((ch = in.read()) != -1) {
			if(ch == delim) {
				ncols++;
				seenToken = false;
			}
			else if(ch == '\n') {
				if(seenToken || ncols > 0)
					ncols++;
				break;
			}
			else if(ch == '\r') {
				if(consumeLF(in))
					ch = '\n';
				if(seenToken || ncols > 0)
					ncols++;
				break;
			}
			else {
				seenToken = true;
			}
		}

		if(ch == -1 && (seenToken || ncols > 0))
			ncols++;

		in.seek(dataStart);
		return new ColumnInfo(ncols, dataStart);
	}

	private long fillFirstSegmentRow(SeekableInput in, DenseBlock[] denseBlocks, int rowOffset, int segLen, int ncols,
		int delim, boolean fill, double fillValue, Set<String> naStrings) throws IOException {
		int col = 0;
		long nextSegPos = -1;

		while(col < segLen) {
			final Token token = parseToken(in, delim, fill, fillValue, naStrings);
			final int bci = col / blen;
			final int within = col % blen;
			denseBlocks[bci].set(rowOffset, within, token.value);
			col++;

			if(token.term == delim) {
				in.read();
				if(col == segLen)
					nextSegPos = in.getPos();
				continue;
			}

			nextSegPos = in.getPos();
			break;
		}

		if(nextSegPos < 0)
			nextSegPos = in.getPos();

		skipRestOfLineFast(in);
		return nextSegPos;
	}

	private MatrixBlock[] allocateBlocks(int rows, int firstBlockCol, int segLen, int ncols) {
		final int c0 = firstBlockCol * blen;
		final int lastBlockCol = (c0 + segLen - 1) / blen;
		final int numBlocks = lastBlockCol - firstBlockCol + 1;
		final MatrixBlock[] blocks = new MatrixBlock[numBlocks];

		for(int bci = 0; bci < numBlocks; bci++) {
			final int bcol = firstBlockCol + bci;
			final int cStart = bcol * blen;
			final int cEnd = Math.min(ncols, cStart + blen);
			final MatrixBlock block = new MatrixBlock(rows, cEnd - cStart, false);
			block.allocateDenseBlock();
			blocks[bci] = block;
		}

		return blocks;
	}

	private DenseBlock[] extractDense(MatrixBlock[] blocks) {
		final DenseBlock[] dense = new DenseBlock[blocks.length];
		for(int i = 0; i < blocks.length; i++) {
			final DenseBlock db = blocks[i].getDenseBlock();
			dense[i] = db;
		}
		return dense;
	}

	private void emitBlocks(OOCStream<IndexedMatrixValue> qOut, MatrixBlock[] blocks, int rowsInBand, int brow,
		int firstBlockCol) {
		for(int bci = 0; bci < blocks.length; bci++) {
			MatrixBlock block = blocks[bci];

			if(block.getNumRows() != rowsInBand)
				block = block.slice(0, rowsInBand - 1, 0, block.getNumColumns() - 1);

			block.recomputeNonZeros();
			block.examSparsity();
			final MatrixIndexes idx = new MatrixIndexes(brow + 1, firstBlockCol + bci + 1);
			qOut.enqueue(new IndexedMatrixValue(idx, block));
		}
	}

	private static Token parseToken(SeekableInput in, int delim, boolean fill, double fillValue, Set<String> naStrings)
		throws IOException {
		int ch;
		do {
			ch = in.read();
			if(ch == -1)
				throw new DMLRuntimeException("Unexpected EOF in CSV token");
		}
		while(ch == ' ' || ch == '\t');

		final StringBuilder buf = new StringBuilder(32);
		while(ch != -1 && ch != delim && ch != '\n' && ch != '\r') {
			buf.append((char) ch);
			ch = in.read();
		}
		if(ch != -1)
			in.seek(in.getPos() - 1);

		int len = buf.length();
		while(len > 0 && (buf.charAt(len - 1) == ' ' || buf.charAt(len - 1) == '\t'))
			buf.setLength(--len);

		final double value;
		if(len == 0) {
			if(fill)
				value = fillValue;
			else
				throw new DMLRuntimeException("Empty value in CSV input");
		}
		else {
			value = UtilFunctions.parseToDouble(buf.toString(), naStrings);
		}

		return new Token(value, ch);
	}

	private static void skipRestOfLineFast(SeekableInput in) throws IOException {
		int ch;
		while((ch = in.read()) != -1) {
			if(ch == '\n')
				return;
			if(ch == '\r') {
				consumeLF(in);
				return;
			}
		}
	}

	private static boolean consumeLF(SeekableInput in) throws IOException {
		final long pos = in.getPos();
		final int next = in.read();
		if(next == '\n')
			return true;
		if(next != -1)
			in.seek(pos);
		return false;
	}

	private static int peek(SeekableInput in) throws IOException {
		final long pos = in.getPos();
		final int ch = in.read();
		if(ch != -1)
			in.seek(pos);
		return ch;
	}

	private interface SeekableInput extends AutoCloseable {
		int read() throws IOException;

		void seek(long pos) throws IOException;

		long getPos();

		@Override
		void close() throws IOException;
	}

	private static final class MultiFileBufferedSeekableInput implements SeekableInput {
		private final FileSystem fs;
		private final List<Path> files;
		private final long[] offsets;
		private final long totalLength;
		private final BufferedSeekableInput[] streams;

		private int currentIdx;
		private BufferedSeekableInput current;
		private long position;

		private MultiFileBufferedSeekableInput(FileSystem fs, List<Path> files) throws IOException {
			if(files.isEmpty())
				throw new DMLRuntimeException("No CSV files to read");
			this.fs = fs;
			this.files = files;
			offsets = new long[files.size()];
			streams = new BufferedSeekableInput[files.size()];
			long offset = 0;
			for(int i = 0; i < files.size(); i++) {
				offsets[i] = offset;
				offset += fs.getFileStatus(files.get(i)).getLen();
			}
			totalLength = offset;
			currentIdx = 0;
			openIfNeeded(0);
			current = streams[0];
			position = 0;
		}

		private void openIfNeeded(int idx) throws IOException {
			if(idx >= files.size())
				return;

			if(streams[idx] == null)
				streams[idx] = new BufferedSeekableInput(fs.open(files.get(idx)));
		}

		@Override
		public int read() throws IOException {
			if(current == null)
				return -1;
			int b = current.read();
			while(b == -1 && currentIdx + 1 < files.size()) {
				currentIdx++;
				openIfNeeded(currentIdx);
				current = streams[currentIdx];
				current.seek(0);
				b = current.read();
			}
			if(b != -1)
				position++;
			return b;
		}

		@Override
		public void seek(long pos) throws IOException {
			if(pos < 0 || pos > totalLength)
				throw new IOException("Seek position out of range: " + pos);
			if(pos == totalLength) {
				currentIdx = files.size();
				current = null;
				position = pos;
				return;
			}
			int idx = findFile(pos);
			openIfNeeded(idx);
			currentIdx = idx;
			current = streams[idx];
			current.seek(pos - offsets[idx]);
			position = pos;
		}

		@Override
		public long getPos() {
			return position;
		}

		@Override
		public void close() throws IOException {
			for(BufferedSeekableInput stream : streams) {
				if(stream != null)
					stream.close();
			}
		}

		private int findFile(long pos) {
			for(int i = 0; i < offsets.length - 1; i++) {
				if(pos < offsets[i + 1])
					return i;
			}
			return offsets.length - 1;
		}
	}

	private static final class BufferedSeekableInput implements SeekableInput {
		private static final int BUF_SIZE = 4 * 1024;

		private final FSDataInputStream in;
		private final byte[] buf = new byte[BUF_SIZE];
		private long bufStart = 0;
		private int bufLen = 0;
		private int bufPos = 0;

		private BufferedSeekableInput(FSDataInputStream in) {
			this.in = in;
		}

		@Override
		public int read() throws IOException {
			if(bufPos >= bufLen) {
				if(!fill())
					return -1;
			}
			return buf[bufPos++] & 0xFF;
		}

		private boolean fill() throws IOException {
			bufStart = in.getPos();
			bufLen = in.read(buf, 0, BUF_SIZE);
			if(bufLen <= 0) {
				bufLen = 0;
				bufPos = 0;
				return false;
			}
			bufPos = 0;
			return true;
		}

		@Override
		public void seek(long pos) throws IOException {
			final long bufEnd = bufStart + bufLen;
			if(pos >= bufStart && pos < bufEnd) {
				bufPos = (int) (pos - bufStart);
			}
			else {
				in.seek(pos);
				bufStart = pos;
				bufLen = 0;
				bufPos = 0;
			}
		}

		@Override
		public long getPos() {
			return bufStart + bufPos;
		}

		@Override
		public void close() throws IOException {
			in.close();
		}
	}

	private static final class ColumnInfo {
		private final int ncols;
		private final long dataStart;

		private ColumnInfo(int ncols, long dataStart) {
			this.ncols = ncols;
			this.dataStart = dataStart;
		}
	}

	private static final class Token {
		private final double value;
		private final int term;

		private Token(double value, int term) {
			this.value = value;
			this.term = term;
		}
	}
}

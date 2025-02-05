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

package org.apache.sysds.runtime.io;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.cog.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class ReaderCOGParallel extends MatrixReader{
	protected final FileFormatPropertiesCOG _props;
	final private int _numThreads;

	public ReaderCOGParallel(FileFormatPropertiesCOG props) {
		_props = props;
		_numThreads = OptimizerUtils.getParallelBinaryReadParallelism();
	}
	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
		JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
		Path path = new Path(fname);
		FileSystem fs = IOUtilFunctions.getFileSystem(path, job);

		BufferedInputStream bis = new BufferedInputStream(fs.open(path));
		return readCOG(bis, estnnz);
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz) throws IOException, DMLRuntimeException {
		BufferedInputStream bis = new BufferedInputStream(is);
		return readCOG(bis, estnnz);
	}

	/**
	 * Reads a COG file from a BufferedInputStream.
	 * Not handling number of columns or rows, as this can be inferred from the data, but
	 * may be used in the future for validation or possibly removed as a requirement for COG.
	 * Specific to COG files, normal TIFFs will break because they aren't tiled, only
	 * tiled data is supported.
	 * @param bis
	 * @return
	 */
	private MatrixBlock readCOG(BufferedInputStream bis, long estnnz) throws IOException {
		COGByteReader byteReader = new COGByteReader(bis);
		COGHeader cogHeader = COGHeader.readCOGHeader(byteReader);

		// Check compatibility of the file with our reader
		// Certain options are not supported, and we need to filter out some non-standard options
		String isCompatible = COGHeader.isCompatible(cogHeader.getIFD());
		if (!isCompatible.equals("")) {
			throw new DMLRuntimeException("Incompatible COG file: " + isCompatible);
		}

		// TODO: Currently only reads the first image which is the full resolution image
		// In the future, this could be extended to read the overviews as well
		// But keep in mind that we are only returning a single MatrixBlock, so there needs to be some special handling
		COGProperties cogP = new COGProperties(cogHeader.getIFD());

		// number of tiles for Width and Length
		int tileCols = cogP.getCols() / cogP.getTileWidth();
		int tileRows = cogP.getRows() / cogP.getTileLength();

		// total number of tiles if every tile contains all bands
		int calculatedAmountTiles = tileCols * tileRows;
		// actual given number of tiles, longer for PlanarConfiguration=2
		int actualAmountTiles = cogP.getTileOffsets().length;

		int currentTileCol = 0;
		int currentTileRow = 0;
		int currentBand = 0;

		ExecutorService pool = CommonThreadPool.get(_numThreads);

		MatrixBlock outputMatrix = createOutputMatrixBlock(cogP.getRows(), cogP.getCols() * cogP.getBands(), cogP.getRows(), estnnz, false, true);

		// Check if the tiles are fully sequential (always starting at a higher byte offset)
		// If that is the case, we can skip the mark/reset calls and avoid buffering large amounts of data
		boolean tilesFullySequential = cogP.tilesFullySequential();

		try {
			ArrayList<Callable<MatrixBlock>> tasks = new ArrayList<>();
			// Principle: We're reading all tiles in sequence as I/O likely won't benefit from parallel reads
			// in most cases.
			// Then: Process the read tile data in parallel
			for (int currenTileIdx = 0; currenTileIdx < actualAmountTiles; currenTileIdx++) {
				// First read the bytes for the new tile
				long bytesToRead = (cogP.getTileOffsets()[currenTileIdx] - byteReader.getTotalBytesRead()) + cogP.getBytesPerTile()[currenTileIdx];
				// Only necessary if we might need to jump back in the stream (when tiles are not fully sequential)
				if (!tilesFullySequential) {
					byteReader.mark(bytesToRead);
				}
				byteReader.skipBytes(cogP.getTileOffsets()[currenTileIdx] - byteReader.getTotalBytesRead());
				byte[] currentTileData = byteReader.readBytes(cogP.getBytesPerTile()[currenTileIdx]);

				if (!tilesFullySequential) {
					byteReader.reset();
				}

				if (cogP.getCompression() == 8) {
					currentTileData = COGCompressionUtils.decompressDeflate(currentTileData);
				}

				TileProcessor tileProcessor;
				if (cogP.getPlanarConfiguration() == 1) {
					// Every band is in the same tile, e.g. RGBRGBRGB
					tileProcessor = new TileProcessor(cogP.getCols() * cogP.getBands(), currentTileData, currentTileRow, currentTileCol,
							cogP.getTileWidth(), cogP.getTileLength(), cogP.getBands(), cogP.getBitsPerSample(), cogP.getSampleFormat(), cogHeader, outputMatrix,
							cogP.getPlanarConfiguration());

					currentTileCol++;
					if (currentTileCol >= tileCols) {
						currentTileCol = 0;
						currentTileRow++;
					}
				} else if (cogP.getPlanarConfiguration() == 2) {
					// Every band is in a different tile, e.g. RRRGGGBBB
					// Note here that first all tiles from a single band are present
					// after that all tiles from the next band are present and so on (so they don't interleave)
					if (currenTileIdx - (currentBand * calculatedAmountTiles) >= calculatedAmountTiles) {
						currentTileCol = 0;
						currentTileRow = 0;
						currentBand++;
					}

					tileProcessor = new TileProcessor(cogP.getCols() * cogP.getBands(), currentTileData, currentTileRow, currentTileCol,
							cogP.getTileWidth(), cogP.getTileLength(), cogP.getBands(), cogP.getBitsPerSample(), cogP.getSampleFormat(), cogHeader, outputMatrix,
							cogP.getPlanarConfiguration(), currentBand);

					currentTileCol++;

					if (currentTileCol >= tileCols) {
						currentTileCol = 0;
						currentTileRow++;
					}
				} else {
					throw new DMLRuntimeException("Unsupported Planar Configuration: " + cogP.getPlanarConfiguration());
				}
				tasks.add(tileProcessor);
			}

			try {
				for (Future<MatrixBlock> result : pool.invokeAll(tasks)) {
					result.get();
				}

				if (outputMatrix.isInSparseFormat()) {
					sortSparseRowsParallel(outputMatrix, cogP.getRows(), _numThreads, pool);
				}
			} catch (Exception e) {
				throw new IOException("Error during parallel task execution.", e);
			}

		} catch (IOException e) {
			throw new IOException("Thread pool issue or file reading error.", e);
		} finally {
			pool.shutdown();
		}

		// TODO: If the tile is compressed, decompress the currentTileData here

		outputMatrix.examSparsity();
		return outputMatrix;
	}

	public class TileProcessor implements Callable<MatrixBlock> {

		private final int clen;
		private final byte[] tileData;
		private final int tileRow;
		private final int tileCol;
		private final int tileWidth;
		private final int tileLength;
		private final int bands;
		private final int[] bitsPerSample;
		private final SampleFormatDataTypes[] sampleFormat;
		private final COGHeader cogHeader;
		private final MatrixBlock _dest;
		private final int planarConfiguration;
		private final boolean sparse;
		private final int band;

		public TileProcessor(int clen, byte[] tileData, int tileRow, int tileCol, int tileWidth, int tileLength,
							 int bands, int[] bitsPerSample, SampleFormatDataTypes[] sampleFormat, COGHeader cogHeader,
							 MatrixBlock dest, int planarConfiguration) {
			this(clen, tileData, tileRow, tileCol, tileWidth, tileLength, bands, bitsPerSample, sampleFormat,
					cogHeader, dest, planarConfiguration, 0);
		}

		public TileProcessor(int clen, byte[] tileData, int tileRow, int tileCol, int tileWidth, int tileLength,
							 int bands, int[] bitsPerSample, SampleFormatDataTypes[] sampleFormat, COGHeader cogHeader,
							 MatrixBlock dest, int planarConfiguration, int band) {
			this.clen = clen;
			this.tileData = tileData;
			this.tileRow = tileRow;
			this.tileCol = tileCol;
			this.tileWidth = tileWidth;
			this.tileLength = tileLength;
			this.bands = bands;
			this.bitsPerSample = bitsPerSample;
			this.sampleFormat = sampleFormat;
			this.cogHeader = cogHeader;
			this._dest = dest;
			this.planarConfiguration = planarConfiguration;
			this.sparse = _dest.isInSparseFormat();
			this.band = band;
		}


		@Override
		public MatrixBlock call() throws Exception {
			if (planarConfiguration==1) {
				processTileByPixel();
			}
			else if (planarConfiguration==2){
				processTileByBand();
			}
			else{
				throw new DMLRuntimeException("Unsupported Planar Configuration: " + planarConfiguration);
			}
			return _dest;
		}

		private void processTileByPixel() {
			int pixelsRead = 0;
			int bytesRead = 0;
			int currentRow = 0;

			MatrixBlock tileMatrix = new MatrixBlock(tileLength, tileWidth*bands, sparse);

			if(sparse) {
				tileMatrix.allocateAndResetSparseBlock(true, SparseBlock.Type.CSR);
				tileMatrix.getSparseBlock().allocate(0,  tileLength*tileWidth*bands);
			}

			while (currentRow < tileLength && pixelsRead < tileWidth) {
				for (int bandIdx = 0; bandIdx < bands; bandIdx++) {
					double value = 0;
					int sampleLength = bitsPerSample[bandIdx] / 8;

					switch (sampleFormat[bandIdx]) {
						case UNSIGNED_INTEGER:
						case UNDEFINED:
							value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, false, false).doubleValue();
							break;
						case SIGNED_INTEGER:
							value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, true, false).doubleValue();
							break;
						case FLOATING_POINT:
							value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, true, false, false).doubleValue();
							break;
					}

					bytesRead += sampleLength;
					tileMatrix.set(currentRow, (pixelsRead * bands) + bandIdx, value);
				}

				pixelsRead++;
				if (pixelsRead >= tileWidth) {
					pixelsRead = 0;
					currentRow++;
				}
			}

			try {
				int rowOffset = tileRow * tileLength;
				int colOffset = tileCol * tileWidth * bands;
				if (sparse) {
					// if outputMatrix is sparse apply synchronisation if tiles are more narrow then outputMatrix
					insertIntoSparse(_dest, tileMatrix, rowOffset, colOffset);
				}
				else {
					// if matrix is dense inserting just the tileMatrix as is
					_dest.copy(rowOffset, rowOffset + tileLength - 1,
							colOffset, colOffset + (tileWidth * bands) -1,
							tileMatrix, false);
				}
			} catch (RuntimeException e) {
				throw new DMLRuntimeException("Error while processing tile", e);
			}
		}

		private void processTileByBand() {
			int pixelsRead = 0;
			int bytesRead = 0;
			int currentRow = 0;

			MatrixBlock tileMatrix = new MatrixBlock(tileLength, tileWidth*bands, sparse);

			if(sparse) {
				tileMatrix.allocateAndResetSparseBlock(true, SparseBlock.Type.CSR);
				tileMatrix.getSparseBlock().allocate(0,  tileLength*tileWidth*bands);
			}

			while (currentRow < tileLength && pixelsRead < tileWidth) {
				double value = 0;
				int sampleLength = bitsPerSample[band] / 8;

				switch (sampleFormat[band]) {
					case UNSIGNED_INTEGER:
					case UNDEFINED:
						value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, false, false).doubleValue();
						break;
					case SIGNED_INTEGER:
						value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, false, true, false).doubleValue();
						break;
					case FLOATING_POINT:
						value = cogHeader.parseByteArray(tileData, sampleLength, bytesRead, true, false, false).doubleValue();
						break;
				}

				bytesRead += sampleLength;
				tileMatrix.set(currentRow, (pixelsRead * bands) + band, value);

				pixelsRead++;
				if (pixelsRead >= tileWidth) {
					pixelsRead = 0;
					currentRow++;
				}
			}

			try {
				int rowOffset = tileRow * tileLength;
				int colOffset = tileCol * tileWidth * bands;
				if (sparse) {
					// if outputMatrix is sparse apply synchronisation if tiles are more narrow then outputMatrix
					insertIntoSparse(_dest, tileMatrix, rowOffset, colOffset);
				}
				else {
					// insert only values the thread is responsible for
					// denseBlocks have zero values by default, so actual current band 0 values dont need to be written
					for (int i = 0; i < tileLength; i++) {
						for (int j = 0; j < tileWidth * bands; j++) {
							if (tileMatrix.get(i, j) != 0) {
								_dest.set(rowOffset + i, colOffset + j, tileMatrix.get(i, j));
							}
						}
					}
				}
			} catch (RuntimeException e) {
				throw new DMLRuntimeException("Error while processing tile", e);
			}
		}

		private void insertIntoSparse(MatrixBlock _dest, MatrixBlock tileMatrix, int rowOffset, int colOffset ) {
			SparseBlock sblock = _dest.getSparseBlock();
			if (tileWidth < clen) {
				// if there is more then one tile in horizontal direction, synchronization is needed
				// such that threads do not write the same rows concurrently
				// appendToSparse and appendRowToSparse require sorting
				if (sblock instanceof SparseBlockMCSR && sblock.get(rowOffset) != null) {
					for (int i = 0; i < tileLength; i++)
						synchronized (sblock.get(rowOffset + i)) {
							_dest.appendRowToSparse(sblock, tileMatrix, i,
									rowOffset,
									colOffset, true);
						}
				}
				else{
					synchronized (_dest) {
						_dest.appendToSparse(
								tileMatrix,
								rowOffset,
								colOffset);
					}
				}
			}
			else {
				// otherwise no further synchronization is needed
				_dest.appendToSparse(tileMatrix, rowOffset, colOffset);
			}
		}
	}
}

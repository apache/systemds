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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.cog.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

public class ReaderCOG extends MatrixReader{
	protected final FileFormatPropertiesCOG _props;

	public ReaderCOG(FileFormatPropertiesCOG props) {
		_props = props;
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
		// TODO: Is the metadata (e.g. the coordinates) necessary in SystemDS? Currently not possible as we only return a MatrixBlock
		// However, this could possibly be changed in the future to somehow also store relevant metadata if desired.
		// Currently this implementation reads the most important data from the header.
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

		// Check if the tiles are fully sequential (always starting at a higher byte offset)
		// If that is the case, we can skip the mark/reset calls and avoid buffering large amounts of data
		boolean tilesFullySequential = cogP.tilesFullySequential();

		MatrixBlock outputMatrix = createOutputMatrixBlock(cogP.getRows(), cogP.getCols() * cogP.getBands(), cogP.getRows(), estnnz, true, false);

		for (int currenTileIdx = 0; currenTileIdx < actualAmountTiles; currenTileIdx++) {
			long bytesToRead = (cogP.getTileOffsets()[currenTileIdx] - byteReader.getTotalBytesRead()) + cogP.getBytesPerTile()[currenTileIdx];
			// Mark the current position in the stream
			// This is used to reset the stream to this position after reading the data
			// Valid until bytesToRead + 1 bytes are read
			// Only necessary if we might need to jump back in the stream (when tiles are not fully sequential)
			if (!tilesFullySequential) {
				byteReader.mark(bytesToRead);
			}
			// Read until offset is reached
			byteReader.skipBytes(cogP.getTileOffsets()[currenTileIdx] - byteReader.getTotalBytesRead());
			byte[] currentTileData = byteReader.readBytes(cogP.getBytesPerTile()[currenTileIdx]);

			if (!tilesFullySequential) {
				byteReader.reset();
			}

			if (cogP.getCompression() == 8) {
				currentTileData = COGCompressionUtils.decompressDeflate(currentTileData);
			}

			int pixelsRead = 0;
			int bytesRead = 0;
			int currentRow = 0;
			if (cogP.getPlanarConfiguration() == 1) {
				// Interleaved
				// RGBRGBRGB
				while (currentRow < cogP.getTileLength() && pixelsRead < cogP.getTileWidth()) {
					for (int bandIdx = 0; bandIdx < cogP.getBands(); bandIdx++) {
						double value = 0;
						int sampleLength = cogP.getBitsPerSample()[bandIdx] / 8;

						switch (cogP.getSampleFormat()[bandIdx]) {
							case UNSIGNED_INTEGER:
							case UNDEFINED:
								// According to the standard, this should be handled as not being there -> 1 (unsigned integer)
								value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, false, false).doubleValue();
								break;
							case SIGNED_INTEGER:
								value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, true, false).doubleValue();
								break;
							case FLOATING_POINT:
								value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, true, false, false).doubleValue();
								break;
						}

						bytesRead += sampleLength;
						outputMatrix.set((currentTileRow * cogP.getTileLength()) + currentRow,
								(currentTileCol * cogP.getTileWidth() * cogP.getBands()) + (pixelsRead * cogP.getBands()) + bandIdx,
								value);
					}

					pixelsRead++;
					if (pixelsRead >= cogP.getTileWidth()) {
						pixelsRead = 0;
						currentRow++;
					}
				}
			} else if (cogP.getPlanarConfiguration() == 2 && calculatedAmountTiles * cogP.getBands() == cogP.getTileOffsets().length) {
				// If every band is stored in different tiles, so first one R, second one G and so on
				// RRRGGGBBB
				// TODO: Currently this doesn't seem standardized properly, there are still open GitHub issues about that
				// e.g.: https://github.com/cogeotiff/cog-spec/issues/17
				// if something changes in the standard, this may need to be adjusted, interleaved is discouraged in COG though
				if (currenTileIdx - (currentBand * calculatedAmountTiles) >= calculatedAmountTiles) {
					currentTileCol = 0;
					currentTileRow = 0;
					currentBand++;
				}

				int sampleLength = cogP.getBitsPerSample()[currentBand] / 8;

				while (currentRow < cogP.getTileLength() && pixelsRead < cogP.getTileWidth()) {
					double value = 0;

					switch (cogP.getSampleFormat()[currentBand]) {
						case UNSIGNED_INTEGER:
						case UNDEFINED:
							// According to the standard, this should be handled as not being there -> 1 (unsigned integer)
							value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, false, false).doubleValue();
							break;
						case SIGNED_INTEGER:
							value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, false, true, false).doubleValue();
							break;
						case FLOATING_POINT:
							value = cogHeader.parseByteArray(currentTileData, sampleLength, bytesRead, true, false, false).doubleValue();
							break;
					}
					bytesRead += sampleLength;
					outputMatrix.set((currentTileRow * cogP.getTileLength()) + currentRow,
							(currentTileCol * cogP.getTileWidth() * cogP.getBands()) + (pixelsRead * cogP.getBands()) + currentBand,
							value);
					pixelsRead++;
					if (pixelsRead >= cogP.getTileWidth()) {
						pixelsRead = 0;
						currentRow++;
					}
				}
			} else {
				throw new DMLRuntimeException("Unsupported Planar Configuration: " + cogP.getPlanarConfiguration());
			}

			currentTileCol++;
			if (currentTileCol >= tileCols) {
				currentTileCol = 0;
				currentTileRow++;
			}
		}

		outputMatrix.examSparsity();
		return outputMatrix;
	}
}

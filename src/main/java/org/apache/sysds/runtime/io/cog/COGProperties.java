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

package org.apache.sysds.runtime.io.cog;

import java.util.Arrays;

/**
 * Properties of a COG file that are useful for reading the file.
 */
public class COGProperties {

	private int rows;
	private int cols;
	private int bands;
	private int[] bitsPerSample;
	private SampleFormatDataTypes[] sampleFormat;
	private int planarConfiguration;
	private int tileWidth;
	private int tileLength;
	private int[] tileOffsets;
	private int[] bytesPerTile;
	private int compression;

	public COGProperties() {

	}

	public COGProperties(IFDTag[] ifdTags) {
		this.initFromIFDTags(ifdTags);
	}

	// Getters and Setters
	public int getRows() {
		return rows;
	}

	public void setRows(int rows) {
		this.rows = rows;
	}

	public int getCols() {
		return cols;
	}

	public void setCols(int cols) {
		this.cols = cols;
	}

	public int getBands() {
		return bands;
	}

	public void setBands(int bands) {
		this.bands = bands;
	}

	public int[] getBitsPerSample() {
		return bitsPerSample;
	}

	public void setBitsPerSample(int[] bitsPerSample) {
		this.bitsPerSample = bitsPerSample;
	}

	public SampleFormatDataTypes[] getSampleFormat() {
		return sampleFormat;
	}

	public void setSampleFormat(SampleFormatDataTypes[] sampleFormat) {
		this.sampleFormat = sampleFormat;
	}

	public int getPlanarConfiguration() {
		return planarConfiguration;
	}

	public void setPlanarConfiguration(int planarConfiguration) {
		this.planarConfiguration = planarConfiguration;
	}

	public int getTileWidth() {
		return tileWidth;
	}

	public void setTileWidth(int tileWidth) {
		this.tileWidth = tileWidth;
	}

	public int getTileLength() {
		return tileLength;
	}

	public void setTileLength(int tileLength) {
		this.tileLength = tileLength;
	}

	public int[] getTileOffsets() {
		return tileOffsets;
	}

	public void setTileOffsets(int[] tileOffsets) {
		this.tileOffsets = tileOffsets;
	}

	public int[] getBytesPerTile() {
		return bytesPerTile;
	}

	public void setBytesPerTile(int[] bytesPerTile) {
		this.bytesPerTile = bytesPerTile;
	}

	public int getCompression() {
		return compression;
	}

	public void setCompression(int compression) {
		this.compression = compression;
	}

	public void initFromIFDTags(IFDTag[] ifdTags) {
		for (IFDTag ifd : ifdTags) {
			IFDTagDictionary tag = ifd.getTagId();
			switch (tag) {
				case ImageWidth:
					this.cols = ifd.getData()[0].intValue();
					break;
				case ImageLength:
					this.rows = ifd.getData()[0].intValue();
					break;
				case SamplesPerPixel:
					this.bands = ifd.getData()[0].intValue();
					break;
				case BitsPerSample:
					this.bitsPerSample = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
					break;
				case TileWidth:
					this.tileWidth = ifd.getData()[0].intValue();
					break;
				case TileLength:
					this.tileLength = ifd.getData()[0].intValue();
					break;
				case TileOffsets:
					this.tileOffsets = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
					break;
				case TileByteCounts:
					if (ifd.getData() != null) {
						this.bytesPerTile = Arrays.stream(ifd.getData()).mapToInt(Number::intValue).toArray();
					} else {
						this.bytesPerTile = new int[this.tileOffsets.length];
						for (int tile = 0; tile < this.tileOffsets.length; tile++) {
							int bits = 0;
							for (int band = 0; band < this.bands; band++) {
								bits += this.bitsPerSample[band];
							}
							this.bytesPerTile[tile] = this.tileWidth * this.tileLength * (bits / 8);
						}
					}
					break;
				case SampleFormat:
					int dataCount = ifd.getDataCount();
					this.sampleFormat = new SampleFormatDataTypes[dataCount];
					for (int i = 0; i < dataCount; i++) {
						this.sampleFormat[i] = SampleFormatDataTypes.valueOf(ifd.getData()[i].intValue());
					}
					break;
				case PlanarConfiguration:
					this.planarConfiguration = ifd.getData()[0].intValue();
					break;
				case Compression:
					this.compression = ifd.getData()[0].intValue();
					break;
				default:
					break;
			}
		}
	}

	public boolean tilesFullySequential() {
		boolean tilesFullySequential = true;
		for (int i = 1; i < getTileOffsets().length; i++) {
			if (getTileOffsets()[i] < getTileOffsets()[i - 1]) {
				tilesFullySequential = false;
				break;
			}
		}
		return tilesFullySequential;
	}
}

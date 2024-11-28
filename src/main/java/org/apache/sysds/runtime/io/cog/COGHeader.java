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

import org.apache.sysds.runtime.DMLRuntimeException;

import java.util.ArrayList;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Represents a header for a COG file. This includes IFDs, endianess etc.
 */
public class COGHeader {
	private boolean isLittleEndian;
	private String GDALMetadata;
	private IFDTag[] IFD;
	private boolean isBigTIFF;
	// Do we even need this or will we throw it away?
	// If we keep this and write it again, we also need to write the additional images
	// So this could very likely not make the cut
	private ArrayList<IFDTag[]> additionalIFDs;

	public COGHeader(boolean isLittleEndian) {
		this.isLittleEndian = isLittleEndian;
		GDALMetadata = "";
		additionalIFDs = new ArrayList<IFDTag[]>();
	}

	public void setIFD(IFDTag[] IFD) {
		this.IFD = IFD;
	}

	public IFDTag[] getIFD() {
		return IFD;
	}

	public void addAdditionalIFD(IFDTag[] IFD) {
		additionalIFDs.add(IFD);
	}

	public ArrayList<IFDTag[]> getAdditionalIFDs() {
		return additionalIFDs;
	}

	public IFDTag[] getSingleAdditionalIFD(int index) {
		return additionalIFDs.get(index);
	}

	public void setSingleAdditionalIFD(int index, IFDTag[] IFD) {
		additionalIFDs.set(index, IFD);
	}

	public void removeSingleAdditionalIFD(int index) {
		additionalIFDs.remove(index);
	}

	public void setLittleEndian(boolean isLittleEndian) {
		this.isLittleEndian = isLittleEndian;
	}

	public boolean isLittleEndian() {
		return isLittleEndian;
	}

	public void setGDALMetadata(String GDALMetadata) {
		this.GDALMetadata = GDALMetadata;
	}

	public String getGDALMetadata() {
		return GDALMetadata;
	}

	public void setBigTIFF(boolean isBigTIFF) {
		this.isBigTIFF = isBigTIFF;
	}

	public boolean isBigTIFF() {
		return isBigTIFF;
	}

	/**
	 * Parses a byte array into a generic number. Can be byte, short, int, float or double
	 * depending on the options given. E.g.: Use .doubleValue() on the result to get a double value easily
	 *
	 * Supported lengths:
	 * isDecimal:
	 * - 4 bytes: float
	 * - 8 bytes: double
	 * otherwise:
	 * - 1 byte: byte
	 * - 2 bytes: short
	 * - 4 bytes: int
	 * Anything else will throw an exception
	 * @param bytes  ???
	 * @param length number of bytes that should be read
	 * @param offset from the start of the byte array
	 * @param isDecimal Whether we are dealing with a floating point number
	 * @param isSigned Whether the number is signed
	 * @param isRational Whether the number is a rational number as specified in the TIFF standard
	 *				   (first 32 bit integer numerator of a fraction, second 32 bit integer denominator)
	 * @return  ???
	 */
	public Number parseByteArray(byte[] bytes, int length, int offset, boolean isDecimal, boolean isSigned, boolean isRational) {
		ByteBuffer buffer = ByteBuffer.wrap(bytes);
		buffer.order(isLittleEndian ? ByteOrder.LITTLE_ENDIAN : ByteOrder.BIG_ENDIAN);
		buffer.position(offset);

		if (isRational && !isSigned) {
			long numerator = Integer.toUnsignedLong(buffer.getInt());
			long denominator = Integer.toUnsignedLong(buffer.getInt());
			return (double)numerator / denominator;
		}
		if (isRational && isSigned) {
			long numerator = buffer.getInt();
			long denominator = buffer.getInt();
			return (double)numerator / denominator;
		}
		if (isDecimal) {
			switch (length) {
				case 4:
					return buffer.getFloat();
				case 8:
					return buffer.getDouble();
				default:
					throw new IllegalArgumentException("Unsupported length: " + length);
			}
		}
		switch (length) {
			case 1:
				return isSigned ? (byte)buffer.get() : Byte.toUnsignedInt(buffer.get());
			case 2:
				return isSigned ? (short)buffer.getShort() : Short.toUnsignedInt(buffer.getShort());
			case 4:
				return isSigned ? (int)buffer.getInt() : Integer.toUnsignedLong(buffer.getInt());
			case 8:
				return isSigned ? (long)buffer.getLong() : buffer.getLong();
			default:
				throw new IllegalArgumentException("Unsupported length: " + length);
		}
	}

	/**
	 * Prepares the COG header by reading the first 4 bytes and determining the byte order.
	 * Needs to be called before anything else is done with the COG header.
	 * @param byteReader  ???
	 * @return  ???
	 */
	private static COGHeader prepareHeader(COGByteReader byteReader) {
		// Read first 4 bytes to determine byte order and make sure it is a valid TIFF
		byte[] header = byteReader.readBytes(4);

		// Read the byte order
		boolean littleEndian = false;
		if ((header[0] & 0xFF) == 0x4D && (header[1] & 0xFF) == 0x4D) {
			littleEndian = false;
		} else if ((header[0] & 0xFF) == 0x49 && (header[1] & 0xFF) == 0x49) {
			littleEndian = true;
		} else {
			throw new DMLRuntimeException("Invalid Byte-Order");
		}

		// Create COGHeader object, initialize with the correct byte order
		COGHeader cogHeader = new COGHeader(littleEndian);

		// Check magic number (42), otherwise this is not a valid TIFF
		int magic = cogHeader.parseByteArray(header, 2, 2, false, false, false).intValue();
		if (magic == 42) {
			cogHeader.setBigTIFF(false);
		} else if (magic == 43) {
			cogHeader.setBigTIFF(true);
		} else {
			throw new DMLRuntimeException("Invalid Magic Number");
		}

		return cogHeader;
	}

	/**
	 * Reads the COG header from the BufferedInputStream.
	 * Handles little endian setting, checking magic number. After this you manually
	 * have to check the compatibility though if you desire to do so.
	 * @param byteReader  ???
	 * @return filled COGHeader object
	 */
	public static COGHeader readCOGHeader(COGByteReader byteReader) {
		COGHeader cogHeader = prepareHeader(byteReader);
		// Read offset of the first IFD
		// Usually this is 8 (right after the header) we are at right now
		// With COG, GDAL usually writes some metadata before the IFD
		short ifdOffsetSize = 4;
		// BigTIFF allows for differently sized offsets
		if (cogHeader.isBigTIFF()) {
			byte[] offsetSize = byteReader.readBytes(2);
			ifdOffsetSize = cogHeader.parseByteArray(offsetSize, 2, 0, false, false, false).shortValue();
			byteReader.skipBytes(2); // Skip the next 2 bytes
		}

		byte[] ifdOffsetRaw = byteReader.readBytes(ifdOffsetSize);
		long ifdOffset = cogHeader.parseByteArray(ifdOffsetRaw, ifdOffsetSize, 0, false, false, false).intValue();

		// If the IFD offset is larger than 8, read that and store it in the COGHeader
		// This is the GDAL metadata
		if (ifdOffset > 8) {
			// Read the metadata from the current position to the IFD offset
			// -8 because the offset is calculated from the beginning of the file
			byte[] metadata = byteReader.readBytes(ifdOffset - (cogHeader.isBigTIFF() ? 16 : 8));
			cogHeader.setGDALMetadata(new String(metadata));
		}

		// If we read the first IFD, we handle it somewhat differently
		// See the if-statement below
		boolean firstIFD = true;
		// Is used at the end of the while loop to determine if there is another IFD
		byte[] nextIFDOffsetRaw;
		int nextIFDOffset = 0;

		// Used in the beginning of the while loop to read the number of tags in the IFD
		byte[] numberOfTagsRaw;
		int numberOfTags;
		// Array to store the IFD tags, initialized after the number of tags were read
		IFDTag[] ifdTags;
		int tagCountLength = cogHeader.isBigTIFF() ? 8 : 4;
		int tagDataLength = cogHeader.isBigTIFF() ? 8 : 4;

		// Read the IFDs, always read the first one
		// The nextIFDOffset ist 0 if there is no next IFD
		while (nextIFDOffset != 0 || firstIFD) {
			// There can be data in-between IFDs, we need to skip that
			// Read until the next IFD, discard any data until then
			byteReader.skipBytes(nextIFDOffset - (firstIFD ? 0 : byteReader.getTotalBytesRead()));

			// Read the number of tags in the IFD and initialize the array
			numberOfTagsRaw = byteReader.readBytes(cogHeader.isBigTIFF() ? 8 : 2);
			numberOfTags = cogHeader.parseByteArray(numberOfTagsRaw, cogHeader.isBigTIFF() ? 8 : 2, 0, false, false, false).intValue();
			ifdTags = new IFDTag[numberOfTags];

			// Read the tags
			for (int i = 0; i < numberOfTags; i++) {
				// Read the tag (Normal 12 bytes, 20 bytes for BigTIFF)
				// 2 bytes tag ID
				// 2 bytes data type
				// 4 bytes data count (8 bytes BigTIFF)
				// 4 bytes data value (can also be offset) (8 bytes BigTIFF)
				byte[] tag = byteReader.readBytes(cogHeader.isBigTIFF() ? 20 : 12);
				int tagId = cogHeader.parseByteArray(tag, 2, 0, false, false, false).intValue();

				int tagType = cogHeader.parseByteArray(tag, 2, 2, false, false, false).intValue();
				TIFFDataTypes dataType = TIFFDataTypes.valueOf(tagType);

				int tagCount = cogHeader.parseByteArray(tag, tagCountLength, 4, false, false, false).intValue();

				Number[] tagData;
				long tagValue = cogHeader.parseByteArray(tag, tagDataLength, cogHeader.isBigTIFF() ? 12 : 8, false, false, false).longValue();

				if (dataType.getSize() * tagCount <= tagDataLength) {
					tagData = parseTagData(tagCount, tag, dataType, cogHeader, cogHeader.isBigTIFF() ? 8 : 4, cogHeader.isBigTIFF() ? 12 : 8);
				} else {
					// If the data in total is larger than 4 bytes it is an offset to the actual data
					// Read the data from the offset
					// tagValue = offset, just assigning this for better readability
					long offset = tagValue;
					// data length = tagCount * data type size
					int totalSize = tagCount * dataType.getSize();

					// Calculate the number of bytes to read in order to reset our reader
					// after going to that offset
					long bytesToRead = (offset - byteReader.getTotalBytesRead()) + totalSize;

					// Mark the current position in the stream
					// This is used to reset the stream to this position after reading the data
					// Valid until bytesToRead + 1 bytes are read
					byteReader.mark((int) bytesToRead);
					// Read until offset is reached
					byteReader.readBytes(offset - byteReader.getTotalBytesRead());
					// Read actual data
					byte[] data = byteReader.readBytes(totalSize);

					tagData = parseTagData(tagCount, data, dataType, cogHeader, 0);

					// Reset the stream to the beginning of the next tag
					byteReader.reset();
				}
				// Read the tag ID and get the corresponding tag from the dictionary (enum)
				IFDTagDictionary tagDictionary = IFDTagDictionary.valueOf(tagId);

				// Create the constructed IFDTag object and store it in the array
				IFDTag ifdTag = new IFDTag(tagDictionary != null ? tagDictionary : IFDTagDictionary.Unknown, (short) tagType, tagCount, tagData);
				ifdTags[i] = ifdTag;
			}
			if (firstIFD) {
				// If this is the first IFD, set it as the main IFD in the COGHeader
				cogHeader.setIFD(ifdTags.clone());
				firstIFD = false;
			} else {
				// If this is not the first IFD, add it as an additional IFD
				cogHeader.addAdditionalIFD(ifdTags.clone());
			}
			// Read the offset to the next IFD. If it is 0, there is no next IFD
			nextIFDOffsetRaw = byteReader.readBytes(4);
			nextIFDOffset = cogHeader.parseByteArray(nextIFDOffsetRaw, 4, 0, false, false, false).intValue();
		}
		return cogHeader;
	}

	/**
	 * Parses the data of an IFD entry from a byte array. Can throw an error if something is not expected,
	 * e.g. when a broken TIFF causes the data size to differ from what is expected.
	 * @param tagCount Number of tags that should be present
	 * @param rawData Raw data where the tags can be found
	 * @param dataType Data Type, used for size calculation
	 * @param cogHeader COGHeader is used for properly parsing the byte array with the correct data type etc.
	 * @param maxSize Should be set to 0 if no other value is useful! Throws an error when the data is too large for the header field
	 * @return  ???
	 */
	private static Number[] parseTagData(int tagCount, byte[] rawData, TIFFDataTypes dataType, COGHeader cogHeader, int maxSize) {
		return parseTagData(tagCount, rawData, dataType, cogHeader, maxSize, 0);
	}

	/**
	 * Parses the data of an IFD entry from a byte array. Can throw an error if something is not expected,
	 * e.g. when a broken TIFF causes the data size to differ from what is expected.
	 * @param tagCount Number of tags that should be present
	 * @param rawData Raw data where the tags can be found
	 * @param dataType Data Type, used for size calculation
	 * @param cogHeader COGHeader is used for properly parsing the byte array with the correct data type etc.
	 * @param maxSize Should be set to 0 if no other value is useful! Throws an error when the data is too large for the header field
	 * @param offset (Optional) offset where to start reading, e.g. when giving in whole tag
	 * @return ???
	 */
	private static Number[] parseTagData(int tagCount, byte[] rawData, TIFFDataTypes dataType, COGHeader cogHeader, int maxSize, int offset) {
		if (maxSize > 0 && dataType.getSize() * tagCount > maxSize) {
			throw new DMLRuntimeException("Error while parsing. Data type " + dataType.toString() + " cannot fit into " + maxSize + " bytes");
		}
		Number[] tagData = new Number[tagCount];
		for (int j = 0; j < tagCount; j++) {
			switch(dataType) {
				case BYTE:
				case ASCII:
				case SHORT:
				case LONG:
				case LONG8:
				case UNDEFINED:
					tagData[j] = cogHeader.parseByteArray(rawData, dataType.getSize(), offset + j * dataType.getSize(), false, false, false);
					break;
				case SBYTE:
				case SSHORT:
				case SLONG:
				case SLONG8:
				case IFD8:
					tagData[j] = cogHeader.parseByteArray(rawData, dataType.getSize(), offset + j * dataType.getSize(), false, true, false);
					break;
				case RATIONAL:
					tagData[j] = cogHeader.parseByteArray(rawData, dataType.getSize(), offset + j * dataType.getSize(), false, false, true);
					break;
				case SRATIONAL:
					tagData[j] = cogHeader.parseByteArray(rawData, dataType.getSize(), offset + j * dataType.getSize(), false, true, true);
					break;
				case FLOAT:
				case DOUBLE:
					tagData[j] = cogHeader.parseByteArray(rawData, dataType.getSize(), offset + j * dataType.getSize(), true, false, false);
					break;
			}
		}

		return tagData;
	}

	/**
	 * Checks a given header for compatibility with the reader
	 * @param IFD  ???
	 * @return empty string if compatible, error message otherwise
	 */
	@SuppressWarnings("incomplete-switch")
	public static String isCompatible(IFDTag[] IFD) {
		boolean hasTileOffsets = false;
		int imageWidth = -1;
		int imageHeight = -1;
		int tileWidth = -1;
		int tileHeight = -1;
		for (IFDTag tag : IFD) {
			// Only 8 bit, 16 bit, 32 bit images are supported
			// This is common practice in TIFF readers
			// 12 bit values e.g. should instead be scaled to 16 bit
			switch (tag.getTagId()) {
				case BitsPerSample:
					Number[] data = tag.getData();
					for (int i = 0; i < data.length; i++) {
						if (data[i].intValue() != 8 && data[i].intValue() != 16 && data[i].intValue() != 32) {
							return "Unsupported bit depth: " + data[i];
						}
					}
					break;
				case TileOffsets:
					if (tag.getData().length > 0) {
						hasTileOffsets = true;
					}
					break;
				case Compression:
					// After implementing additional decompression methods, this can be extended
					// TODO: LZW would be a great addition as it is widely used
					// Furthermore, JPEG support would also be a good addition
					// 1: none, 8: deflate
					if (tag.getData()[0].intValue() != 1 && tag.getData()[0].intValue() != 8) {
						return "Unsupported compression: " + tag.getData()[0];
					}
					break;
				case ImageWidth:
					imageWidth = tag.getData()[0].intValue();
					break;
				case ImageLength:
					imageHeight = tag.getData()[0].intValue();
					break;
				case TileWidth:
					tileWidth = tag.getData()[0].intValue();
					break;
				case TileLength:
					tileHeight = tag.getData()[0].intValue();
					break;
			}
		}
		if (!hasTileOffsets) {
			return "No tile offsets found";
		}
		if (imageWidth % tileWidth != 0 || imageHeight % tileHeight != 0) {
			return "Image can't be split into tiles equally";
		}
		return "";
	}
}

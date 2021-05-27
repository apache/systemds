/*
 * This file is part of jHDF. A pure Java library for accessing HDF5 files.
 *
 * http://jhdf.io
 *
 * Copyright (c) 2020 James Mudd
 *
 * MIT License see 'LICENSE' file
 */
package org.apache.sysds.runtime.io.hdf5.object.datatype;

import org.apache.sysds.runtime.io.hdf5.HdfFileChannel;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

import static org.apache.sysds.runtime.io.hdf5.Constants.NULL;
import static org.apache.sysds.runtime.io.hdf5.Constants.SPACE;
import static org.apache.sysds.runtime.io.hdf5.Utils.stripLeadingIndex;

/**
 * Data type representing strings.
 *
 * @author James Mudd
 */
public class StringData extends DataType {

	private final PaddingType paddingType;

	private final Charset charset;

	@Override
	public Object fillData(ByteBuffer buffer, int[] dimensions, HdfFileChannel hdfFc) {
		final Object data = Array.newInstance(getJavaType(), dimensions);
		fillFixedLengthStringData(data, dimensions, buffer, getSize(), getCharset(), getStringPaddingHandler());
		return data;
	}

	private static void fillFixedLengthStringData(Object data, int[] dims, ByteBuffer buffer, int stringLength, Charset charset, StringPaddingHandler stringPaddingHandler) {
		if (dims.length > 1) {
			for (int i = 0; i < dims[0]; i++) {
				Object newArray = Array.get(data, i);
				fillFixedLengthStringData(newArray, stripLeadingIndex(dims), buffer, stringLength, charset, stringPaddingHandler);
			}
		} else {
			for (int i = 0; i < dims[0]; i++) {
				ByteBuffer elementBuffer = Utils.createSubBuffer(buffer, stringLength);
				stringPaddingHandler.setBufferLimit(elementBuffer);
				Array.set(data, i, charset.decode(elementBuffer).toString());
			}
		}
	}

    public enum PaddingType {
		NULL_TERMINATED(new NullTerminated()),
		NULL_PADDED(new NullPadded()),
		SPACE_PADDED(new SpacePadded());

		private final StringPaddingHandler stringPaddingHandler;

		PaddingType(StringPaddingHandler stringPaddingHandler) {
			this.stringPaddingHandler = stringPaddingHandler;
		}
	}

	public StringData(ByteBuffer bb) {
		super(bb);

		final int paddingTypeValue = Utils.bitsToInt(classBits, 0, 4);
		switch (paddingTypeValue) {
			case 0:
				paddingType = PaddingType.NULL_TERMINATED;
				break;
			case 1:
				paddingType = PaddingType.NULL_PADDED;
				break;
			case 2:
				paddingType = PaddingType.SPACE_PADDED;
				break;
			default:
				throw new HdfException("Unrecognized padding type. Value is: " + paddingTypeValue);
		}

		final int charsetIndex = Utils.bitsToInt(classBits, 4, 4);
		switch (charsetIndex) {
		case 0:
			charset = StandardCharsets.US_ASCII;
			break;
		case 1:
			charset = StandardCharsets.UTF_8;
			break;
		default:
			throw new HdfException("Unrecognized Charset. Index is: " + charsetIndex);
		}
	}

	public PaddingType getPaddingType() {
		return paddingType;
	}

	public StringPaddingHandler getStringPaddingHandler() {
		return paddingType.stringPaddingHandler;
	}

	public Charset getCharset() {
		return charset;
	}

	@Override
	public Class<?> getJavaType() {
		return String.class;
	}

	public interface StringPaddingHandler {
		void setBufferLimit(ByteBuffer byteBuffer);
	}

	/* package */ static class NullTerminated implements StringPaddingHandler {
		@Override
		public void setBufferLimit(ByteBuffer byteBuffer) {
			final int limit = byteBuffer.limit();
			int i = 0;
			while (i < limit && byteBuffer.get(i) != NULL) {
				i++;
			}
			// Set the limit to terminate before the null
			byteBuffer.limit(i);
		}
	}

	/* package */ static class NullPadded implements StringPaddingHandler {
		@Override
		public void setBufferLimit(ByteBuffer byteBuffer) {
			int i = byteBuffer.limit() - 1;
			while (i >= 0 && byteBuffer.get(i) == NULL) {
				i--;
			}
			// Set the limit to terminate before the nulls
			byteBuffer.limit(i + 1);
		}
	}

	/* package */ static class SpacePadded implements StringPaddingHandler {
		@Override
		public void setBufferLimit(ByteBuffer byteBuffer) {
			int i = byteBuffer.limit() - 1;
			while (i >= 0 && byteBuffer.get(i) == SPACE) {
				i--;
			}
			// Set the limit to terminate before the spaces
			byteBuffer.limit(i + 1);
		}
	}

	@Override
	public String toString() {
		return "StringData{" +
				"paddingType=" + paddingType +
				", charset=" + charset +
				'}';
	}
}

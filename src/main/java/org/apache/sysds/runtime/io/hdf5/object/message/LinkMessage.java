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


package org.apache.sysds.runtime.io.hdf5.object.message;

import org.apache.sysds.runtime.io.hdf5.Superblock;
import org.apache.sysds.runtime.io.hdf5.Utils;
import org.apache.sysds.runtime.io.hdf5.exceptions.HdfException;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.BitSet;

import static java.nio.charset.StandardCharsets.US_ASCII;
import static java.nio.charset.StandardCharsets.UTF_8;

public class LinkMessage extends Message {

	public enum LinkType {
		HARD, SOFT, EXTERNAL;

		private static LinkType fromInt(int typeInt) {
			switch (typeInt) {
			case 0:
				return HARD;
			case 1:
				return SOFT;
			case 64:
				return EXTERNAL;
			default:
				throw new HdfException("Unrecognized link type: " + typeInt);
			}
		}
	}

	private static final int CREATION_ORDER_PRESENT = 2;
	private static final int LINK_TYPE_PRESENT = 3;
	private static final int LINK_CHARACTER_SET_PRESENT = 4;

	private final byte version;
	private final LinkType linkType;
	private final long creationOrder;
	private final String linkName;
	private long hardLinkAddress;
	private String softLink;
	private String externalFile;
	private String externalPath;

	public static LinkMessage fromBuffer(ByteBuffer bb, Superblock sb) {
		return new LinkMessage(bb, sb, null);
	}

	/* package */ LinkMessage(ByteBuffer bb, Superblock sb, BitSet messageFlags) {
		super(messageFlags);

		// Version
		version = bb.get();
		if (version != 1) {
			throw new HdfException("Unrecognized version = " + version);
		}

		// Flags
		final BitSet flags = BitSet.valueOf(new byte[] { bb.get() });

		// Size of length of link name
		final int sizeOfLengthOfLinkNameIndex = Utils.bitsToInt(flags, 0, 2);
		final int sizeOfLengthOfLinkName;
		switch (sizeOfLengthOfLinkNameIndex) {
		case 0:
			sizeOfLengthOfLinkName = 1;
			break;
		case 1:
			sizeOfLengthOfLinkName = 2;
			break;
		case 2:
			sizeOfLengthOfLinkName = 4;
			break;
		case 3:
			sizeOfLengthOfLinkName = 8;
			break;
		default:
			throw new HdfException("Unrecognized size of link name");
		}

		if (flags.get(LINK_TYPE_PRESENT)) {
			linkType = LinkType.fromInt(Utils.readBytesAsUnsignedInt(bb, 1));
		} else {
			linkType = LinkType.HARD; // No link type specified so is hard link
		}

		if (flags.get(CREATION_ORDER_PRESENT)) {
			creationOrder = Utils.readBytesAsUnsignedLong(bb, 8);
		} else {
			creationOrder = -1;
		}

		final Charset linkNameCharset;
		if (flags.get(LINK_CHARACTER_SET_PRESENT)) {
			int charsetValue = Utils.readBytesAsUnsignedInt(bb, 1);
			switch (charsetValue) {
			case 0:
				linkNameCharset = US_ASCII;
				break;
			case 1:
				linkNameCharset = UTF_8;
				break;
			default:
				throw new HdfException("Unknown link charset value = " + charsetValue);
			}
		} else {
			linkNameCharset = US_ASCII;
		}

		final int lengthOfLinkName = Utils.readBytesAsUnsignedInt(bb, sizeOfLengthOfLinkName);

		ByteBuffer nameBuffer = Utils.createSubBuffer(bb, lengthOfLinkName);

		linkName = linkNameCharset.decode(nameBuffer).toString();

		// Link Information
		switch (linkType) {
		case HARD: // Hard Link
			hardLinkAddress = Utils.readBytesAsUnsignedLong(bb, sb.getSizeOfOffsets());
			break;
		case SOFT: // Soft link
			int lengthOfSoftLink = Utils.readBytesAsUnsignedInt(bb, 2);
			ByteBuffer linkBuffer = Utils.createSubBuffer(bb, lengthOfSoftLink);
			softLink = US_ASCII.decode(linkBuffer).toString();
			break;
		case EXTERNAL: // External link
			int lengthOfExternalLink = Utils.readBytesAsUnsignedInt(bb, 2);
			ByteBuffer externalLinkBuffer = Utils.createSubBuffer(bb, lengthOfExternalLink);
			// Skip first byte contains version = 0 and flags = 0
			externalLinkBuffer.position(1);
			externalFile = Utils.readUntilNull(externalLinkBuffer);
			externalPath = Utils.readUntilNull(externalLinkBuffer);
			break;
		default:
			throw new HdfException("Unrecognized link type = " + linkType);
		}
	}

	public byte getVersion() {
		return version;
	}

	public LinkType getLinkType() {
		return linkType;
	}

	public long getCreationOrder() {
		return creationOrder;
	}

	public String getLinkName() {
		return linkName;
	}

	public long getHardLinkAddress() {
		if (linkType == LinkType.HARD) {
			return hardLinkAddress;
		} else {
			throw new HdfException("This link message is not a hard link. Link type is: " + linkType);
		}
	}

	public String getSoftLink() {
		if (linkType == LinkType.SOFT) {
			return softLink;
		} else {
			throw new HdfException("This link message is not a soft link. Link type is: " + linkType);
		}
	}

	public String getExternalFile() {
		if (linkType == LinkType.EXTERNAL) {
			return externalFile;
		} else {
			throw new HdfException("This link message is not a external link. Link type is: " + linkType);
		}
	}

	public String getExternalPath() {
		if (linkType == LinkType.EXTERNAL) {
			return externalPath;
		} else {
			throw new HdfException("This link message is not a external link. Link type is: " + linkType);
		}
	}
}

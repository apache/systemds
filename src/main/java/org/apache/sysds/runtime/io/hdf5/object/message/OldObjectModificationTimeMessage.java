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

import org.apache.sysds.runtime.io.hdf5.Utils;
import java.nio.ByteBuffer;
import java.time.LocalDateTime;
import java.util.BitSet;

import static java.lang.Integer.parseInt;
import static java.nio.charset.StandardCharsets.US_ASCII;

public class OldObjectModificationTimeMessage extends Message {

	final LocalDateTime modificationTime;

	public OldObjectModificationTimeMessage(ByteBuffer bb, BitSet flags) {
		super(flags);

		final ByteBuffer yearBuffer = Utils.createSubBuffer(bb, 4);
		final int year = parseInt(US_ASCII.decode(yearBuffer).toString());

		final ByteBuffer monthBuffer = Utils.createSubBuffer(bb, 2);
		final int month = parseInt(US_ASCII.decode(monthBuffer).toString());

		final ByteBuffer dayBuffer = Utils.createSubBuffer(bb, 2);
		final int day = parseInt(US_ASCII.decode(dayBuffer).toString());

		final ByteBuffer hourBuffer = Utils.createSubBuffer(bb, 2);
		final int hour = parseInt(US_ASCII.decode(hourBuffer).toString());

		final ByteBuffer minuteBuffer = Utils.createSubBuffer(bb, 2);
		final int minute = parseInt(US_ASCII.decode(minuteBuffer).toString());

		final ByteBuffer secondBuffer = Utils.createSubBuffer(bb, 2);
		final int second = parseInt(US_ASCII.decode(secondBuffer).toString());

		// Skip reserved bytes
		bb.position(bb.position() + 2);

		this.modificationTime = LocalDateTime.of(year, month, day, hour, minute, second);
	}

	public LocalDateTime getModifiedTime() {
		return modificationTime;
	}
}

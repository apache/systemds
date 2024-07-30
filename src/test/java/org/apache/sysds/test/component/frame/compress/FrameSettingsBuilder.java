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

package org.apache.sysds.test.component.frame.compress;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.frame.data.compress.FrameCompressionSettingsBuilder;
import org.junit.Test;

public class FrameSettingsBuilder {
	@Test
	public void builderTest1() {
		var a = new FrameCompressionSettingsBuilder();
		a.sampleRatio(0.2);
		var s = a.create();
		assertEquals(0.2, s.sampleRatio, 0.0);
	}

	@Test
	public void builderTest2() {
		var a = new FrameCompressionSettingsBuilder();
		a.threads(13);
		var s = a.create();
		assertEquals(13, s.k);
	}
}

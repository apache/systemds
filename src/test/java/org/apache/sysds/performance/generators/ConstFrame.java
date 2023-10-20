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

package org.apache.sysds.performance.generators;

import java.util.Arrays;

import org.apache.sysds.runtime.frame.data.FrameBlock;

public class ConstFrame implements Const<FrameBlock> {

	protected FrameBlock fb;

	public ConstFrame(FrameBlock fb) {
		this.fb = fb;
	}

	@Override
	public FrameBlock take() {
		return fb;
	}

	@Override
	public void generate(int N) throws InterruptedException {
		// do nothing
	}

	@Override
	public final boolean isEmpty() {
		return false;
	}

	@Override
	public final int defaultWaitTime() {
		return 0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" Schema:");
		sb.append(Arrays.toString(fb.getSchema()));
		return sb.toString();
	}

	@Override
	public void change(FrameBlock t) {
		fb = t;
	}
}

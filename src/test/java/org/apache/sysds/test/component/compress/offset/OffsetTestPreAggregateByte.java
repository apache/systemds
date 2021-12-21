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

package org.apache.sysds.test.component.compress.offset;

import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public class OffsetTestPreAggregateByte extends OffsetTestPreAggregate {

	public OffsetTestPreAggregateByte(int[] data, OFF_TYPE type) {
		super(data, type);
	}

	@Override
	protected void preAggMapRow(int row) {
		double[] preAV = new double[1];
		byte[] m = new byte[data.length];
		a.preAggregateDenseMap(this.leftM, preAV, row, 1 + row, 0, leftM.getNumColumns(), 2, m);
		verifyPreAggMapRowByte(preAV, row);
	}

	@Override
	protected void preAggMapRowAll1(int row) {
		double[] preAV = new double[2];
		byte[] m = new byte[data.length];
		fill(m, (byte) 1);
		a.preAggregateDenseMap(this.leftM, preAV, row, 1 + row, 0, leftM.getNumColumns(), 2, m);
		verifyPreAggMapRowAllBytes1(preAV, row);
	}

	private final void fill(byte[] a, byte v) {
		for(int i = 0; i < a.length; i++)
			a[i] = v;
	}

	@Override
	protected void preAggMapRowOne1(int row) {
		if(data.length > 1) {
			double[] preAV = new double[2];
			byte[] m = new byte[data.length];
			m[1] = 1;
			a.preAggregateDenseMap(this.leftM, preAV, row, 1 + row, 0, leftM.getNumColumns(), 2, m);
			verifyPreAggMapRowOne1(preAV, row);
		}
	}

	@Override
	public void preAggMapAllRowsOne1() {
		if(data.length > 1) {
			double[] preAV = new double[4];
			byte[] m = new byte[data.length];
			m[1] = 1;
			a.preAggregateDenseMap(this.leftM, preAV, 0, 2, 0, leftM.getNumColumns(), 2, m);
			verifyPreAggAllOne1(preAV);
		}
	}

	@Override
	protected void preAggMapSubOfRow(int row) {
		if(data.length > 2) {
			double[] preAV = new double[2];
			byte[] m = new byte[data.length];
			m[1] = 1;
			a.preAggregateDenseMap(this.leftM, preAV, row, 1 + row, 0, data[data.length - 1], 2, m);
			verifyPreAggMapSubOfRow(preAV, row);
		}
	}

	@Override
	protected void preAggMapSubOfRowV2(int row, int nVal) {
		if(data.length > 3) {
			double[] preAV = new double[2];
			byte[] m = new byte[data.length];
			m[1] = 1;
			a.preAggregateDenseMap(this.leftM, preAV, row, 1 + row, 0, data[data.length - 2], nVal, m);
			verifyPreAggMapSubOfRowV2(preAV, row);
		}
	}

	@Override
	protected void preAggMapOutOfRangeBefore(int row) {
		double[] preAV = null; // should not need access this therefore we make a null argument here.
		byte[] m = null; // new byte[data.length];
		a.preAggregateDenseMap(this.leftM, preAV, row, 1 + row, -412, data[0] - 1, 2, m);
	}

	@Override
	protected void preAggMapOutOfRangeAfter(int row) {
		double[] preAV = null; // should not need access this therefore we make a null argument here.
		byte[] m = null;// new char[data.length];
		int id = data[data.length - 1] + 10;
		a.preAggregateDenseMap(this.leftM, preAV, row, 1 + row, id, id + 10, 2, m);
	}


	@Override
	protected  double[] multiRowPreAggRange(int rl, int ru){
		double[] preAV = new double[2 * (ru - rl)];
		byte[] m = new byte[data.length];
		m[1] = 1;
		a.preAggregateDenseMap(this.leftM, preAV, rl, ru, 0, leftM.getNumColumns(), 2, m);
		return preAV;
	}

	@Override
	protected  double[] multiRowPreAggRangeBeforeLast(int rl, int ru){
		double[] preAV = new double[2 * (ru - rl)];
		byte[] m = new byte[data.length];
		m[1] = 1;
		a.preAggregateDenseMap(this.leftM, preAV, rl, ru, 0, data[data.length - 1], 2, m);
		return preAV;
	}
}

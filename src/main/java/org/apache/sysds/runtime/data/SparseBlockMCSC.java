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


package org.apache.sysds.runtime.data;

import java.util.Iterator;

public class SparseBlockMCSC extends SparseBlock{

	private static final long serialVersionUID = 112364695245614881L;

	@Override
	public void allocate(int r) {

	}

	@Override
	public void allocate(int r, int nnz) {

	}

	@Override
	public void allocate(int r, int ennz, int maxnnz) {

	}

	@Override
	public void compact(int r) {

	}

	@Override
	public int numRows() {
		return 0;
	}

	@Override
	public boolean isThreadSafe() {
		return false;
	}

	@Override
	public boolean isContiguous() {
		return false;
	}

	@Override
	public boolean isAllocated(int r) {
		return false;
	}

	@Override
	public void reset() {

	}

	@Override
	public void reset(int ennz, int maxnnz) {

	}

	@Override
	public void reset(int r, int ennz, int maxnnz) {

	}

	@Override
	public long size() {
		return 0;
	}

	@Override
	public int size(int r) {
		return 0;
	}

	@Override
	public long size(int rl, int ru) {
		return 0;
	}

	@Override
	public long size(int rl, int ru, int cl, int cu) {
		return 0;
	}

	@Override
	public boolean isEmpty(int r) {
		return false;
	}

	@Override
	public boolean checkValidity(int rlen, int clen, long nnz, boolean strict) {
		return false;
	}

	@Override
	public int[] indexes(int r) {
		return new int[0];
	}

	@Override
	public double[] values(int r) {
		return new double[0];
	}

	@Override
	public int pos(int r) {
		return 0;
	}

	@Override
	public boolean set(int r, int c, double v) {
		return false;
	}

	@Override
	public void set(int r, SparseRow row, boolean deep) {

	}

	@Override
	public boolean add(int r, int c, double v) {
		return false;
	}

	@Override
	public void append(int r, int c, double v) {

	}

	@Override
	public void setIndexRange(int r, int cl, int cu, double[] v, int vix, int vlen) {

	}

	@Override
	public void setIndexRange(int r, int cl, int cu, double[] v, int[] vix, int vpos, int vlen) {

	}

	@Override
	public void deleteIndexRange(int r, int cl, int cu) {

	}

	@Override
	public void sort() {

	}

	@Override
	public void sort(int r) {

	}

	@Override
	public double get(int r, int c) {
		return 0;
	}

	@Override
	public SparseRow get(int r) {
		return null;
	}

	@Override
	public int posFIndexLTE(int r, int c) {
		return 0;
	}

	@Override
	public int posFIndexGTE(int r, int c) {
		return 0;
	}

	@Override
	public int posFIndexGT(int r, int c) {
		return 0;
	}

	@Override
	public Iterator<Integer> getNonEmptyRowsIterator(int rl, int ru) {
		return null;
	}

	@Override
	public String toString() {
		return null;
	}
}

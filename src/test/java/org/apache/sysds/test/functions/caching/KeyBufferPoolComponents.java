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

package org.apache.sysds.test.functions.caching;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import org.apache.sysds.common.Warnings;
import org.apache.sysds.runtime.controlprogram.caching.ByteBuffer;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlockFactory;
import org.apache.sysds.runtime.controlprogram.caching.CacheDataInput;
import org.apache.sysds.runtime.controlprogram.caching.CacheDataOutput;
import org.apache.sysds.runtime.controlprogram.caching.PageCache;
import org.apache.sysds.runtime.data.TensorBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

public class KeyBufferPoolComponents extends AutomatedTestBase 
{
	@Override
	public void setUp() {}
	
	@Test
	public void testDataStreamsDense() {
		testSerialization(100, 100, 0.7);
	}
	
	@Test
	public void testDataStreamsSparse() {
		testSerialization(100, 100, 0.07);
	}
	
	@Test
	public void testBufferDense() {
		testBufferSerialization(100, 100, 0.7);
	}
	
	@Test
	public void testBufferSparse() {
		testBufferSerialization(100, 100, 0.007);
	}
	
	@Test
	public void testCacheBlockFactory() {
		Assert.assertEquals(new MatrixBlock(), CacheBlockFactory.newInstance(0));
		Assert.assertEquals(
			new FrameBlock().getInMemorySize(), CacheBlockFactory.newInstance(1).getInMemorySize());
		Assert.assertEquals(
			new TensorBlock().getInMemorySize(), CacheBlockFactory.newInstance(2).getInMemorySize());
		Assert.assertThrows(RuntimeException.class, ()->CacheBlockFactory.newInstance(3));
		
		Assert.assertEquals(new MatrixBlock(), CacheBlockFactory.newInstance(new MatrixBlock()));
		Assert.assertEquals(
			new FrameBlock().getInMemorySize(), CacheBlockFactory.newInstance(new FrameBlock()).getInMemorySize());
		Assert.assertEquals(
			new TensorBlock().getInMemorySize(), CacheBlockFactory.newInstance(new TensorBlock()).getInMemorySize());
		Assert.assertThrows(RuntimeException.class, ()->CacheBlockFactory.newInstance(null));
		
		Assert.assertThrows(RuntimeException.class, ()->CacheBlockFactory.getCode(null));
		Assert.assertThrows(RuntimeException.class, ()->CacheBlockFactory.getPairList(null));
	}
	
	@Test
	public void testPageCache() {
		//coverage for classes w/ only static methods
		new Warnings();
		new PageCache();
		
		PageCache.init();
		for(int i=7; i<256; i++) {
			PageCache.putPage(new byte[i]);
			PageCache.putPage(new byte[i]);
		}
		int count = 0;
		for(int i=7; i<256; i++)
			count += PageCache.getPage(i)!=null ? 1: 0;
		System.out.println("Found "+count+" pages for reuse."); //120
		PageCache.clear();
	}
	
	private void testSerialization(int rows, int cols, double sparsity) {
		try {
			MatrixBlock mb = MatrixBlock.randOperations(rows, cols, sparsity);
			byte[] barr = new byte[(int)mb.getExactSizeOnDisk()];
			CacheDataOutput dos = new CacheDataOutput(barr);
			mb.write(dos);
			CacheDataInput dis = new CacheDataInput(barr);
			MatrixBlock mb2 = new MatrixBlock();
			mb2.readFields(dis);
			TestUtils.compareMatrices(mb, mb2, 1e-14);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	private void testBufferSerialization(int rows, int cols, double sparsity) {
		try {
			MatrixBlock mb = MatrixBlock.randOperations(rows, cols, sparsity);
			ByteBuffer buff = new ByteBuffer((int)mb.getExactSizeOnDisk());
			Future<?> check = CommonThreadPool.get().submit(()->buff.checkSerialized());
			buff.serializeBlock(mb);
			check.get(); //check non-blocking after serialization
			MatrixBlock mb2 = (MatrixBlock) buff.deserializeBlock();
			TestUtils.compareMatrices(mb, mb2, 1e-14);
		} catch (IOException | InterruptedException | ExecutionException e) {
			throw new RuntimeException(e);
		}
	}
}

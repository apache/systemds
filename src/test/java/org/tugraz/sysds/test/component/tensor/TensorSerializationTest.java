/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.component.tensor;

import java.io.DataInput;
import java.io.DataOutput;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheDataInput;
import org.tugraz.sysds.runtime.controlprogram.caching.CacheDataOutput;
import org.tugraz.sysds.runtime.data.HeterogTensor;
import org.tugraz.sysds.runtime.data.HomogTensor;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.TestUtils;


public class TensorSerializationTest 
{
	@Test
	public void testSerializeHomogTensorFP32() {
		HomogTensor tb1 = createHomogTensor(ValueType.FP32, 70, 30, 0.7);
		HomogTensor tb2 = serializeAndDeserializeHomog(tb1);
		compareHomogTensors(tb1, tb2);
	}
	
	@Test
	public void testSerializeHomogTensorFP64() {
		HomogTensor tb1 = createHomogTensor(ValueType.FP64, 70, 30, 0.7);
		HomogTensor tb2 = serializeAndDeserializeHomog(tb1);
		compareHomogTensors(tb1, tb2);
	}
	
	@Test
	public void testSerializeHomogTensorINT32() {
		HomogTensor tb1 = createHomogTensor(ValueType.INT32, 70, 30, 0.7);
		HomogTensor tb2 = serializeAndDeserializeHomog(tb1);
		compareHomogTensors(tb1, tb2);
	}
	
	@Test
	public void testSerializeHomogTensorINT64() {
		HomogTensor tb1 = createHomogTensor(ValueType.INT64, 70, 30, 0.7);
		HomogTensor tb2 = serializeAndDeserializeHomog(tb1);
		compareHomogTensors(tb1, tb2);
	}
	
	@Test
	public void testSerializeHomogTensorBoolean() {
		HomogTensor tb1 = createHomogTensor(ValueType.BOOLEAN, 70, 30, 0.7);
		HomogTensor tb2 = serializeAndDeserializeHomog(tb1);
		compareHomogTensors(tb1, tb2);
	}

	@Test
	public void testSerializeHeterogTensorFP32() {
		HeterogTensor tb1 = createHeterogTensor(ValueType.FP32, 70, 30, 0.7);
		HeterogTensor tb2 = serializeAndDeserializeHeterog(tb1);
		compareHeterogTensors(tb1, tb2);
	}

	@Test
	public void testSerializeHeterogTensorFP64() {
		HeterogTensor tb1 = createHeterogTensor(ValueType.FP64, 70, 30, 0.7);
		HeterogTensor tb2 = serializeAndDeserializeHeterog(tb1);
		compareHeterogTensors(tb1, tb2);
	}

	@Test
	public void testSerializeHeterogTensorINT32() {
		HeterogTensor tb1 = createHeterogTensor(ValueType.INT32, 70, 30, 0.7);
		HeterogTensor tb2 = serializeAndDeserializeHeterog(tb1);
		compareHeterogTensors(tb1, tb2);
	}

	@Test
	public void testSerializeHeterogTensorINT64() {
		HeterogTensor tb1 = createHeterogTensor(ValueType.INT64, 70, 30, 0.7);
		HeterogTensor tb2 = serializeAndDeserializeHeterog(tb1);
		compareHeterogTensors(tb1, tb2);
	}

	@Test
	public void testSerializeHeterogTensorBoolean() {
		HeterogTensor tb1 = createHeterogTensor(ValueType.BOOLEAN, 70, 30, 0.7);
		HeterogTensor tb2 = serializeAndDeserializeHeterog(tb1);
		compareHeterogTensors(tb1, tb2);
	}

	private HomogTensor serializeAndDeserializeHomog(HomogTensor tb1) {
		try {
			//serialize and deserialize tensor block
			byte[] bdata = new byte[(int)tb1.getExactSerializedSize()];
			DataOutput dout = new CacheDataOutput(bdata);
			tb1.write(dout); //tb1 serialized into bdata
			DataInput din = new CacheDataInput(bdata);
			HomogTensor tb2 = new HomogTensor();
			tb2.readFields(din); //bdata deserialized into tb2
			return tb2;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private HeterogTensor serializeAndDeserializeHeterog(HeterogTensor tb1) {
		try {
			//serialize and deserialize tensor block
			byte[] bdata = new byte[(int)tb1.getExactSerializedSize()];
			DataOutput dout = new CacheDataOutput(bdata);
			tb1.write(dout); //tb1 serialized into bdata
			DataInput din = new CacheDataInput(bdata);
			HeterogTensor tb2 = new HeterogTensor();
			tb2.readFields(din); //bdata deserialized into tb2
			return tb2;
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}

	private HomogTensor createHomogTensor(ValueType vt, int rows, int cols, double sparsity) {
		return DataConverter.convertToHomogTensor(TestUtils.round(
			MatrixBlock.randOperations(rows, cols, sparsity, 0, 1, "uniform", 7)), vt);
	}

	private HeterogTensor createHeterogTensor(ValueType vt, int rows, int cols, double sparsity) {
		return DataConverter.convertToHeterogTensor(TestUtils.round(
				MatrixBlock.randOperations(rows, cols, sparsity, 0, 1, "uniform", 7)), vt);
	}

	private void compareHomogTensors(HomogTensor tb1, HomogTensor tb2) {
		Assert.assertEquals(tb1.getValueType(), tb2.getValueType());
		Assert.assertEquals(tb1.getNumRows(), tb2.getNumRows());
		Assert.assertEquals(tb1.getNumColumns(), tb2.getNumColumns());
		for(int i=0; i<tb1.getNumRows(); i++)
			for(int j=0; j<tb1.getNumColumns(); j++)
				Assert.assertEquals(Double.valueOf(tb1.get(i, j)),
					Double.valueOf(tb2.get(i, j)));
	}

	private void compareHeterogTensors(HeterogTensor tb1, HeterogTensor tb2) {
		Assert.assertArrayEquals(tb1.getSchema(), tb2.getSchema());
		Assert.assertEquals(tb1.getNumRows(), tb2.getNumRows());
		Assert.assertEquals(tb1.getNumColumns(), tb2.getNumColumns());
		for(int i=0; i<tb1.getNumRows(); i++)
			for(int j=0; j<tb1.getNumColumns(); j++)
				Assert.assertEquals(Double.valueOf(tb1.get(i, j)),
						Double.valueOf(tb2.get(i, j)));
	}
}

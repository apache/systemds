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

import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory.OFF_TYPE_SPECIALIZATIONS;
import org.junit.Test;

public class NegativeOffsetTest {
	static{
		CompressedMatrixBlock.debug = true;
	}

	@Test(expected = Exception.class)
	public void incorrectConstruct() throws Exception{
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream fos = new DataOutputStream(bos);
		
		fos.writeByte((byte)OFF_TYPE_SPECIALIZATIONS.BYTE.ordinal());
		fos.writeInt(0);
		fos.writeInt(2);
		fos.writeInt(10);
		fos.writeInt(2);

		fos.writeByte(5);
		fos.writeByte(4);

		// Serialize in
		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream fis = new DataInputStream(bis);

		AOffset n = OffsetFactory.readIn(fis);
		n.verify(10);
	}


	@Test(expected = Exception.class)
	public void notEmptyIteratorOnEmpty(){
		AOffset a = OffsetFactory.createOffset(new int[]{});
		AOffset spy = spy(a);
		AOffset b = OffsetFactory.createOffset(new int[]{1,2,3});
		when(spy.getIterator()).thenReturn(b.getIterator());
		spy.verify(0);
	}

	@Test(expected = Exception.class)
	public void verifyIncorrectSizeGivenOnEmpty(){
		AOffset a = OffsetFactory.createOffset(new int[]{});
		a.verify(2);
	}

	@Test(expected = Exception.class)
	public void toManyElementsForVerify1(){
		AOffset b = OffsetFactory.createOffset(new int[]{1,2,3,5});
		b.verify(2);
	}

	@Test(expected = Exception.class)
	public void toManyElementsForVerify2(){
		AOffset b = OffsetFactory.createOffset(new int[]{1,2,3});
		b.verify(1);
	}

	@Test(expected = Exception.class)
	public void toManyElementsForVerify3(){
		AOffset b = OffsetFactory.createOffset(new int[]{1,2,3,5});
		b.verify(-1);// stupid however valid test.
	}

}

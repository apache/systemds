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

package org.apache.sysds.test.functions.transform;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoder;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderBagOfWords;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderComposite;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.EncoderOmit;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class ColumnEncoderMixedFunctionalityTests extends AutomatedTestBase
{
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testCompositeConstructor1() {
		ColumnEncoderComposite cEnc1 = new ColumnEncoderComposite(null, 1);
		ColumnEncoderComposite cEnc2 = new ColumnEncoderComposite(cEnc1);
		assert cEnc1.getColID() == cEnc2.getColID();

	}
	@Test
	public void testCompositeConstructor2() {
		List<ColumnEncoder> encoderList = new ArrayList<>();
		encoderList.add( new ColumnEncoderComposite(null, 1));
		encoderList.add( new ColumnEncoderComposite(null, 2));
		DMLRuntimeException e = assertThrows(DMLRuntimeException.class, () ->  new ColumnEncoderComposite(encoderList, null));
		assertTrue(e.getMessage().contains("Tried to create Composite Encoder with no encoders or mismatching columnIDs"));
	}

	@Test
	public void testEncoderFactoryGetUnsupportedEncoderType(){
		DMLRuntimeException e = assertThrows(DMLRuntimeException.class, () ->  EncoderFactory.getEncoderType(new ColumnEncoderComposite()));
		assertTrue(e.getMessage().contains("Unsupported encoder type: org.apache.sysds.runtime.transform.encode.ColumnEncoderComposite"));
	}

	@Test
	public void testEncoderFactoryCreateUnsupportedInstanceType(){
		// type(7) = composite, which we don't use for encoding the type
		DMLRuntimeException e = assertThrows(DMLRuntimeException.class, () ->  EncoderFactory.createInstance(7));
		assertTrue(e.getMessage().contains("Unsupported encoder type: Composite"));
	}

	@Test
	public void testMultiColumnEncoderApplyWithWrongInputCharacteristics1(){
		// apply call without metadata about encoders
		MultiColumnEncoder mEnc = new MultiColumnEncoder();
		DMLRuntimeException e = assertThrows(DMLRuntimeException.class, () -> mEnc.apply(null, null, 0));
		assertTrue(e.getMessage().contains("MultiColumnEncoder apply without Encoder Characteristics should not be called directly"));
	}

	@Test
	public void testMultiColumnEncoderApplyWithWrongInputCharacteristics2(){
		// apply with LegacyEncoders + non FrameBlock Inputs
		MultiColumnEncoder mEnc = new MultiColumnEncoder();
		mEnc.addReplaceLegacyEncoder(new EncoderOmit());
		DMLRuntimeException e = assertThrows(DMLRuntimeException.class, () -> mEnc.apply(new MatrixBlock(), null, 0, 0, null, 0L));
		assertTrue(e.getMessage().contains("LegacyEncoders do not support non FrameBlock Inputs"));
	}

	@Test
	public void testMultiColumnEncoderApplyWithWrongInputCharacteristics3(){
		// #CompositeEncoders != #cols
		ArrayList<ColumnEncoder> encs = new ArrayList<>();
		encs.add(new ColumnEncoderBagOfWords());
		ArrayList<ColumnEncoderComposite> cEncs = new ArrayList<>();
		cEncs.add(new ColumnEncoderComposite(encs));
		MultiColumnEncoder mEnc = new MultiColumnEncoder(cEncs);
		FrameBlock in = new FrameBlock(2, Types.ValueType.FP64);
		DMLRuntimeException e = assertThrows(DMLRuntimeException.class, () -> mEnc.apply(in, null, 0, 0, null, 0L));
		assertTrue(e.getMessage().contains("Not every column in has a CompositeEncoder. Please make sure every column has a encoder or slice the input accordingly"));
	}

	@Test
	public void testMultiColumnEncoderApplyWithWrongInputCharacteristics4(){
		// input has 0 rows
		MultiColumnEncoder mEnc = new MultiColumnEncoder();
		MatrixBlock in = new MatrixBlock();
		DMLRuntimeException e = assertThrows(DMLRuntimeException.class, () -> mEnc.apply(in, null, 0, 0, null, 0L));
		assertTrue(e.getMessage().contains("Invalid input with wrong number or rows"));
	}
}

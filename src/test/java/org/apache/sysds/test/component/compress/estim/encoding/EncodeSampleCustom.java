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

package org.apache.sysds.test.component.compress.estim.encoding;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.estim.encoding.SparseEncoding;
import org.apache.sysds.test.component.compress.offset.OffsetTests;
import org.junit.Test;

import scala.NotImplementedError;

public class EncodeSampleCustom {

	protected static final Log LOG = LogFactory.getLog(EncodeSampleCustom.class.getName());

	@Test
	public void testC1() {
		int[] d1 = readData("src/test/resources/component/compress/sample/s1.dat");
		int[] d2 = readData("src/test/resources/component/compress/sample/s2.dat");
		int m1 = Arrays.stream(d1).max().getAsInt() + 1;
		int m2 = Arrays.stream(d2).max().getAsInt() + 1;
		AMapToData dm1 = MapToFactory.create(d1.length, d1, m1);
		AMapToData dm2 = MapToFactory.create(d2.length, d2, m2);

		DenseEncoding de1 = new DenseEncoding(dm1);
		DenseEncoding de2 = new DenseEncoding(dm2);

		try {
			de1.combine(de2);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed combine");
		}
	}

	@Test
	public void testSparse() {
		// Custom combine from US Census Encoded dataset.
		AMapToData Z0 = MapToFactory.create(77, 0);
		AOffset O0 = OffsetFactory.createOffset(new int[] {4036, 4382, 4390, 4764, 4831, 4929, 5013, 6964, 7018, 7642,
			8306, 8559, 8650, 9041, 9633, 9770, 11000, 11702, 11851, 11890, 11912, 13048, 15859, 16164, 16191, 16212,
			17927, 18344, 19007, 19614, 19806, 20878, 21884, 21924, 22245, 22454, 23185, 23825, 24128, 24829, 25835, 26130,
			26456, 26767, 27058, 28094, 28250, 28335, 28793, 30175, 30868, 32526, 32638, 33464, 33536, 33993, 34096, 34146,
			34686, 35863, 36655, 37212, 37535, 37832, 38328, 38689, 39802, 39810, 39835, 40065, 40554, 41221, 41420, 42133,
			42914, 43027, 43092});
		AMapToData Z1 = MapToFactory.create(65, 0);
		AOffset O1 = OffsetFactory.createOffset(new int[] {294, 855, 1630, 1789, 1872, 1937, 2393, 2444, 3506, 4186, 5210,
			6048, 6073, 8645, 9147, 9804, 9895, 13759, 14041, 14198, 16138, 16548, 16566, 17249, 18257, 18484, 18777,
			18881, 19138, 19513, 20127, 21443, 23264, 23432, 24050, 24332, 24574, 24579, 25246, 25513, 25686, 27075, 31190,
			31305, 31429, 31520, 31729, 32073, 32670, 33529, 34453, 34947, 36224, 37219, 38412, 39505, 39799, 40074, 40569,
			40610, 40745, 41755, 41761, 41875, 44394});
		SparseEncoding a = EncodingFactory.createSparse(Z0, O0, 50000);
		SparseEncoding b = EncodingFactory.createSparse(Z1, O1, 50000);

		a.combine(b);
	}

	@Test
	public void testSparse_2() {
		// Custom combine from US Census Encoded dataset.
		AMapToData Z0 = MapToFactory.create(8, 0);
		AOffset O0 = OffsetFactory.createOffset(new int[] {40065, 40554, 41221, 41420, 42133, 42914, 43027, 43092});
		AMapToData Z1 = MapToFactory.create(7, 0);
		AOffset O1 = OffsetFactory.createOffset(new int[] {40569, 40610, 40745, 41755, 41761, 41875, 44394});
		SparseEncoding a = EncodingFactory.createSparse(Z0, O0, 50000);
		SparseEncoding b = EncodingFactory.createSparse(Z1, O1, 50000);

		a.combine(b);
	}

	@Test
	public void testSparse_3() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 9});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 5, 6, 7});
		int[] exp = new int[] {1, 2, 3, 5, 6, 7, 9};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_4() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 9});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 5, 6, 10});
		int[] exp = new int[] {1, 2, 3, 5, 6, 9, 10};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_5() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 9});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 5, 6, 10, 11, 12});
		int[] exp = new int[] {1, 2, 3, 5, 6, 9, 10, 11, 12};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_6() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 9, 12});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 5, 6, 10, 11, 12});
		int[] exp = new int[] {1, 2, 3, 5, 6, 9, 10, 11, 12};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_7() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 9, 11, 12});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 5, 6, 10, 11, 12});
		int[] exp = new int[] {1, 2, 3, 5, 6, 9, 10, 11, 12};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_8() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 9, 11, 12, 13, 14, 15, 16});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 5, 6, 10, 11, 12});
		int[] exp = new int[] {1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_9() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 9, 11, 12, 13, 14, 15, 16});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 12, 17});
		int[] exp = new int[] {1, 2, 3, 9, 11, 12, 13, 14, 15, 16, 17};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_10() {
		AOffset a = OffsetFactory.createOffset(new int[] {16});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 12, 17});
		int[] exp = new int[] {1, 2, 3, 12, 16, 17};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_11() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 16, 18});
		AOffset b = OffsetFactory.createOffset(new int[] {17});
		int[] exp = new int[] {1, 2, 3, 16, 17, 18};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_12() {
		AOffset a = OffsetFactory.createOffset(new int[] {466, 496, 499});
		AOffset b = OffsetFactory.createOffset(new int[] {479, 496, 497});
		int[] exp = new int[] {466, 479, 496, 497, 499};
		compareSparse(a, b, exp);
	}

	@Test
	public void testSparse_13() {
		AOffset a = OffsetFactory.createOffset(new int[] {466, 496, 497});
		AOffset b = OffsetFactory.createOffset(new int[] {479, 496, 499});
		int[] exp = new int[] {466, 479, 496, 497, 499};
		compareSparse(a, b, exp);
	}

	public void compareSparse(AOffset a, AOffset b, int[] exp) {
		try {
			AMapToData Z0 = MapToFactory.create(a.getSize(), 0);
			AMapToData Z1 = MapToFactory.create(b.getSize(), 0);
			SparseEncoding aa = EncodingFactory.createSparse(Z0, a, 50000);
			SparseEncoding bb = EncodingFactory.createSparse(Z1, b, 50000);
			SparseEncoding c = (SparseEncoding) aa.combine(bb);
			OffsetTests.compare(c.getOffsets(), exp);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed combining sparse correctly.\n" + a + "\n" + b + "\nExpected:" + Arrays.toString(exp));
		}
	}

	@Test
	public void combineSimilarOffsetButNotMap() {

		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 16, 18});
		AMapToData Z0 = MapToFactory.create(a.getSize(), 0);
		AMapToData Z1 = MapToFactory.create(a.getSize(), 0);

		SparseEncoding aa = EncodingFactory.createSparse(Z0, a, 50000);
		SparseEncoding bb = EncodingFactory.createSparse(Z1, a, 50000);
		IEncode c = aa.combine(bb);
		assertTrue(c != aa);
	}

	@Test
	public void combineSimilarMapButNotOffsets() {
		AOffset a = OffsetFactory.createOffset(new int[] {1, 2, 3, 16, 18});
		AOffset b = OffsetFactory.createOffset(new int[] {1, 2, 3, 17, 18});
		AMapToData Z0 = MapToFactory.create(a.getSize(), 0);

		SparseEncoding aa = EncodingFactory.createSparse(Z0, a, 50000);
		SparseEncoding bb = EncodingFactory.createSparse(Z0, b, 50000);
		IEncode c = aa.combine(bb);
		assertTrue(c != aa);
	}

	private static int[] readData(String path) {
		try {

			File file = new File(path);
			Scanner s = new Scanner(new FileReader(file));

			int length = s.nextInt();
			int[] ret = new int[length];
			s.nextLine();
			s.useDelimiter(Pattern.compile(","));
			for(int i = 0; i < length; i++)
				ret[i] = s.nextInt();

			s.close();

			return ret;
		}
		catch(IOException e) {
			fail("failed to read:" + path);
			throw new NotImplementedError();
		}
	}
}

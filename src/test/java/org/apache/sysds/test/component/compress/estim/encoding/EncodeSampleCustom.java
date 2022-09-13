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
import org.apache.sysds.runtime.compress.estim.encoding.DenseEncoding;
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

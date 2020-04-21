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

package org.apache.sysds.test.component.federated;

import org.apache.sysds.runtime.instructions.fed.InitFEDInstruction;
import org.apache.sysds.conf.DMLConfig;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class FederatedUrlParserTest 
{
	@Test(expected = IllegalArgumentException.class)
	public void parseIPAddress_negative_1() {
		// Fail if the input is empty.
		InitFEDInstruction.parseURL("");
	}

	@Test(expected = IllegalArgumentException.class)
	public void parseIPAddress_negative_2() {
		// Fail if there is no file specified.
		InitFEDInstruction.parseURL("192.13.4.2");
	}

	@Test(expected = IllegalArgumentException.class)
	public void parseIPAddress_negative_3() {
		// Fail if there is no file specified even if port is.
		InitFEDInstruction.parseURL("192.13.4.2:132");
	}

	@Test(expected = IllegalArgumentException.class)
	public void parseIPAddress_negative_4() {
		// Fail if the input clearly is an ipv4 address but malformed.
		InitFEDInstruction.parseURL("192.131111.4.2:132/file.txt");
	}

	@Test
	public void parseIPAddress_1() {
		// Parse Ip normally, with filepath and port.
		String[] values = InitFEDInstruction.parseURL("192.13.4.2:132/file.txt");
		assertEquals("192.13.4.2", values[0]);
		assertEquals("132", values[1]);
		assertEquals("file.txt", values[2]);
	}

	@Test
	public void parseIPAddress_2() {
		// Parse Ip normally, with filepath without port specified.
		String[] values = InitFEDInstruction.parseURL("123.123.41.22/file.txt");
		assertEquals("123.123.41.22", values[0]);
		assertEquals(DMLConfig.DEFAULT_FEDERATED_PORT, values[1]);
		assertEquals("file.txt", values[2]);
	}

	@Test
	public void parseURLAddress_1() {
		// Parse URL address fine with port.
		String[] values = InitFEDInstruction.parseURL("hello.com:132/file.txt");
		assertEquals("hello.com", values[0]);
		assertEquals("132", values[1]);
		assertEquals("file.txt", values[2]);
	}

	@Test
	public void parseURLAddress_2() {
		// parse URL without port.
		String[] values = InitFEDInstruction.parseURL("hello.com/file.txt");
		assertEquals("hello.com", values[0]);
		assertEquals(DMLConfig.DEFAULT_FEDERATED_PORT, values[1]);
		assertEquals("file.txt", values[2]);
	}

	@Test
	public void parseURLAddress_3() {
		// Parse URL with extended characters
		// Here Japanese: Hello.World
		String[] values = InitFEDInstruction.parseURL("今日は.世界/file.txt");
		assertEquals("今日は.世界", values[0]);
		assertEquals(DMLConfig.DEFAULT_FEDERATED_PORT, values[1]);
		assertEquals("file.txt", values[2]);
	}

	@Test
	public void parseFilePath_1() {
		// Parse Filepath even if it is nested.
		String[] values = InitFEDInstruction.parseURL("今日は.世界/fu_u_u/bar.txt");
		assertEquals("fu_u_u/bar.txt", values[2]);
	}

	@Test
	public void parseFilePath_2() {
		// Parse Filepath even if it is a folder out.
		String[] values = InitFEDInstruction.parseURL("今日は.世界/../bar.txt");
		assertEquals("../bar.txt", values[2]);
	}

	@Test
	public void parseFilePath_3() {
		// Parse Filepath with special characters.
		String[] values = InitFEDInstruction.parseURL("今日は.世界/../bar世界.txt");
		assertEquals("../bar世界.txt", values[2]);
	}

	@Test
	public void parseStaticFilePath_1() {
		// Parse Filepath even if it is nested.
		String[] values = InitFEDInstruction.parseURL("今日は.世界//fu_u_u/bar.txt");
		assertEquals("/fu_u_u/bar.txt", values[2]);
	}

	@Test
	public void parseStaticFilePath_2() {
		// Parse Filepath even if it is a folder out.
		String[] values = InitFEDInstruction.parseURL("今日は.世界//bar.txt");
		assertEquals("/bar.txt", values[2]);
	}

	@Test
	public void parseStaticFilePath_3() {
		// Parse Filepath with special characters.
		String[] values = InitFEDInstruction.parseURL("今日は.世界//bar世界.txt");
		assertEquals("/bar世界.txt", values[2]);
	}

	@Test(expected = IllegalArgumentException.class)
	public void parseQuery_negative_1() {
		// All Query flags should fail.
		InitFEDInstruction.parseURL("今日は.世界/../bar世界.txt?shouldnothappen");
	}

	@Test(expected = IllegalArgumentException.class)
	public void parseReference_negative_1() {
		// All Reference flags should fail.
		InitFEDInstruction.parseURL("今日は.世界/../bar世界.txt#shouldnothappen");
	}

	@Test(expected = IllegalArgumentException.class)
	public void parseReferenceAndQuery_negative_1() {
		// If both Reference and Query it should still should fail.
		InitFEDInstruction.parseURL("今日は.世界/../bar世界.txt?please#dont");
	}

	@Test
	public void checkDefaultPortIsValid() {
		int defaultPort = Integer.parseInt(DMLConfig.DEFAULT_FEDERATED_PORT);
		// The highest port number allowed.
		int IANA_limit = 49152;
		assertTrue(defaultPort <= IANA_limit);
		assertTrue(defaultPort > 0);
	}
}

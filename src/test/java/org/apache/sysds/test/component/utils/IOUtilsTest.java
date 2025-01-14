package org.apache.sysds.test.component.utils;

import static org.junit.Assert.assertEquals;

import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.junit.Test;

public class IOUtilsTest {
	

	@Test 
	public void getTo(){
		String in = ",\"yyy\"·4,";
		assertEquals(0, getTo(in, 0, ","));
		assertEquals(8, getTo(in, 1, ","));
		assertEquals("\"yyy\"·4", in.substring(1, getTo(in, 1, ",")));
	}

	@Test 
	public void getTo2(){
		String in = ",y,";
		assertEquals(0,getTo(in, 0, ","));
		assertEquals(2,getTo(in, 1, ","));
	}

	@Test 
	public void getTo3(){
		String in = "a,b,c";
		assertEquals("a",in.substring(0,getTo(in, 0, ",")));
		assertEquals("b",in.substring(2,getTo(in, 2, ",")));
		assertEquals("c",in.substring(4,getTo(in, 4, ",")));
	}

	@Test 
	public void getTo4(){
		String in = "a,\",\",c";
		assertEquals("a",in.substring(0,getTo(in, 0, ",")));
		assertEquals("\",\"",in.substring(2,getTo(in, 2, ",")));
	}

	private int getTo(String in, int from, String delim){
		return IOUtilFunctions.getTo(in, from, ",", in.length(), 1);
	}
}

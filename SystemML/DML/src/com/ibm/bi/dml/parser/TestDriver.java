/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.IOException;

import com.ibm.bi.dml.utils.DMLException;

 

 

public class TestDriver 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static void runExample(String program) throws IOException, ParseException, DMLException {
		System.out.println("===============================================");
		System.out.println("Processing program " + program);
		System.out.println("===============================================");
		System.out.println("After parsing ");
		
		DMLQLParser dmlqlParser = new DMLQLParser(program);
		run(dmlqlParser);
	}
	
	private static void runExampleFromFile(String fileName) throws IOException, ParseException, DMLException {
		System.out.println("===============================================");
		System.out.println("Processing program from file " + fileName);
		System.out.println("===============================================");
		System.out.println("After parsing ");
		
	//	DMLQLParser dmlqlParser = new DMLQLParser(new File(fileName), "UTF-8");
	//	run(dmlqlParser);
	}
	
	private static void run(DMLQLParser dmlqlParser) throws ParseException, DMLException, IOException {
		DMLProgram dmlp = dmlqlParser.parse();
		if (dmlp != null)
			System.out.println(dmlp.toString());
		System.out.println("===============================================");
		System.out.println("Hops");
		DMLTranslator dmlt = new DMLTranslator(dmlp);
		dmlt.validateParseTree(dmlp);
		dmlt.constructHops(dmlp);
		 
		dmlt.printHops(dmlp);
	 
		System.out.println("===============================================");
		
	}
	
	public static void main(String[] args) throws IOException, ParseException, DMLException {

		 
		/*
		 * Example 1:
		 * 
		 * A = B %*% C; D = A + E; 
		 */

		//String ex1 = "Read B ; Read C ; A = B %*% C ; Read E; D = A + E; Write A ; Write D;";
		 
		//runExample(ex1);
		/*
		 * Example 2:
		 * 
		 * C = A %*% B 
		 * D = A + B + E 
		 * write (C)
		 * write (D)
		 * 
		 */
		
		// String ex2 = "Read B ; Read A ; C = A %*% B ; Read E; D = A + B + E; Write C ; Write D;";
		
		// runExample(ex2); 
		 
		 /* 
		  * Run basic example from file
		  */
		 runExampleFromFile("scripts/example0.1.R");
		 
		 /* 
		  * Run example 2 from file
		  */
		 runExampleFromFile("scripts/example.txt");
	}

}

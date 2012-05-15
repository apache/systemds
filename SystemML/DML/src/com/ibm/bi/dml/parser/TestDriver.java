package com.ibm.bi.dml.parser;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.utils.DMLException;

 

 

public class TestDriver {

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

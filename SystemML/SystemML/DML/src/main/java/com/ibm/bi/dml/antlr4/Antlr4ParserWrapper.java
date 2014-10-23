/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.antlr4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

import org.antlr.v4.runtime.BailErrorStrategy;
import org.antlr.v4.runtime.DefaultErrorStrategy;
import org.antlr.v4.runtime.atn.PredictionMode;
import org.antlr.v4.runtime.misc.ParseCancellationException;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import com.ibm.bi.dml.antlr4.DmlParser.DmlprogramContext;
import com.ibm.bi.dml.antlr4.DmlParser.FunctionStatementContext;
import com.ibm.bi.dml.antlr4.DmlParser.StatementContext;
import com.ibm.bi.dml.antlr4.DmlSyntacticErrorListener.CustomDmlErrorListener;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.FunctionStatement;
import com.ibm.bi.dml.parser.FunctionStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.ImportStatement;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.ParForStatement;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;

public class Antlr4ParserWrapper {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static CustomDmlErrorListener ERROR_LISTENER_INSTANCE = new CustomDmlErrorListener();
	
	public static String currentPath = null; 
	public static HashMap<String,String> argVals = null; 
	
	
	public static com.ibm.bi.dml.parser.StatementBlock getStatementBlock(com.ibm.bi.dml.parser.Statement current) {
		com.ibm.bi.dml.parser.StatementBlock blk = null;
		if(current instanceof ParForStatement) {
    		blk = new com.ibm.bi.dml.parser.ParForStatementBlock();
    		blk.addStatement(current);
    	}
		else if(current instanceof ForStatement) {
    		blk = new com.ibm.bi.dml.parser.ForStatementBlock();
    		blk.addStatement(current);
    	}
    	else if(current instanceof IfStatement) {
    		blk = new com.ibm.bi.dml.parser.IfStatementBlock();
    		blk.addStatement(current);
    	}
    	else if(current instanceof WhileStatement) {
    		blk = new com.ibm.bi.dml.parser.WhileStatementBlock();
    		blk.addStatement(current);
    	}
    	else {
    		// This includes ImportStatement
    		blk = new StatementBlock();
    		blk.addStatement(current);
    	}
		return blk;
	}
	
	public void cleanUpState() {
		ERROR_LISTENER_INSTANCE = new CustomDmlErrorListener();
		currentPath = null;
		argVals = null;
		com.ibm.bi.dml.antlr4.DmlSyntacticErrorListener.atleastOneError = false;
		com.ibm.bi.dml.antlr4.DmlSyntacticErrorListener.currentFileName = new Stack<String>();
	}
	
	/**
	 * 
	 * @param fileName
	 * @return null if atleast one error
	 */
	public DMLProgram parse(String fileName) {
		DMLProgram dmlPgm = null;
		try {
			if(!(new File(fileName)).exists()) {
				System.err.println("ERROR: Cannot open file:" + fileName);
				return null;
			}
			
			org.antlr.v4.runtime.ANTLRInputStream in;
			try {
				in = new org.antlr.v4.runtime.ANTLRInputStream(new java.io.FileInputStream(fileName));
			} catch (FileNotFoundException e) {
				System.err.println("ERROR: Cannot find file:" + fileName);
				return null;
			} catch (IOException e) {
				System.err.println("ERROR: Cannot open file:" + fileName);
				return null;
			}
			
			com.ibm.bi.dml.antlr4.DmlLexer lexer = new com.ibm.bi.dml.antlr4.DmlLexer(in);
			org.antlr.v4.runtime.CommonTokenStream tokens = new org.antlr.v4.runtime.CommonTokenStream(lexer);
			com.ibm.bi.dml.antlr4.DmlParser antlr4Parser = new com.ibm.bi.dml.antlr4.DmlParser(tokens);
			
			DmlprogramContext ast = null;
			boolean tryOptimizedParsing = false; // For now no optimization, since it is not able to parse integer value. 
			
			if(tryOptimizedParsing) {
				// Try faster and simpler SLL
				antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.SLL);
				antlr4Parser.removeErrorListeners();
				antlr4Parser.setErrorHandler(new BailErrorStrategy());
				try{
					ast = antlr4Parser.dmlprogram();
					// If successful, no need to try out full LL(*) ... SLL was enough
				}
				catch(ParseCancellationException ex) {
					// Error occurred, so now try full LL(*) for better error messages
					tokens.reset();
					antlr4Parser.reset();
					com.ibm.bi.dml.antlr4.DmlSyntacticErrorListener.currentFileName.push(fileName);
					// Set our custom error listener
					antlr4Parser.addErrorListener(ERROR_LISTENER_INSTANCE);
					antlr4Parser.setErrorHandler(new DefaultErrorStrategy());
					antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.LL);
					ast = antlr4Parser.dmlprogram();
				}
			}
			else {
				// Set our custom error listener
				antlr4Parser.removeErrorListeners();
				antlr4Parser.addErrorListener(ERROR_LISTENER_INSTANCE);
				com.ibm.bi.dml.antlr4.DmlSyntacticErrorListener.currentFileName.push(fileName);
				
				// Now do the parsing
				ast = antlr4Parser.dmlprogram();
			}
			
			org.antlr.v4.runtime.tree.ParseTree tree = ast;
				
			// And also do syntactic validation
			org.antlr.v4.runtime.tree.ParseTreeWalker walker = new ParseTreeWalker();
	        com.ibm.bi.dml.antlr4.DmlSyntacticValidator validator = new DmlSyntacticValidator();
	        walker.walk(validator, tree);
	        
	        com.ibm.bi.dml.antlr4.DmlSyntacticErrorListener.currentFileName.pop();
	        
	        if(com.ibm.bi.dml.antlr4.DmlSyntacticErrorListener.atleastOneError) {
	        	return null;
	        }
	        
	        dmlPgm = new DMLProgram();
	        
	     // First add all the functions
	        for(FunctionStatementContext fn : ast.functionBlocks) {
	        	FunctionStatementBlock functionStmtBlk = new FunctionStatementBlock();
	        	functionStmtBlk.addStatement(fn.info.stmt);
	        	try {
	        		// TODO: currently the logic of nested namespace is not clear.
	        		String namespace = DMLProgram.DEFAULT_NAMESPACE;
					dmlPgm.addFunctionStatementBlock(namespace, fn.info.functionName, functionStmtBlk);
				} catch (LanguageException e) {
					System.err.println("ERROR: line: " + fn.start.getLine() + ":" + fn.start.getCharPositionInLine() + " cannot process the function " + fn.info.functionName);
					return null;
				}
	        }
	        
	        // First add all the statements
	        for(StatementContext stmtCtx : ast.blocks) {
	        	com.ibm.bi.dml.parser.Statement current = stmtCtx.info.stmt;
	        	if(current == null) {
	        		System.err.println("ERROR: line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot process the statement");
					return null;
	        	}
	        	
	        	if(current instanceof ImportStatement) {
	        		// Handle import statements separately
	        		if(stmtCtx.info.namespaces != null) {
	        			// Add the DMLProgram entries into current program
		        		for(Map.Entry<String, DMLProgram> entry : stmtCtx.info.namespaces.entrySet()) {
		        			// Don't add DMLProgram into the current program, just add function statements
		        			// dmlPgm.getNamespaces().put(entry.getKey(), entry.getValue());
		        			// Add function statements to current dml program
		        			DMLProgram importedPgm = entry.getValue();
		        			
		        			try {
								for(FunctionStatementBlock importedFnBlk : importedPgm.getFunctionStatementBlocks()) {
									if(importedFnBlk.getStatements() != null && importedFnBlk.getStatements().size() == 1) {
										String functionName = ((FunctionStatement)importedFnBlk.getStatement(0)).getName();
										dmlPgm.addFunctionStatementBlock(entry.getKey(), functionName, importedFnBlk);
									}
									else {
										System.err.println("ERROR: line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " incorrect number of functions in the imported function block .... strange");
					    				return null;
									}
								}
								if(importedPgm.getStatementBlocks() != null && importedPgm.getStatementBlocks().size() > 0) {
									System.err.println("WARNING: only the functions can be imported from the namespace " + entry.getKey());
								}
							} catch (LanguageException e) {
								System.err.println("ERROR: line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot import functions from the file in the import statement");
			    				return null;
							}
		        		}
	        		}
	        		else {
	        			System.err.println("ERROR: line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot process the import statement");
	    				return null;
	        		}
	        	}
	        	
	        	// Now wrap statement into individual statement block
	        	// merge statement will take care of merging these blocks
	        	dmlPgm.addStatementBlock(getStatementBlock(current));
	        }
	        
	        dmlPgm.mergeStatementBlocks();
		}
		catch(Exception e) {
			//e.printStackTrace();
			System.err.println("ERROR: Couldnot parse the file " + fileName);
		}		
        
        return dmlPgm;
	}
}

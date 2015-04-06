/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser.python;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.BailErrorStrategy;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.DefaultErrorStrategy;
import org.antlr.v4.runtime.atn.PredictionMode;
import org.antlr.v4.runtime.misc.ParseCancellationException;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.FunctionStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.ImportStatement;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.ParForStatement;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.ParseException;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.parser.antlr4.DMLParserWrapper;
import com.ibm.bi.dml.parser.python.PydmlParser.FunctionStatementContext;
import com.ibm.bi.dml.parser.python.PydmlParser.PmlprogramContext;
import com.ibm.bi.dml.parser.python.PydmlParser.StatementContext;
import com.ibm.bi.dml.parser.python.PydmlSyntacticErrorListener.CustomDmlErrorListener;

/**
 * Logic of this wrapper is similar to DMLParserWrapper.
 * 
 * Note: ExpressionInfo and StatementInfo are simply wrapper objects and are reused in both DML and PyDML parsers.
 *
 */
public class PyDMLParserWrapper {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
			"US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public static CustomDmlErrorListener ERROR_LISTENER_INSTANCE = new CustomDmlErrorListener();

	public static String currentPath = null; 
	public static HashMap<String,String> argVals = null; 

	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());

	/**
	 * Custom wrapper to convert statement into statement blocks. Called by doParse and in PydmlSyntacticValidator for for, parfor, while, ...
	 * @param current a statement
	 * @return corresponding statement block
	 */
	public static StatementBlock getStatementBlock(com.ibm.bi.dml.parser.Statement current) {
		StatementBlock blk = null;
		if(current instanceof ParForStatement) {
			blk = new ParForStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof ForStatement) {
			blk = new ForStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof IfStatement) {
			blk = new IfStatementBlock();
			blk.addStatement(current);
		}
		else if(current instanceof WhileStatement) {
			blk = new WhileStatementBlock();
			blk.addStatement(current);
		}
		else {
			// This includes ImportStatement
			blk = new StatementBlock();
			blk.addStatement(current);
		}
		return blk;
	}

	/**
	 * This is needed because unit test is invoked in single jvm.
	 */
	private void cleanUpState() {
		ERROR_LISTENER_INSTANCE = new CustomDmlErrorListener();
		currentPath = null;
		argVals = null;
		PydmlSyntacticErrorListener.atleastOneError = false;
		PydmlSyntacticErrorListener.currentFileName = new Stack<String>();
	}

	/**
	 * Parses the passed file with command line parameters. You can either pass both (local file) or just dmlScript (hdfs) or just file name (import command)
	 * @param fileName either full path or null --> only used for better error handling
	 * @param dmlScript required
	 * @param argVals
	 * @return
	 * @throws ParseException
	 */
	public DMLProgram parse(String fileName, String dmlScript, HashMap<String,String> argVals) throws ParseException {
		DMLProgram prog = null;
		
		if(dmlScript == null || dmlScript.trim().compareTo("") == 0) {
			throw new ParseException("Incorrect usage of parse. Please pass dmlScript not just filename");
		}
		
		// Set the pipeline required for ANTLR parsing
		PyDMLParserWrapper parser = new PyDMLParserWrapper();
		PyDMLParserWrapper.argVals = argVals;
		prog = parser.doParse(fileName, dmlScript);
		parser.cleanUpState();
		
		if(prog == null) {
			throw new ParseException("One or more errors found during parsing. Cannot proceed ahead.");
		}
		return prog;
		// Use //+ "Here is the parse tree:\n" + tree.toStringTree(antlr4Parser).replaceAll("expression ", "")
	}

	/**
	 * This function is supposed to be called directly only from PydmlSyntacticValidator when it encounters 'import'
	 * @param fileName
	 * @return null if atleast one error
	 */
	public DMLProgram doParse(String fileName, String dmlScript) throws ParseException {
		DMLProgram dmlPgm = null;
		
		ANTLRInputStream in;
		try {
			if(dmlScript == null) {
				dmlScript = DMLParserWrapper.readDMLScript(fileName);
			}
			
			InputStream stream = new ByteArrayInputStream(dmlScript.getBytes());
			in = new org.antlr.v4.runtime.ANTLRInputStream(stream);
//			else {
//				if(!(new File(fileName)).exists()) {
//					throw new ParseException("ERROR: Cannot open file:" + fileName);
//				}
//				in = new ANTLRInputStream(new FileInputStream(fileName));
//			}
		} catch (FileNotFoundException e) {
			throw new ParseException("ERROR: Cannot find file:" + fileName);
		} catch (IOException e) {
			throw new ParseException("ERROR: Cannot open file:" + fileName);
		} catch (LanguageException e) {
			throw new ParseException("ERROR: " + e.getMessage());
		}

		PmlprogramContext ast = null;
		
		try {
			PydmlLexer lexer = new PydmlLexer(in);
			CommonTokenStream tokens = new CommonTokenStream(lexer);
			PydmlParser antlr4Parser = new PydmlParser(tokens);
			
			boolean tryOptimizedParsing = false; // For now no optimization, since it is not able to parse integer value. 
	
			if(tryOptimizedParsing) {
				// Try faster and simpler SLL
				antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.SLL);
				antlr4Parser.removeErrorListeners();
				antlr4Parser.setErrorHandler(new BailErrorStrategy());
				try{
					ast = antlr4Parser.pmlprogram();
					// If successful, no need to try out full LL(*) ... SLL was enough
				}
				catch(ParseCancellationException ex) {
					// Error occurred, so now try full LL(*) for better error messages
					tokens.reset();
					antlr4Parser.reset();
					if(fileName != null) {
						PydmlSyntacticErrorListener.currentFileName.push(fileName);
					}
					else {
						PydmlSyntacticErrorListener.currentFileName.push("MAIN_SCRIPT");
					}
					// Set our custom error listener
					antlr4Parser.addErrorListener(ERROR_LISTENER_INSTANCE);
					antlr4Parser.setErrorHandler(new DefaultErrorStrategy());
					antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.LL);
					ast = antlr4Parser.pmlprogram();
				}
			}
			else {
				// Set our custom error listener
				antlr4Parser.removeErrorListeners();
				antlr4Parser.addErrorListener(ERROR_LISTENER_INSTANCE);
				PydmlSyntacticErrorListener.currentFileName.push(fileName);
	
				// Now do the parsing
				ast = antlr4Parser.pmlprogram();
			}
		}
		catch(Exception e) {
			throw new ParseException("ERROR: Cannot parse the program:" + fileName);
		}
		

		try {
			// Now convert the parse tree into DMLProgram
			// Do syntactic validation while converting 
			ParseTree tree = ast;
			// And also do syntactic validation
			ParseTreeWalker walker = new ParseTreeWalker();
			PydmlSyntacticValidator validator = new PydmlSyntacticValidator();
			walker.walk(validator, tree);
			PydmlSyntacticErrorListener.currentFileName.pop();
			if(PydmlSyntacticErrorListener.atleastOneError) {
				return null;
			}
			dmlPgm = createDMLProgram(ast);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new ParseException("ERROR: Cannot translate the parse tree into DMLProgram");
		}
		
		return dmlPgm;
	}

	private DMLProgram createDMLProgram(PmlprogramContext ast) {

		DMLProgram dmlPgm = new DMLProgram();

		// First add all the functions
		for(FunctionStatementContext fn : ast.functionBlocks) {
			FunctionStatementBlock functionStmtBlk = new FunctionStatementBlock();
			functionStmtBlk.addStatement(fn.info.stmt);
			try {
				// TODO: currently the logic of nested namespace is not clear.
				String namespace = DMLProgram.DEFAULT_NAMESPACE;
				dmlPgm.addFunctionStatementBlock(namespace, fn.info.functionName, functionStmtBlk);
			} catch (LanguageException e) {
				LOG.error("line: " + fn.start.getLine() + ":" + fn.start.getCharPositionInLine() + " cannot process the function " + fn.info.functionName);
				return null;
			}
		}

		// Then add all the statements
		for(StatementContext stmtCtx : ast.blocks) {
			Statement current = stmtCtx.info.stmt;
			if(current == null) {
				LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot process the statement");
				return null;
			}
			
			// Ignore Newline logic 
			if(current.isEmptyNewLineStatement()) {
				continue;
			}

			if(current instanceof ImportStatement) {
				// Handle import statements separately
				if(stmtCtx.info.namespaces != null) {
					// Add the DMLProgram entries into current program
					for(Map.Entry<String, DMLProgram> entry : stmtCtx.info.namespaces.entrySet()) {
						dmlPgm.getNamespaces().put(entry.getKey(), entry.getValue());
//						// Don't add DMLProgram into the current program, just add function statements
//						// dmlPgm.getNamespaces().put(entry.getKey(), entry.getValue());
//						// Add function statements to current dml program
//						DMLProgram importedPgm = entry.getValue();
//
//						try {
//							for(FunctionStatementBlock importedFnBlk : importedPgm.getFunctionStatementBlocks()) {
//								if(importedFnBlk.getStatements() != null && importedFnBlk.getStatements().size() == 1) {
//									String functionName = ((FunctionStatement)importedFnBlk.getStatement(0)).getName();
//									dmlPgm.addFunctionStatementBlock(entry.getKey(), functionName, importedFnBlk);
//								}
//								else {
//									LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " incorrect number of functions in the imported function block .... strange");
//									return null;
//								}
//							}
//							if(importedPgm.getStatementBlocks() != null && importedPgm.getStatementBlocks().size() > 0) {
//								LOG.warn("Only the functions can be imported from the namespace " + entry.getKey());
//							}
//						} catch (LanguageException e) {
//							LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot import functions from the file in the import statement");
//							return null;
//						}
					}
				}
				else {
					LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot process the import statement");
					return null;
				}
			}

			// Now wrap statement into individual statement block
			// merge statement will take care of merging these blocks
			dmlPgm.addStatementBlock(getStatementBlock(current));
		}

		dmlPgm.mergeStatementBlocks();
		return dmlPgm;
	}
}

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

package org.apache.sysml.parser.dml;

import java.io.ByteArrayInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

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
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.ImportStatement;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParseException;
import org.apache.sysml.parser.common.CustomErrorListener;
import org.apache.sysml.parser.common.CustomErrorListener.ParseIssue;
import org.apache.sysml.parser.dml.DmlParser.FunctionStatementContext;
import org.apache.sysml.parser.dml.DmlParser.ProgramrootContext;
import org.apache.sysml.parser.dml.DmlParser.StatementContext;

/**
 * This is the main entry point for the Antlr4 parser.
 * Dml.g4 is the grammar file which enforces syntactic structure of DML program. 
 * DmlSyntaticValidator on other hand captures little bit of semantic as well as does the job of translation of Antlr AST to DMLProgram.
 * At a high-level, DmlSyntaticValidator implements call-back methods that are called by walker.walk(validator, tree)
 * The callback methods are of two type: enterSomeASTNode() and exitSomeASTNode()
 * It is important to note that almost every node in AST has either ExpressionInfo or StatementInfo object associated with it.
 * The key design decision is that while "exiting" the node (i.e. callback to exitSomeASTNode), we use information in given AST node and construct an object of type Statement or Expression and put it in StatementInfo or ExpressionInfo respectively. 
 * This way it avoids any bugs due to lookahead and one only has to "think as an AST node", thereby making any changes to parse code much simpler :)
 * 
 * Note: to add additional builtin function, one only needs to modify DmlSyntaticValidator (which is java file and provides full Eclipse tooling support) not g4. 
 * 
 * To separate logic of semantic validation, DmlSyntaticValidatorHelper contains functions that do semantic validation. Currently, there is no semantic validation as most of it is delegated to subsequent validation phase. 
 * 
 * Whenever there is a parse error, it goes through CustomErrorListener. This allows us to pipe the error messages to any future pipeline as well as control the format in an elegant manner.
 * There are three types of messages passed:
 * - Syntactic errors: When passed DML script doesnot conform to syntatic structure enforced by Dml.g4
 * - Validation errors: Errors due to translation of AST to  DMLProgram
 * - Validation warnings: Messages to inform users that there might be potential bug in their program
 * 
 * As of this moment, Antlr4ParserWrapper is stateful and cannot be multithreaded. This is not big deal because each users calls SystemML in different process.
 * If in future we intend to make it multi-threaded, look at cleanUpState method and resolve the dependency accordingly.    
 *
 */
public class DMLParserWrapper extends AParserWrapper
{
	private static final Log LOG = LogFactory.getLog(DMLScript.class.getName());

	/**
	 * Parses the passed file with command line parameters. You can either pass both (local file) or just dmlScript (hdfs) or just file name (import command)
	 * @param fileName either full path or null --> only used for better error handling
	 * @param dmlScript required
	 * @param argVals
	 * @return
	 * @throws ParseException
	 */
	@Override
	public DMLProgram parse(String fileName, String dmlScript, Map<String,String> argVals) throws ParseException {
		DMLProgram prog = doParse(fileName, dmlScript, null, argVals);
		
		return prog;
	}
	
	/**
	 * This function is supposed to be called directly only from DmlSyntacticValidator when it encounters 'import'
	 * @param fileName script file name
	 * @param dmlScript script file contents
	 * @param sourceNamespace namespace from source statement
	 * @param argVals script arguments
	 * @return null if at least one error
	 * @throws ParseException
	 */
	public DMLProgram doParse(String fileName, String dmlScript, String sourceNamespace, Map<String,String> argVals) throws ParseException {
		DMLProgram dmlPgm = null;
		
		ANTLRInputStream in;
		try {
			if(dmlScript == null) {
				dmlScript = readDMLScript(fileName, LOG);
			}
			
			InputStream stream = new ByteArrayInputStream(dmlScript.getBytes());
			in = new ANTLRInputStream(stream);
		} catch (FileNotFoundException e) {
			throw new ParseException("Cannot find file: " + fileName, e);
		} catch (IOException e) {
			throw new ParseException("Cannot open file: " + fileName, e);
		} catch (LanguageException e) {
			throw new ParseException(e.getMessage(), e);
		}

		ProgramrootContext ast = null;
		CustomErrorListener errorListener = new CustomErrorListener();
		
		try {
			DmlLexer lexer = new DmlLexer(in);
			CommonTokenStream tokens = new CommonTokenStream(lexer);
			DmlParser antlr4Parser = new DmlParser(tokens);
			
			boolean tryOptimizedParsing = false; // For now no optimization, since it is not able to parse integer value. 
	
			if(tryOptimizedParsing) {
				// Try faster and simpler SLL
				antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.SLL);
				antlr4Parser.removeErrorListeners();
				antlr4Parser.setErrorHandler(new BailErrorStrategy());
				try{
					ast = antlr4Parser.programroot();
					// If successful, no need to try out full LL(*) ... SLL was enough
				}
				catch(ParseCancellationException ex) {
					// Error occurred, so now try full LL(*) for better error messages
					tokens.reset();
					antlr4Parser.reset();
					if(fileName != null) {
						errorListener.setCurrentFileName(fileName);
					}
					else {
						errorListener.setCurrentFileName("MAIN_SCRIPT");
					}
					// Set our custom error listener
					antlr4Parser.addErrorListener(errorListener);
					antlr4Parser.setErrorHandler(new DefaultErrorStrategy());
					antlr4Parser.getInterpreter().setPredictionMode(PredictionMode.LL);
					ast = antlr4Parser.programroot();
				}
			}
			else {
				// Set our custom error listener
				antlr4Parser.removeErrorListeners();
				antlr4Parser.addErrorListener(errorListener);
				errorListener.setCurrentFileName(fileName);
	
				// Now do the parsing
				ast = antlr4Parser.programroot();
			}
		}
		catch(Exception e) {
			throw new ParseException("ERROR: Cannot parse the program:" + fileName, e);
		}
		

		// Now convert the parse tree into DMLProgram
		// Do syntactic validation while converting 
		ParseTree tree = ast;
		// And also do syntactic validation
		ParseTreeWalker walker = new ParseTreeWalker();
		DmlSyntacticValidator validator = new DmlSyntacticValidator(errorListener, argVals, sourceNamespace);
		walker.walk(validator, tree);
		errorListener.unsetCurrentFileName();
		if (errorListener.isAtleastOneError()) {
			List<ParseIssue> parseIssues = errorListener.getParseIssues();
			throw new ParseException(parseIssues, dmlScript);
		}
		dmlPgm = createDMLProgram(ast, sourceNamespace);
		
		return dmlPgm;
	}
	
	private DMLProgram createDMLProgram(ProgramrootContext ast, String sourceNamespace) {

		DMLProgram dmlPgm = new DMLProgram();
		String namespace = (sourceNamespace != null && sourceNamespace.length() > 0) ? sourceNamespace : DMLProgram.DEFAULT_NAMESPACE;
		dmlPgm.getNamespaces().put(namespace, dmlPgm);

		// First add all the functions
		for(FunctionStatementContext fn : ast.functionBlocks) {
			FunctionStatementBlock functionStmtBlk = new FunctionStatementBlock();
			functionStmtBlk.addStatement(fn.info.stmt);
			try {
				dmlPgm.addFunctionStatementBlock(namespace, fn.info.functionName, functionStmtBlk);
			} catch (LanguageException e) {
				LOG.error("line: " + fn.start.getLine() + ":" + fn.start.getCharPositionInLine() + " cannot process the function " + fn.info.functionName);
				return null;
			}
		}

		// Then add all the statements
		for(StatementContext stmtCtx : ast.blocks) {
			org.apache.sysml.parser.Statement current = stmtCtx.info.stmt;
			if(current == null) {
				LOG.error("line: " + stmtCtx.start.getLine() + ":" + stmtCtx.start.getCharPositionInLine() + " cannot process the statement");
				return null;
			}

			if(current instanceof ImportStatement) {
				// Handle import statements separately
				if(stmtCtx.info.namespaces != null) {
					// Add the DMLProgram entries into current program
					for(Map.Entry<String, DMLProgram> entry : stmtCtx.info.namespaces.entrySet()) {
						// TODO handle namespace key already exists for different program value instead of overwriting
						DMLProgram prog = entry.getValue();
						if (prog != null && prog.getNamespaces().size() > 0) {
							dmlPgm.getNamespaces().put(entry.getKey(), prog);
						}
						
						// Add dependent programs (handle imported script that also imports scripts)
						for(Map.Entry<String, DMLProgram> dependency : entry.getValue().getNamespaces().entrySet()) {
							String depNamespace = dependency.getKey();
							DMLProgram depProgram = dependency.getValue();
							if (dmlPgm.getNamespaces().get(depNamespace) == null) {
								dmlPgm.getNamespaces().put(depNamespace, depProgram);
							}
						}
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

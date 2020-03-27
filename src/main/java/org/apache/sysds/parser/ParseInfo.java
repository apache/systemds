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

package org.apache.sysds.parser;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.misc.Interval;

public interface ParseInfo {

	public void setBeginLine(int beginLine);

	public void setBeginColumn(int beginColumn);

	public void setEndLine(int endLine);

	public void setEndColumn(int endColumn);

	public void setText(String text);
	
	public void setFilename(String filename);

	public int getBeginLine();

	public int getBeginColumn();

	public int getEndLine();

	public int getEndColumn();

	public String getText();
	
	public String getFilename();

	public static ParseInfo ctxAndFilenameToParseInfo(ParserRuleContext ctx, String fname) {
		ParseInfo pi = new ParseInfo() {
			private int beginLine;
			private int beginColumn;
			private int endLine;
			private int endColumn;
			private String text;
			private String filename;

			@Override
			public void setBeginLine(int beginLine) {
				this.beginLine = beginLine;
			}

			@Override
			public void setBeginColumn(int beginColumn) {
				this.beginColumn = beginColumn;
			}

			@Override
			public void setEndLine(int endLine) {
				this.endLine = endLine;
			}

			@Override
			public void setEndColumn(int endColumn) {
				this.endColumn = endColumn;
			}

			@Override
			public void setText(String text) {
				this.text = text;
			}

			@Override
			public void setFilename(String filename) {
				this.filename = filename;
			}
			@Override
			public int getBeginLine() {
				return beginLine;
			}

			@Override
			public int getBeginColumn() {
				return beginColumn;
			}

			@Override
			public int getEndLine() {
				return endLine;
			}

			@Override
			public int getEndColumn() {
				return endColumn;
			}

			@Override
			public String getText() {
				return text;
			}
			
			@Override
			public String getFilename() {
				return filename;
			}
		};
		pi.setBeginLine(ctx.start.getLine());
		pi.setBeginColumn(ctx.start.getCharPositionInLine());
		pi.setEndLine(ctx.stop.getLine());
		pi.setEndColumn(ctx.stop.getCharPositionInLine());
		// preserve whitespace if possible
		if ((ctx.start != null) && (ctx.stop != null) && (ctx.start.getStartIndex() != -1)
				&& (ctx.stop.getStopIndex() != -1) && (ctx.start.getStartIndex() <= ctx.stop.getStopIndex())
				&& (ctx.start.getInputStream() != null)) {
			String text = ctx.start.getInputStream()
					.getText(Interval.of(ctx.start.getStartIndex(), ctx.stop.getStopIndex()));
			if (text != null) {
				text = text.trim();
			}
			pi.setText(text);
		} else {
			String text = ctx.getText();
			if (text != null) {
				text = text.trim();
			}
			pi.setText(text);
		}
		pi.setFilename(fname);
		return pi;
	}
}

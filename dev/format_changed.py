#!/usr/bin/env python3
# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
#
# Line-scoped Java formatting against dev/CodeStyle_eclipse.xml.
#
# The Eclipse formatter only works on whole files, and the existing tree is not
# fully formatter-clean, so formatting a whole edited file would flag lines the
# PR never touched. This script therefore formats each changed file but keeps
# only the formatting changes that fall on the lines the PR edited (like
# clang-format-diff): it diffs the original against the fully-formatted version
# and restricts the result to the changed line ranges.
#
#   --check (default): print the changed-line formatting fixes and exit 1 if any.
#   --fix            : apply only the changed-line formatting fixes in place.
#
# Usage: dev/format_changed.py [--check|--fix] [base-ref]
#
import difflib
import fnmatch
import os
import re
import subprocess
import sys

FMT_VERSION = "2.24.1"
CONFIG = "dev/CodeStyle_eclipse.xml"
EXCLUDE_FILE = "dev/format-exclude.txt"
SRC_PREFIX = r"src/(main|test)/java/"
SRC_RE = re.compile(r"^" + SRC_PREFIX + r".+\.java$")


# --- small process / IO helpers ------------------------------------------------

def git(*args, check=True):
	# core.quotePath=false so non-ASCII paths are emitted verbatim (not \NNN
	# escaped), otherwise SRC_RE would silently skip them and bypass the check.
	proc = subprocess.run(["git", "-c", "core.quotePath=false", *args],
			capture_output=True, text=True)
	if check and proc.returncode != 0:
		sys.exit(f"ERROR: `git {' '.join(args)}` failed:\n{proc.stderr.strip()}")
	return proc.stdout


def read_text(path):
	# newline="" keeps line endings byte-exact so a check-mode restore (or a
	# fix-mode non-flagged file) round-trips without CRLF->LF rewrites.
	with open(path, encoding="utf-8", newline="") as fh:
		return fh.read()


def write_text(path, text):
	with open(path, "w", encoding="utf-8", newline="") as fh:
		fh.write(text)


def strip_src_prefix(path):
	return re.sub(r"^" + SRC_PREFIX, "", path)


# --- exemption list ------------------------------------------------------------

def load_excludes():
	# glob patterns of files exempt from the style check, one per line
	patterns = []
	if os.path.exists(EXCLUDE_FILE):
		with open(EXCLUDE_FILE, encoding="utf-8") as fh:
			for line in fh:
				s = line.strip()
				if s and not s.startswith("#"):
					patterns.append(s)
	return patterns


def is_excluded(path, patterns):
	base = os.path.basename(path)
	return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(base, p) for p in patterns)


# --- git ref / file discovery --------------------------------------------------

def ref_exists(ref):
	return subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref + "^{commit}"],
			capture_output=True).returncode == 0


def resolve_base(explicit):
	if explicit:
		if not ref_exists(explicit):
			sys.exit(f"Base ref not found: {explicit}")
		return explicit
	for ref in ("upstream/main", "origin/main", "main"):
		if ref_exists(ref):
			return ref
	sys.exit("Could not determine a base ref; pass one explicitly.")


def merge_base(base):
	mb = git("merge-base", base, "HEAD", check=False).strip()
	return mb or base


def base_has(mergebase, path):
	return subprocess.run(["git", "cat-file", "-e", f"{mergebase}:{path}"],
			capture_output=True).returncode == 0


def discover_files(mergebase):
	tracked = [f for f in git("diff", "--name-only", "--diff-filter=ACMR",
			mergebase, "--").splitlines() if SRC_RE.match(f)]
	untracked = [f for f in git("ls-files", "--others", "--exclude-standard").splitlines()
			if SRC_RE.match(f)]
	seen = set(tracked)
	files = tracked + [f for f in untracked if f not in seen]

	patterns = load_excludes()
	skipped = [f for f in files if is_excluded(f, patterns)]
	files = [f for f in files if not is_excluded(f, patterns)]
	return files, skipped, set(untracked)


# --- changed-line ranges (pure parsing, unit-tested) ---------------------------

def _hunk_range(header):
	# parse the new-side (+) range of a `@@ -a,b +c,d @@` unified-diff header;
	# returns a 1-based inclusive (start, end), or None for pure deletions / non-headers
	m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", header)
	if not m:
		return None
	start = int(m.group(1))
	count = int(m.group(2)) if m.group(2) is not None else 1
	return (start, start + count - 1) if count > 0 else None


def parse_hunks(diff_text):
	# all new-side ranges in a single-file unified diff (used by tests and below)
	ranges = []
	for line in diff_text.splitlines():
		r = _hunk_range(line)
		if r is not None:
			ranges.append(r)
	return ranges


def changed_ranges(mergebase, files):
	# one batched `git diff` for all files, split per file on the `+++ b/` header
	ranges = {f: [] for f in files}
	if not files:
		return ranges
	out = git("diff", "-U0", mergebase, "--", *files)
	current = None
	for line in out.splitlines():
		if line.startswith("+++ b/"):
			current = line[len("+++ b/"):]
		elif current is not None:
			r = _hunk_range(line)
			if r is not None and current in ranges:
				ranges[current].append(r)
	return ranges


def overlaps(i1, i2, ranges):
	# original-side region [i1, i2) (0-based half-open) vs 1-based inclusive ranges
	lo, hi = i1 + 1, i2  # convert to 1-based inclusive; insert (i1==i2) -> lo>hi
	for (s, e) in ranges:
		if i1 == i2:  # pure insertion between original lines i1 and i1+1
			if s - 1 <= i1 <= e:
				return True
		elif not (hi < s or lo > e):
			return True
	return False


def line_scoped_result(original_text, formatted_text, ranges):
	# reconstruct a file that keeps original content everywhere except on the
	# formatting hunks that intersect the PR-edited ranges; returns the new text
	# plus the kept hunks (for reporting). Pure function -- no IO.
	a = original_text.splitlines(keepends=True)
	b = formatted_text.splitlines(keepends=True)
	sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
	result = []
	kept = []
	for tag, i1, i2, j1, j2 in sm.get_opcodes():
		if tag == "equal":
			result.extend(a[i1:i2])
		elif overlaps(i1, i2, ranges):
			result.extend(b[j1:j2])
			kept.append((i1, len(a[i1:i2]), len(b[j1:j2]), a[i1:i2], b[j1:j2]))
		else:
			result.extend(a[i1:i2])
	return "".join(result), kept


# --- formatter -----------------------------------------------------------------

def run_formatter(files):
	includes = ",".join(strip_src_prefix(f) for f in files)
	subprocess.run(["mvn", "-q", "-ntp", "-B",
			f"net.revelc.code.formatter:formatter-maven-plugin:{FMT_VERSION}:format",
			f"-Dconfigfile={os.getcwd()}/{CONFIG}",
			"-Dmaven.compiler.source=17", "-Dmaven.compiler.target=17",
			f"-Dformatter.includes={includes}"], check=True)


def print_report(path, kept):
	print(f"\n--- a/{path}")
	print(f"+++ b/{path}")
	for i1, n_old, n_new, olds, news in kept:
		print(f"@@ -{i1 + 1},{n_old} +{i1 + 1},{n_new} @@ (changed lines)")
		for ln in olds:
			print("-" + ln.rstrip("\n"))
		for ln in news:
			print("+" + ln.rstrip("\n"))


# --- CLI -----------------------------------------------------------------------

def parse_args(argv):
	mode = "check"
	positionals = []
	for a in argv:
		if a == "--fix":
			mode = "fix"
		elif a == "--check":
			mode = "check"
		elif a.startswith("-"):
			sys.exit(f"Unknown option: {a}  (usage: format_changed.py [--check|--fix] [base-ref])")
		else:
			positionals.append(a)
	if len(positionals) > 1:
		sys.exit(f"Expected at most one base ref, got: {positionals}")
	return mode, (positionals[0] if positionals else None)


def compute_ranges(mergebase, files, untracked, originals):
	batched = changed_ranges(mergebase, files)
	ranges_by_file = {}
	for f in files:
		r = batched[f]
		if not r and (f in untracked or not base_has(mergebase, f)):
			# brand-new/untracked file has no base version: treat every line as edited
			r = [(1, max(1, len(originals[f].splitlines())))]
		ranges_by_file[f] = r
	return ranges_by_file


def main():
	mode, base_arg = parse_args(sys.argv[1:])
	os.chdir(git("rev-parse", "--show-toplevel").strip())
	base = resolve_base(base_arg)
	mergebase = merge_base(base)

	files, skipped, untracked = discover_files(mergebase)
	if skipped:
		print(f"Skipping style-exempt files ({EXCLUDE_FILE}):")
		for f in skipped:
			print(f"  {f}")
	if not files:
		print(f"No changed Java source files to check (base: {base}).")
		return 0

	originals = {f: read_text(f) for f in files}
	ranges_by_file = compute_ranges(mergebase, files, untracked, originals)

	reports = []
	wrote_fix = False
	try:
		try:
			run_formatter(files)
		except subprocess.CalledProcessError as e:
			sys.exit(f"ERROR: could not run the Eclipse formatter (is `mvn` on PATH?): {e}")

		results = {}
		for f in files:
			result, kept = line_scoped_result(originals[f], read_text(f), ranges_by_file[f])
			results[f] = result
			if kept:
				reports.append((f, kept))

		if mode == "fix":
			for f in files:
				write_text(f, results[f])
			wrote_fix = True
	finally:
		# never leave mvn's whole-file reformat on disk: check mode is read-only,
		# and fix mode must restore originals unless it fully wrote the scoped results
		if mode == "check" or (mode == "fix" and not wrote_fix):
			for f in files:
				write_text(f, originals[f])

	if reports and mode == "check":
		for path, kept in reports:
			print_report(path, kept)
		print("\nERROR: the changes above are required on lines this PR edited "
			"(per dev/CodeStyle_eclipse.xml).")
		print("Fix locally with:  dev/format-changed.sh")
		return 1
	if reports and mode == "fix":
		print("Applied changed-line formatting. Review and commit the result.")
	else:
		print("All PR-edited Java lines are correctly formatted.")
	return 0


if __name__ == "__main__":
	sys.exit(main())

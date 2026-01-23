    i New version 1.19.0 available (installed version is 1.18.3) at https://github.com/docker/scout-cli
    ✓ Image stored for indexing
    ✓ Indexed 346 packages
    ✓ Provenance obtained from attestation
    ✗ Detected 11 vulnerable packages with a total of 30 vulnerabilities
    ✓ Report written to docker/scout_results/medium-cves/sysds_output15-1.md
<h2>:mag: Vulnerabilities of <code>apache/systemds:latest</code></h2>

<details open="true"><summary>:package: Image Reference</strong> <code>apache/systemds:latest</code></summary>
<table>
<tr><td>digest</td><td><code>sha256:98fd77b0c3749d5d26fff317eaf6c1aa18e9fc98aa504b3178fd8a7215bbfd99</code></td><tr><tr><td>vulnerabilities</td><td><img alt="critical: 1" src="https://img.shields.io/badge/critical-1-8b1924"/> <img alt="high: 9" src="https://img.shields.io/badge/high-9-e25d68"/> <img alt="medium: 16" src="https://img.shields.io/badge/medium-16-fbb552"/> <img alt="low: 3" src="https://img.shields.io/badge/low-3-fce1a9"/> <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/></td></tr>
<tr><td>platform</td><td>linux/amd64</td></tr>
<tr><td>size</td><td>378 MB</td></tr>
<tr><td>packages</td><td>346</td></tr>
</table>
</details></table>
</details>

<table>
<tr><td valign="top">
<details><summary><img alt="critical: 1" src="https://img.shields.io/badge/C-1-8b1924"/> <img alt="high: 2" src="https://img.shields.io/badge/H-2-e25d68"/> <img alt="medium: 5" src="https://img.shields.io/badge/M-5-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>io.netty/netty</strong> <code>3.10.6.Final</code> (maven)</summary>

<small><code>pkg:maven/io.netty/netty@3.10.6.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2019-20444?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="critical 9.1: CVE--2019--20444" src="https://img.shields.io/badge/CVE--2019--20444-lightgrey?label=critical%209.1&labelColor=8b1924"/></a> <i>Inconsistent Interpretation of HTTP Requests ('HTTP Request/Response Smuggling')</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>CVSS Score</td><td><code>9.1</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>11.096%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>93rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

HttpObjectDecoder.java in Netty before 4.1.44 allows an HTTP header that lacks a colon, which might be interpreted as a separate header with an incorrect syntax, or might be interpreted as an "invalid fold."

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-37137?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="high 7.5: CVE--2021--37137" src="https://img.shields.io/badge/CVE--2021--37137-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>2.383%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>85th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
The Snappy frame decoder function doesn't restrict the chunk length which may lead to excessive memory usage. Beside this it also may buffer reserved skippable chunks until the whole chunk was received which may lead to excessive memory usage as well.

This vulnerability can be triggered by supplying malicious input that decompresses to a very big size (via a network stream or a file) or by sending a huge skippable chunk.

### Impact

All users of SnappyFrameDecoder are affected and so the application may be in risk for a DoS attach due excessive memory usage.

### References
https://github.com/netty/netty/blob/netty-4.1.67.Final/codec/src/main/java/io/netty/handler/codec/compression/SnappyFrameDecoder.java#L79
https://github.com/netty/netty/blob/netty-4.1.67.Final/codec/src/main/java/io/netty/handler/codec/compression/SnappyFrameDecoder.java#L171
https://github.com/netty/netty/blob/netty-4.1.67.Final/codec/src/main/java/io/netty/handler/codec/compression/SnappyFrameDecoder.java#L185

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-37136?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="high 7.5: CVE--2021--37136" src="https://img.shields.io/badge/CVE--2021--37136-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>1.021%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>77th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
The Bzip2 decompression decoder function doesn't allow setting size restrictions on the decompressed output data (which affects the allocation size used during decompression).


All users of Bzip2Decoder are affected. The malicious input can trigger an OOME and so a DoS attack

### Workarounds
No workarounds other than not using the `Bzip2Decoder`

### References

Relevant code areas:

https://github.com/netty/netty/blob/netty-4.1.67.Final/codec/src/main/java/io/netty/handler/codec/compression/Bzip2Decoder.java#L80
https://github.com/netty/netty/blob/netty-4.1.67.Final/codec/src/main/java/io/netty/handler/codec/compression/Bzip2Decoder.java#L294
https://github.com/netty/netty/blob/netty-4.1.67.Final/codec/src/main/java/io/netty/handler/codec/compression/Bzip2Decoder.java#L305

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-43797?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="medium 6.5: CVE--2021--43797" src="https://img.shields.io/badge/CVE--2021--43797-lightgrey?label=medium%206.5&labelColor=fbb552"/></a> <i>Inconsistent Interpretation of HTTP Requests ('HTTP Request/Response Smuggling')</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>CVSS Score</td><td><code>6.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.432%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>62nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact

Netty currently just skips control chars when these are present at the beginning / end of the header name. We should better fail fast as these are not allowed by the spec and could lead to HTTP request smuggling.

Failing to do the validation might cause netty to "sanitize" header names before it forward these to another remote system when used as proxy. This remote system can't see the invalid usage anymore and so not do the validation itself.



</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-21290?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="medium 6.2: CVE--2021--21290" src="https://img.shields.io/badge/CVE--2021--21290-lightgrey?label=medium%206.2&labelColor=fbb552"/></a> <i>Creation of Temporary File With Insecure Permissions</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>CVSS Score</td><td><code>6.2</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.025%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>6th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact

When netty's multipart decoders are used local information disclosure can occur via the local system temporary directory if temporary storing uploads on the disk is enabled.

The CVSSv3.1 score of this vulnerability is calculated to be a [6.2/10](https://nvd.nist.gov/vuln-metrics/cvss/v3-calculator?vector=AV:L/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N&version=3.1)

### Vulnerability Details

On unix-like systems, the temporary directory is shared between all user. As such, writing to this directory using APIs that do not explicitly set the file/directory permissions can lead to information disclosure. Of note, this does not impact modern MacOS Operating Systems.

The method `File.createTempFile` on unix-like systems creates a random file, but, by default will create this file with the permissions `-rw-r--r--`. Thus, if sensitive information is written to this file, other local users can read this information.

This is the case in netty's `AbstractDiskHttpData` is vulnerable.

https://github.com/netty/netty/blob/e5951d46fc89db507ba7d2968d2ede26378f0b04/codec-http/src/main/java/io/netty/handler/codec/http/multipart/AbstractDiskHttpData.java#L80-L101

`AbstractDiskHttpData` is used as a part of the `DefaultHttpDataFactory` class which is used by `HttpPostRequestDecoder` / `HttpPostMultiPartRequestDecoder`.

You may be affected by this vulnerability your project contains the following code patterns:

```java
channelPipeline.addLast(new HttpPostRequestDecoder(...));
```

```java
channelPipeline.addLast(new HttpPostMultiPartRequestDecoder(...));
```

### Patches

This has been patched in version `4.1.59.Final`.

### Workarounds

Specify your own `java.io.tmpdir` when you start the JVM or use `DefaultHttpDataFactory.setBaseDir(...)` to set the directory to something that is only readable by the current user.

### References

 - [CWE-378: Creation of Temporary File With Insecure Permissions](https://cwe.mitre.org/data/definitions/378.html)
 - [CWE-379: Creation of Temporary File in Directory with Insecure Permissions](https://cwe.mitre.org/data/definitions/379.html)

### Similar Vulnerabilities

Similar, but not the same.

 - JUnit 4 - https://github.com/junit-team/junit4/security/advisories/GHSA-269g-pwp5-87pp
 - Google Guava - https://github.com/google/guava/issues/4011
 - Apache Ant - https://nvd.nist.gov/vuln/detail/CVE-2020-1945
 - JetBrains Kotlin Compiler - https://nvd.nist.gov/vuln/detail/CVE-2020-15824

### For more information
If you have any questions or comments about this advisory:
* Open an issue in [netty](https://github.com/netty/netty)
* Email us [here](mailto:netty-security@googlegroups.com)

### Original Report

> Hi Netty Security Team,
> 
> I've been working on some security research leveraging custom CodeQL queries to detect local information disclosure vulnerabilities in java applications. This was the result from running this query against the netty project:
> https://lgtm.com/query/7723301787255288599/
> 
> Netty contains three local information disclosure vulnerabilities, so far as I can tell.
> 
> One is here, where the private key for the certificate is written to a temporary file.
> 
> https://github.com/netty/netty/blob/e5951d46fc89db507ba7d2968d2ede26378f0b04/handler/src/main/java/io/netty/handler/ssl/util/SelfSignedCertificate.java#L316-L346
> 
> One is here, where the certificate is written to a temporary file.
> 
> https://github.com/netty/netty/blob/e5951d46fc89db507ba7d2968d2ede26378f0b04/handler/src/main/java/io/netty/handler/ssl/util/SelfSignedCertificate.java#L348-L371
> 
> The final one is here, where the 'AbstractDiskHttpData' creates a temporary file if the getBaseDirectory() method returns null. I believe that 'AbstractDiskHttpData' is used as a part of the file upload support? If this is the case, any files uploaded would be similarly vulnerable.
> 
> https://github.com/netty/netty/blob/e5951d46fc89db507ba7d2968d2ede26378f0b04/codec-http/src/main/java/io/netty/handler/codec/http/multipart/AbstractDiskHttpData.java#L91
> 
> All of these vulnerabilities exist because `File.createTempFile(String, String)` will create a temporary file in the system temporary directory if the 'java.io.tmpdir' system property is not explicitly set. It is my understanding that when java creates a file, by default, and using this method, the permissions on that file utilize the umask. In a majority of cases, this means that the file that java creates has the permissions: `-rw-r--r--`, thus, any other local user on that system can read the contents of that file.
> 
> Impacted OS:
> - Any OS where the system temporary directory is shared between multiple users. This is not the case for MacOS or Windows.
> 
> Mitigation.
> 
> Moving to the `Files` API instead will fix this vulnerability. 
> https://docs.oracle.com/javase/8/docs/api/java/nio/file/Files.html#createTempFile-java.nio.file.Path-java.lang.String-java.lang.String-java.nio.file.attribute.FileAttribute...-
> 
> This API will explicitly set the posix file permissions to something safe, by default.
> 
> I recently disclosed a similar vulnerability in JUnit 4:
> https://github.com/junit-team/junit4/security/advisories/GHSA-269g-pwp5-87pp
> 
> If you're also curious, this vulnerability in Jetty was also mine, also involving temporary directories, but is not the same vulnerability as in this case.
> https://github.com/eclipse/jetty.project/security/advisories/GHSA-g3wg-6mcf-8jj6
> 
> I would appreciate it if we could perform disclosure of this vulnerability leveraging the GitHub security advisories feature here. GitHub has a nice credit system that I appreciate, plus the disclosures, as you can see from the sampling above, end up looking very nice.
> https://github.com/netty/netty/security/advisories
> 
> This vulnerability disclosure follows Google's [90-day vulnerability disclosure policy](https://www.google.com/about/appsecurity/) (I'm not an employee of Google, I just like their policy). Full disclosure will occur either at the end of the 90-day deadline or whenever a patch is made widely available, whichever occurs first.
> 
> Cheers,
> Jonathan Leitschuh

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-21409?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="medium 5.9: CVE--2021--21409" src="https://img.shields.io/badge/CVE--2021--21409-lightgrey?label=medium%205.9&labelColor=fbb552"/></a> <i>Inconsistent Interpretation of HTTP Requests ('HTTP Request/Response Smuggling')</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>CVSS Score</td><td><code>5.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:H/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>3.244%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>87th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
The content-length header is not correctly validated if the request only use a single Http2HeaderFrame with the endStream set to to true. This could lead to request smuggling if the request is proxied to a remote peer and translated to HTTP/1.1

This is a followup of https://github.com/netty/netty/security/advisories/GHSA-wm47-8v5p-wjpj which did miss to fix this one case. 

### Patches
This was fixed as part of 4.1.61.Final

### Workarounds
Validation can be done by the user before proxy the request by validating the header.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-21295?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="medium 5.9: CVE--2021--21295" src="https://img.shields.io/badge/CVE--2021--21295-lightgrey?label=medium%205.9&labelColor=fbb552"/></a> <i>Inconsistent Interpretation of HTTP Requests ('HTTP Request/Response Smuggling')</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>CVSS Score</td><td><code>5.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:H/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.360%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>58th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
If a Content-Length header is present in the original HTTP/2 request, the field is not validated by `Http2MultiplexHandler` as it is propagated up.  This is fine as long as the request is not proxied through as HTTP/1.1.
If the request comes in as an HTTP/2 stream, gets converted into the HTTP/1.1 domain objects (`HttpRequest`, `HttpContent`, etc.) via `Http2StreamFrameToHttpObjectCodec `and then sent up to the child channel's pipeline and proxied through a remote peer as HTTP/1.1 this may result in request smuggling.  

In a proxy case, users  may assume the content-length is validated somehow, which is not the case.  If the request is forwarded to a backend channel that is a HTTP/1.1 connection, the Content-Length now has meaning and needs to be checked.

An attacker can smuggle requests inside the body as it gets downgraded from HTTP/2 to HTTP/1.1.   A sample attack request looks like:

```
POST / HTTP/2
:authority:: externaldomain.com
Content-Length: 4

asdfGET /evilRedirect HTTP/1.1
Host: internaldomain.com
```

Users are only affected if all of this is `true`:
 * `HTTP2MultiplexCodec` or `Http2FrameCodec` is used
 * `Http2StreamFrameToHttpObjectCodec` is used to convert to HTTP/1.1 objects
 * These  HTTP/1.1 objects are forwarded to another remote peer.
 

### Patches
This has been patched in 4.1.60.Final

### Workarounds
The user can do the validation by themselves by implementing a custom `ChannelInboundHandler` that is put in the `ChannelPipeline` behind `Http2StreamFrameToHttpObjectCodec`.

### References
Related change to workaround the problem: https://github.com/Netflix/zuul/pull/980 

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2019-20445?s=github&n=netty&ns=io.netty&t=maven&vr=%3C4.0.0"><img alt="medium : CVE--2019--20445" src="https://img.shields.io/badge/CVE--2019--20445-lightgrey?label=medium%20&labelColor=fbb552"/></a> <i>Inconsistent Interpretation of HTTP Requests ('HTTP Request/Response Smuggling')</i>

<table>
<tr><td>Affected range</td><td><code><4.0.0</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>EPSS Score</td><td><code>2.837%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>86th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

HttpObjectDecoder.java in Netty before 4.1.44 allows a Content-Length header to be accompanied by a second Content-Length header, or by a Transfer-Encoding header.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 4" src="https://img.shields.io/badge/H-4-e25d68"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>com.google.protobuf/protobuf-java</strong> <code>3.7.1</code> (maven)</summary>

<small><code>pkg:maven/com.google.protobuf/protobuf-java@3.7.1</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-7254?s=github&n=protobuf-java&ns=com.google.protobuf&t=maven&vr=%3C3.25.5"><img alt="high 8.7: CVE--2024--7254" src="https://img.shields.io/badge/CVE--2024--7254-lightgrey?label=high%208.7&labelColor=e25d68"/></a> <i>Improper Input Validation</i>

<table>
<tr><td>Affected range</td><td><code><3.25.5</code></td></tr>
<tr><td>Fixed version</td><td><code>3.25.5</code></td></tr>
<tr><td>CVSS Score</td><td><code>8.7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.085%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>25th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary
When parsing unknown fields in the Protobuf Java Lite and Full library, a maliciously crafted message can cause a StackOverflow error and lead to a program crash.

Reporter: Alexis Challande, Trail of Bits Ecosystem Security Team <ecosystem@trailofbits.com>

Affected versions: This issue affects all versions of both the Java full and lite Protobuf runtimes, as well as Protobuf for Kotlin and JRuby, which themselves use the Java Protobuf runtime.

### Severity
[CVE-2024-7254](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2024-7254) **High** CVSS4.0 Score 8.7 (NOTE: there may be a delay in publication)
This is a potential Denial of Service. Parsing nested groups as unknown fields with DiscardUnknownFieldsParser or Java Protobuf Lite parser, or against Protobuf map fields, creates unbounded recursions that can be abused by an attacker.

### Proof of Concept
For reproduction details, please refer to the unit tests (Protobuf Java [LiteTest](https://github.com/protocolbuffers/protobuf/blob/a037f28ff81ee45ebe008c64ab632bf5372242ce/java/lite/src/test/java/com/google/protobuf/LiteTest.java) and [CodedInputStreamTest](https://github.com/protocolbuffers/protobuf/blob/a037f28ff81ee45ebe008c64ab632bf5372242ce/java/core/src/test/java/com/google/protobuf/CodedInputStreamTest.java)) that identify the specific inputs that exercise this parsing weakness.

### Remediation and Mitigation
We have been working diligently to address this issue and have released a mitigation that is available now. Please update to the latest available versions of the following packages:
* protobuf-java (3.25.5, 4.27.5, 4.28.2)
* protobuf-javalite (3.25.5, 4.27.5, 4.28.2)
* protobuf-kotlin (3.25.5, 4.27.5, 4.28.2)
* protobuf-kotlin-lite (3.25.5, 4.27.5, 4.28.2)
* com-protobuf [JRuby gem only] (3.25.5, 4.27.5, 4.28.2)

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2022-3510?s=github&n=protobuf-java&ns=com.google.protobuf&t=maven&vr=%3E%3D3.0.0%2C%3C3.16.3"><img alt="high 7.5: CVE--2022--3510" src="https://img.shields.io/badge/CVE--2022--3510-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code>>=3.0.0<br/><3.16.3</code></td></tr>
<tr><td>Fixed version</td><td><code>3.16.3</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.073%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>22nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A parsing issue similar to CVE-2022-3171, but with Message-Type Extensions in protobuf-java core and lite versions prior to 3.21.7, 3.20.3, 3.19.6 and 3.16.3 can lead to a denial of service attack. Inputs containing multiple instances of non-repeated embedded messages with repeated or unknown fields causes objects to be converted back-n-forth between mutable and immutable forms, resulting in potentially long garbage collection pauses. We recommend updating to the versions mentioned above.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2022-3509?s=github&n=protobuf-java&ns=com.google.protobuf&t=maven&vr=%3E%3D3.0.0%2C%3C3.16.3"><img alt="high 7.5: CVE--2022--3509" src="https://img.shields.io/badge/CVE--2022--3509-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code>>=3.0.0<br/><3.16.3</code></td></tr>
<tr><td>Fixed version</td><td><code>3.16.3</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.131%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>33rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A parsing issue similar to CVE-2022-3171, but with textformat in protobuf-java core and lite versions prior to 3.21.7, 3.20.3, 3.19.6 and 3.16.3 can lead to a denial of service attack. Inputs containing multiple instances of non-repeated embedded messages with repeated or unknown fields causes objects to be converted back-n-forth between mutable and immutable forms, resulting in potentially long garbage collection pauses. We recommend updating to the versions mentioned above.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-22569?s=github&n=protobuf-java&ns=com.google.protobuf&t=maven&vr=%3C3.16.1"><img alt="high 7.5: CVE--2021--22569" src="https://img.shields.io/badge/CVE--2021--22569-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Incorrect Behavior Order</i>

<table>
<tr><td>Affected range</td><td><code><3.16.1</code></td></tr>
<tr><td>Fixed version</td><td><code>3.16.1</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.291%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>52nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

## Summary

A potential Denial of Service issue in protobuf-java was discovered in the parsing procedure for binary data.

Reporter: [OSS-Fuzz](https://github.com/google/oss-fuzz)

Affected versions: All versions of Java Protobufs (including Kotlin and JRuby) prior to the versions listed below. Protobuf "javalite" users (typically Android) are not affected.

## Severity

[CVE-2021-22569](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-22569) **High** - CVSS Score: 7.5,  An implementation weakness in how unknown fields are parsed in Java. A small (~800 KB) malicious payload can occupy the parser for several minutes by creating large numbers of short-lived objects that cause frequent, repeated GC pauses.

## Proof of Concept

For reproduction details, please refer to the oss-fuzz issue that identifies the specific inputs that exercise this parsing weakness.

## Remediation and Mitigation

Please update to the latest available versions of the following packages:

- protobuf-java (3.16.1, 3.18.2, 3.19.2) 
- protobuf-kotlin (3.18.2, 3.19.2)
- google-protobuf [JRuby  gem only] (3.19.2) 


</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2022-3171?s=github&n=protobuf-java&ns=com.google.protobuf&t=maven&vr=%3C3.16.3"><img alt="medium 5.7: CVE--2022--3171" src="https://img.shields.io/badge/CVE--2022--3171-lightgrey?label=medium%205.7&labelColor=fbb552"/></a> <i>Improper Input Validation</i>

<table>
<tr><td>Affected range</td><td><code><3.16.3</code></td></tr>
<tr><td>Fixed version</td><td><code>3.16.3</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:A/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.083%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>25th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

## Summary
A potential Denial of Service issue in `protobuf-java` core and lite was discovered in the parsing procedure for binary and text format data. Input streams containing multiple instances of non-repeated [embedded messages](http://developers.google.com/protocol-buffers/docs/encoding#embedded) with repeated or unknown fields causes objects to be converted back-n-forth between mutable and immutable forms, resulting in potentially long garbage collection pauses. 

Reporter: [OSS Fuzz](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=48771)

Affected versions: This issue affects both the Java full and lite Protobuf runtimes, as well as Protobuf for Kotlin and JRuby, which themselves use the Java Protobuf runtime.

## Severity

[CVE-2022-3171](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2022-3171) Medium - CVSS Score: 5.7 (NOTE: there may be a delay in publication)

## Remediation and Mitigation

Please update to the latest available versions of the following packages:

protobuf-java (3.21.7, 3.20.3, 3.19.6, 3.16.3)
protobuf-javalite (3.21.7, 3.20.3, 3.19.6, 3.16.3)
protobuf-kotlin (3.21.7, 3.20.3, 3.19.6, 3.16.3)
protobuf-kotlin-lite (3.21.7, 3.20.3, 3.19.6, 3.16.3)
google-protobuf [JRuby gem only] (3.21.7, 3.20.3, 3.19.6)


</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-22570?s=gitlab&n=protobuf-java&ns=com.google.protobuf&t=maven&vr=%3C3.15.0"><img alt="medium 5.5: CVE--2021--22570" src="https://img.shields.io/badge/CVE--2021--22570-lightgrey?label=medium%205.5&labelColor=fbb552"/></a> <i>OWASP Top Ten 2017 Category A9 - Using Components with Known Vulnerabilities</i>

<table>
<tr><td>Affected range</td><td><code><3.15.0</code></td></tr>
<tr><td>Fixed version</td><td><code>3.15.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.150%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>36th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Nullptr dereference when a null char is present in a proto symbol. The symbol is parsed incorrectly, leading to an unchecked call into the proto file's name during generation of the resulting error message. Since the symbol is incorrectly parsed, the file is nullptr.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>openssl</strong> <code>3.3.3-r0</code> (apk)</summary>

<small><code>pkg:apk/alpine/openssl@3.3.3-r0?os_name=alpine&os_version=3.20</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-9230?s=alpine&n=openssl&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C3.3.5-r0"><img alt="high : CVE--2025--9230" src="https://img.shields.io/badge/CVE--2025--9230-lightgrey?label=high%20&labelColor=e25d68"/></a> 

<table>
<tr><td>Affected range</td><td><code><3.3.5-r0</code></td></tr>
<tr><td>Fixed version</td><td><code>3.3.5-r0</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.029%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>8th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2025-9231?s=alpine&n=openssl&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C3.3.5-r0"><img alt="medium : CVE--2025--9231" src="https://img.shields.io/badge/CVE--2025--9231-lightgrey?label=medium%20&labelColor=fbb552"/></a> 

<table>
<tr><td>Affected range</td><td><code><3.3.5-r0</code></td></tr>
<tr><td>Fixed version</td><td><code>3.3.5-r0</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.016%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>3rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2025-9232?s=alpine&n=openssl&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C3.3.5-r0"><img alt="medium : CVE--2025--9232" src="https://img.shields.io/badge/CVE--2025--9232-lightgrey?label=medium%20&labelColor=fbb552"/></a> 

<table>
<tr><td>Affected range</td><td><code><3.3.5-r0</code></td></tr>
<tr><td>Fixed version</td><td><code>3.3.5-r0</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.028%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>7th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.eclipse.jetty/jetty-server</strong> <code>9.4.52.v20230823</code> (maven)</summary>

<small><code>pkg:maven/org.eclipse.jetty/jetty-server@9.4.52.v20230823</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-13009?s=github&n=jetty-server&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.4.0%2C%3C%3D9.4.56"><img alt="high 7.2: CVE--2024--13009" src="https://img.shields.io/badge/CVE--2024--13009-lightgrey?label=high%207.2&labelColor=e25d68"/></a> <i>Improper Resource Shutdown or Release</i>

<table>
<tr><td>Affected range</td><td><code>>=9.4.0<br/><=9.4.56</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.57.v20241219</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.2</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:L/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.130%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>33rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

In Eclipse Jetty versions 9.4.0 to 9.4.56 a buffer can be incorrectly released when confronted with a gzip error when inflating a request body. This can result in corrupted and/or inadvertent sharing of data between requests.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-8184?s=github&n=jetty-server&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.3.12%2C%3C%3D9.4.55"><img alt="medium 5.9: CVE--2024--8184" src="https://img.shields.io/badge/CVE--2024--8184-lightgrey?label=medium%205.9&labelColor=fbb552"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code>>=9.3.12<br/><=9.4.55</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.56</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>1.487%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>81st percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
Remote DOS attack can cause out of memory 

### Description
There exists a security vulnerability in Jetty's `ThreadLimitHandler.getRemote()` which
can be exploited by unauthorized users to cause remote denial-of-service (DoS) attack.  By
repeatedly sending crafted requests, attackers can trigger OutofMemory errors and exhaust the
server's memory.

### Affected Versions

* Jetty 12.0.0-12.0.8 (Supported)
* Jetty 11.0.0-11.0.23 (EOL)
* Jetty 10.0.0-10.0.23 (EOL)
* Jetty 9.3.12-9.4.55 (EOL)

### Patched Versions

* Jetty 12.0.9
* Jetty 11.0.24
* Jetty 10.0.24
* Jetty 9.4.56

### Workarounds

Do not use `ThreadLimitHandler`.  
Consider use of `QoSHandler` instead to artificially limit resource utilization.

### References

Jetty 12 - https://github.com/jetty/jetty.project/pull/11723

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>com.fasterxml.jackson.core/jackson-core</strong> <code>2.13.4</code> (maven)</summary>

<small><code>pkg:maven/com.fasterxml.jackson.core/jackson-core@2.13.4</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-52999?s=github&n=jackson-core&ns=com.fasterxml.jackson.core&t=maven&vr=%3C2.15.0"><img alt="high 8.7: CVE--2025--52999" src="https://img.shields.io/badge/CVE--2025--52999-lightgrey?label=high%208.7&labelColor=e25d68"/></a> <i>Stack-based Buffer Overflow</i>

<table>
<tr><td>Affected range</td><td><code><2.15.0</code></td></tr>
<tr><td>Fixed version</td><td><code>2.15.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>8.7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.033%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>9th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
With older versions  of jackson-core, if you parse an input file and it has deeply nested data, Jackson could end up throwing a StackoverflowError if the depth is particularly large.

### Patches
jackson-core 2.15.0 contains a configurable limit for how deep Jackson will traverse in an input document, defaulting to an allowable depth of 1000. Change is in https://github.com/FasterXML/jackson-core/pull/943. jackson-core will throw a StreamConstraintsException if the limit is reached.
jackson-databind also benefits from this change because it uses jackson-core to parse JSON inputs.

### Workarounds
Users should avoid parsing input files from untrusted sources.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>com.google.guava/guava</strong> <code>14.0.1</code> (maven)</summary>

<small><code>pkg:maven/com.google.guava/guava@14.0.1</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2018-10237?s=github&n=guava&ns=com.google.guava&t=maven&vr=%3E%3D11.0%2C%3C24.1.1-android"><img alt="medium 5.9: CVE--2018--10237" src="https://img.shields.io/badge/CVE--2018--10237-lightgrey?label=medium%205.9&labelColor=fbb552"/></a> <i>Deserialization of Untrusted Data</i>

<table>
<tr><td>Affected range</td><td><code>>=11.0<br/><24.1.1-android</code></td></tr>
<tr><td>Fixed version</td><td><code>24.1.1-android</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>3.259%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>87th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Unbounded memory allocation in Google Guava 11.0 through 24.x before 24.1.1 allows remote attackers to conduct denial of service attacks against servers that depend on this library and deserialize attacker-provided data, because the AtomicDoubleArray class (when serialized with Java serialization) and the CompoundOrdering class (when serialized with GWT serialization) perform eager allocation without appropriate checks on what a client has sent and whether the data size is reasonable.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2023-2976?s=github&n=guava&ns=com.google.guava&t=maven&vr=%3E%3D1.0%2C%3C32.0.0-android"><img alt="medium 5.5: CVE--2023--2976" src="https://img.shields.io/badge/CVE--2023--2976-lightgrey?label=medium%205.5&labelColor=fbb552"/></a> <i>Creation of Temporary File in Directory with Insecure Permissions</i>

<table>
<tr><td>Affected range</td><td><code>>=1.0<br/><32.0.0-android</code></td></tr>
<tr><td>Fixed version</td><td><code>32.0.0-android</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.079%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>24th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Use of Java's default temporary directory for file creation in `FileBackedOutputStream` in Google Guava versions 1.0 to 31.1 on Unix systems and Android Ice Cream Sandwich allows other users and apps on the machine with access to the default Java temporary directory to be able to access the files created by the class.

Even though the security vulnerability is fixed in version 32.0.0, maintainers recommend using version 32.0.1 as version 32.0.0 breaks some functionality under Windows.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2020-8908?s=github&n=guava&ns=com.google.guava&t=maven&vr=%3C32.0.0-android"><img alt="low 3.3: CVE--2020--8908" src="https://img.shields.io/badge/CVE--2020--8908-lightgrey?label=low%203.3&labelColor=fce1a9"/></a> <i>Improper Handling of Alternate Encoding</i>

<table>
<tr><td>Affected range</td><td><code><32.0.0-android</code></td></tr>
<tr><td>Fixed version</td><td><code>32.0.0-android</code></td></tr>
<tr><td>CVSS Score</td><td><code>3.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.072%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>22nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A temp directory creation vulnerability exists in Guava prior to version 32.0.0 allowing an attacker with access to the machine to potentially access data in a temporary directory created by the Guava `com.google.common.io.Files.createTempDir()`. The permissions granted to the directory created default to the standard unix-like /tmp ones, leaving the files open. Maintainers recommend explicitly changing the permissions after the creation of the directory, or removing uses of the vulnerable method.


</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 2" src="https://img.shields.io/badge/L-2-fce1a9"/> <!-- unspecified: 0 --><strong>busybox</strong> <code>1.36.1-r29</code> (apk)</summary>

<small><code>pkg:apk/alpine/busybox@1.36.1-r29?os_name=alpine&os_version=3.20</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-60876?s=alpine&n=busybox&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C%3D1.36.1-r29"><img alt="medium : CVE--2025--60876" src="https://img.shields.io/badge/CVE--2025--60876-lightgrey?label=medium%20&labelColor=fbb552"/></a> 

<table>
<tr><td>Affected range</td><td><code><=1.36.1-r29</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>EPSS Score</td><td><code>0.052%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>16th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2025-46394?s=alpine&n=busybox&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C1.36.1-r31"><img alt="low : CVE--2025--46394" src="https://img.shields.io/badge/CVE--2025--46394-lightgrey?label=low%20&labelColor=fce1a9"/></a> 

<table>
<tr><td>Affected range</td><td><code><1.36.1-r31</code></td></tr>
<tr><td>Fixed version</td><td><code>1.36.1-r31</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.031%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>8th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-58251?s=alpine&n=busybox&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C1.36.1-r31"><img alt="low : CVE--2024--58251" src="https://img.shields.io/badge/CVE--2024--58251-lightgrey?label=low%20&labelColor=fce1a9"/></a> 

<table>
<tr><td>Affected range</td><td><code><1.36.1-r31</code></td></tr>
<tr><td>Fixed version</td><td><code>1.36.1-r31</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.024%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>6th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.eclipse.jetty/jetty-servlets</strong> <code>9.4.52.v20230823</code> (maven)</summary>

<small><code>pkg:maven/org.eclipse.jetty/jetty-servlets@9.4.52.v20230823</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-9823?s=github&n=jetty-servlets&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.0.0%2C%3C9.4.54"><img alt="medium 5.3: CVE--2024--9823" src="https://img.shields.io/badge/CVE--2024--9823-lightgrey?label=medium%205.3&labelColor=fbb552"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code>>=9.0.0<br/><9.4.54</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.54</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.591%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>69th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Description
There exists a security vulnerability in Jetty's DosFilter which can be exploited by unauthorized users to cause remote denial-of-service (DoS) attack on the server using DosFilter. By repeatedly sending crafted requests, attackers can trigger OutofMemory errors and exhaust the server's memory finally.


Vulnerability details
The Jetty DoSFilter (Denial of Service Filter) is a security filter designed to protect web applications against certain types of Denial of Service (DoS) attacks and other abusive behavior. It helps to mitigate excessive resource consumption by limiting the rate at which clients can make requests to the server.  The DoSFilter monitors and tracks client request patterns, including request rates, and can take actions such as blocking or delaying requests from clients that exceed predefined thresholds.  The internal tracking of requests in DoSFilter is the source of this OutOfMemory condition.


Impact
Users of the DoSFilter may be subject to DoS attacks that will ultimately exhaust the memory of the server if they have not configured session passivation or an aggressive session inactivation timeout.


Patches
The DoSFilter has been patched in all active releases to no longer support the session tracking mode, even if configured.


Patched releases:

  *  9.4.54
  *  10.0.18
  *  11.0.18
  *  12.0.3

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.eclipse.jetty/jetty-http</strong> <code>9.4.52.v20230823</code> (maven)</summary>

<small><code>pkg:maven/org.eclipse.jetty/jetty-http@9.4.52.v20230823</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-6763?s=github&n=jetty-http&ns=org.eclipse.jetty&t=maven&vr=%3E%3D7.0.0%2C%3C%3D12.0.11"><img alt="medium 6.3: CVE--2024--6763" src="https://img.shields.io/badge/CVE--2024--6763-lightgrey?label=medium%206.3&labelColor=fbb552"/></a> <i>Improper Validation of Syntactic Correctness of Input</i>

<table>
<tr><td>Affected range</td><td><code>>=7.0.0<br/><=12.0.11</code></td></tr>
<tr><td>Fixed version</td><td><code>12.0.12</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:N/VC:N/VI:L/VA:N/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>1.022%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>77th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

## Summary

Eclipse Jetty is a lightweight, highly scalable, Java-based web server and Servlet engine . It includes a utility class, `HttpURI`, for URI/URL parsing.

The `HttpURI` class does insufficient validation on the authority segment of a URI.  However the behaviour of `HttpURI` differs from the common browsers in how it handles a URI that would be considered invalid if fully validated against the RRC.  Specifically `HttpURI` and the browser may differ on the value of the host extracted from an invalid URI and thus a combination of Jetty and a vulnerable browser may be vulnerable to a open redirect attack or to a SSRF attack if the URI is used after passing validation checks.

## Details

### Affected components

The vulnerable component is the `HttpURI` class when used as a utility class in an application.  The Jetty usage of the class is not vulnerable.

### Attack overview

The `HttpURI` class does not well validate the authority section of a URI. When presented with an illegal authority that may contain user info (eg username:password#@hostname:port), then the parsing of the URI is not failed.  Moreover, the interpretation of what part of the authority is the host name differs from a common browser in  that they also do not fail, but they select a different host name from the illegal URI.

### Attack scenario

A typical attack scenario is illustrated in the diagram below. The Validator checks whether the attacker-supplied URL is on the blocklist. If not, the URI is passed to the Requester for redirection. The Requester is responsible for sending requests to the hostname specified by the URI.

This attack occurs when the Validator is the `org.eclipse.jetty.http.HttpURI` class and the Requester is the `Browser` (include chrome, firefox and Safari). An attacker can send a malformed URI to the Validator (e.g., `http://browser.check%23%40vulndetector.com/` ). After validation, the Validator finds that the hostname is not on the blocklist. However, the Requester can still send requests to the domain with the hostname `vulndetector.com`.

## PoC

payloads:

```
http://browser.check &@vulndetector.com/
http://browser.check #@vulndetector.com/
http://browser.check?@vulndetector.com/
http://browser.check#@vulndetector.com/
http://vulndetector.com\\/
```

The problem of 302 redirect parsing in HTML tag scenarios. Below is a poc example. After clicking the button, the browser will open "browser.check", and jetty will parse this URL as "vulndetector.com".

```
<a href="http://browser.check#@vulndetector.com/"></a>
```
A comparison of the parsing differences between Jetty and chrome is shown in the table below (note that neither should accept the URI as valid).

| Invalid URI                                       | Jetty            | Chrome        |
| ---------------------------------------------- | ---------------- | ------------- |
| http://browser.check &@vulndetector.com/ | vulndetector.com | browser.check |
| http://browser.check #@vulndetector.com/ | vulndetector.com | browser.check |
| http://browser.check?@vulndetector.com/    | vulndetector.com | browser.check |
| http://browser.check#@vulndetector.com/    | vulndetector.com | browser.check |

The problem of 302 redirect parsing in HTTP 302 Location

| Input                    | Jetty          | Chrome        |
| ------------------------ | -------------- | ------------- |
| http://browser.check%5c/ | browser.check\ | browser.check |

It is noteworthy that Spring Web also faced similar security vulnerabilities, being affected by the aforementioned four types of payloads. These issues have since been resolved and have been assigned three CVE numbers [3-5].

## Impact

The impact of this vulnerability is limited to developers that use the Jetty HttpURI directly.  Example: your project implemented a blocklist to block on some hosts based on HttpURI's handling of authority section.  The vulnerability will help attackers bypass the protections that developers have set up for hosts. The vulnerability will lead to **SSRF**[1] and **URL Redirection**[2] vulnerabilities in several cases. 

## Mitigation

The attacks outlined above rely on decoded user data being passed to the `HttpURI` class. Application should not pass decoded user data as an encoded URI to any URI class/method, including `HttpURI`.  Such applications are likely to be vulnerable in other ways. 
The immediate solution is to upgrade to a version of the class that will fully validate the characters of the URI authority.  Ultimately, Jetty will deprecate and remove support for user info in the authority per [RFC9110 Section 4.2.4](https://datatracker.ietf.org/doc/html/rfc9110#section-4.2.4). 

Note that the Chrome (and other browsers) parse the invalid user info section improperly as well (due to flawed WhatWG URL parsing rules that do not apply outside of a Web Browser).

## Reference

[1] https://cwe.mitre.org/data/definitions/918.html
[2] https://cwe.mitre.org/data/definitions/601.html

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>zlib</strong> <code>1.3.1-r1</code> (apk)</summary>

<small><code>pkg:apk/alpine/zlib@1.3.1-r1?os_name=alpine&os_version=3.20</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2026-22184?s=alpine&n=zlib&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C%3D1.3.1-r1"><img alt="medium : CVE--2026--22184" src="https://img.shields.io/badge/CVE--2026--22184-lightgrey?label=medium%20&labelColor=fbb552"/></a> 

<table>
<tr><td>Affected range</td><td><code><=1.3.1-r1</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>EPSS Score</td><td><code>0.038%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>11th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <img alt="unspecified: 1" src="https://img.shields.io/badge/U-1-lightgrey"/><strong>lz4</strong> <code>1.9.4-r5</code> (apk)</summary>

<small><code>pkg:apk/alpine/lz4@1.9.4-r5?os_name=alpine&os_version=3.20</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-62813?s=alpine&n=lz4&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C%3D1.9.4-r5"><img alt="unspecified : CVE--2025--62813" src="https://img.shields.io/badge/CVE--2025--62813-lightgrey?label=unspecified%20&labelColor=lightgrey"/></a> 

<table>
<tr><td>Affected range</td><td><code><=1.9.4-r5</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>EPSS Score</td><td><code>0.018%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>3rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>
</details></td></tr>
</table>


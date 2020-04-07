<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

# Apache SystemDS Documentation

The primary SystemDS documentation is written in markdown (*.md) and can be converted to HTML using
[Jekyll](http://jekyllrb.com).

## Ubuntu Install

1. Install Ruby development environment [Jekyll on Ubuntu](https://jekyllrb.com/docs/installation/ubuntu/).
2. Install Jekyll inside the Ruby environnement.

   - `gem install jekyll jekyll-redirect-from bundler pygments.rb`
   - To do this you might need to change permissions on folders `/var/lib/gems` and `/var/lib/gems/2.5.0`.
   - The Pygments package is optional, and does require sudo permissions.

3. Install python dependencies (Optionally).

   - `pip install Pygments`

4. Launch the Documentation locally

   - `jekyll serve -w`
   - This is done from the root of the gh-pages branch of the system.
   - The serving will per default be on localhost port 4000 [Link](http://localhost:4000)

## Mac Install (Deprecated)

Jekyll (and optionally Pygments) can be installed on the Mac OS in the following manner.

```bash
brew install ruby
gem install jekyll
gem install jekyll-redirect-from
gem install bundler
brew install python
pip install Pygments
gem install pygments.rb
```

To generate SystemDS documentation in HTML, navigate to the ```docs``` folder, the root directory of the
documentation. From there, you can have Jekyll convert the markdown files to HTML. If you run in server mode,
Jekyll will serve up the generated documentation by default at http://127.0.0.1:4000. Modifications
to *.md files will be converted to HTML and can be viewed in a web browser.

```bash
jekyll serve -w
```

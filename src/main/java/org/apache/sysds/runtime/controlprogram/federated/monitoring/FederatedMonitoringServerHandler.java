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

package org.apache.sysds.runtime.controlprogram.federated.monitoring;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpObject;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.LastHttpContent;
import io.netty.util.CharsetUtil;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.controllers.BaseController;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.controllers.CoordinatorController;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FederatedMonitoringServerHandler extends SimpleChannelInboundHandler<HttpObject> {

    private final Map<String, BaseController> _allControllers = new HashMap<>();
    {
        _allControllers.put("/coordinators", new CoordinatorController());
    }

    private final static ThreadLocal<Request> _currentRequest = new ThreadLocal<>();

    @Override
    protected void channelRead0(ChannelHandlerContext ctx, HttpObject msg) {

        if (msg instanceof LastHttpContent) {
            final ByteBuf jsonBuf = ((LastHttpContent) msg).content();
            final Request request = _currentRequest.get();
            request.setBody(jsonBuf.toString(CharsetUtil.UTF_8));

            _currentRequest.remove();

            final FullHttpResponse response = processRequest(request);
            ctx.write(response);

        } else if (msg instanceof HttpRequest) {
            final HttpRequest httpRequest = (HttpRequest) msg;
            final Request request = new Request();
            request.setContext(httpRequest);

            _currentRequest.set(request);
        }

    }

    @Override
    public void channelReadComplete(ChannelHandlerContext ctx) {
        ctx.flush();
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();
        ctx.close();
    }

    private FullHttpResponse processRequest(final Request request) {
        try {
            final BaseController controller = parseController(request.getContext().uri());
            final String method = request.getContext().method().name();

            switch (method) {
                case "GET":
                    final Long id = parseId(request.getContext().uri());

                    if (id != null) {
                        return controller.get(request, id);
                    }

                    return controller.getAll(request);
                case "PUT":
                    return controller.create(request);
                case "POST":
                    return controller.update(request, parseId(request.getContext().uri()));
                case "DELETE":
                    return controller.delete(request, parseId(request.getContext().uri()));
                default:
                    throw new IllegalArgumentException("Method is not supported!");
            }
        } catch (RuntimeException ex) {
            ex.printStackTrace();
            return null;
        }
    }

    private BaseController parseController(final String currentPath) {
        final Optional<String> controller = _allControllers.keySet().stream()
                .filter(currentPath::startsWith)
                .findFirst();

        return controller.map(_allControllers::get).orElseThrow(() ->
                new IllegalArgumentException("Such controller does not exist!"));
    }

    private Long parseId(final String uri) {
        final Pattern pattern = Pattern.compile("^[/][a-z]+[/]");
        final Matcher matcher = pattern.matcher(uri);

        if (matcher.find()) {
            return Long.valueOf(uri.substring(matcher.end()));
        }
        return null;
    }
}

"""The stream, as Prime Agent sees it: an MCP integration the kernel imports.

Prime Agent does not hand MCP servers to the model as tools. Each server is a Python-backed
skill that the model imports and calls inside the persistent IPython kernel, so this file is
the whole client. ``McpIntegration`` connects over streamable HTTP, discovers the server's
tools and binds each one as an async method::

    import shogym_stream
    record = json.loads(await shogym_stream.pull())

Two class attributes are load-bearing:

``url``       where ``serve.py`` is listening. The host overrides it from the ``mcpServers``
              entry in ``.prime/agent/settings.json`` when there is one, and falls back to
              this value when there is not, so the two must agree.
``bearer_token_env``
              the reason ``SHOGYM_MCP_TOKEN`` exists. ``McpIntegration._open_session`` resolves
              a token before every connection and raises ``NotEnabled`` without one; there is
              no unauthenticated path through it. ``serve.py`` authenticates nobody, so the
              value is a formality, but it must be set and non-empty, in the shell that
              launches ``prime-agent``, or the kernel cannot connect at all.
"""

from contextlib import AsyncExitStack

from rlm import McpIntegration

# Prime Agent's own default would make this quickstart unusable, so the client is built here
# instead. ``McpIntegration._open_session`` hands the MCP SDK a bare
# ``httpx.AsyncClient(headers=...)`` with no ``timeout``, which silently inherits httpx's 5s
# *inactivity* defaults in place of the SDK's own 30s general / 300s SSE-read ones. Any tool that
# goes five seconds without emitting a response byte then fails, and the call that ends a task
# does exactly that: it seals and grades server-side before it answers. The surfaced error
# names none of this: the real ``httpx.ReadTimeout`` is swallowed into a debug log, and what
# reaches the caller is ``SSE stream ended without a response`` inside an ExceptionGroup.
#
# Upstream: https://github.com/PrimeIntellect-ai/prime-agent/issues/784
# Delete ``_open_session`` below once that lands; the base class will then do the right thing.
_SDK_TIMEOUT, _SDK_SSE_READ_TIMEOUT = 30.0, 300.0


def _timed_http_client(headers):
    """An HTTP client carrying the MCP SDK's documented timeouts, not httpx's defaults.

    Prefers the SDK's own factory, so each SDK version gets its matching client implementation
    (mcp 2.x builds on httpx2, not httpx); falls back to constructing one directly.
    """
    try:
        from mcp.shared._httpx_utils import create_mcp_http_client
    except ImportError:
        pass
    else:
        return create_mcp_http_client(headers=headers)

    import httpx

    return httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(_SDK_TIMEOUT, read=_SDK_SSE_READ_TIMEOUT),
    )


class ShogymStream(McpIntegration):
    server = "shogym"  # the `mcpServers` key, and the `mcp:shogym` id in auth.json
    url = "http://127.0.0.1:8973/mcp"  # keep in step with serve.py's PORT
    bearer_token_env = "SHOGYM_MCP_TOKEN"

    async def _open_session(self, stack: AsyncExitStack):
        """As the base method, but with a client that will wait for a slow tool.

        Only the ``http_client=`` transport branch is reimplemented, because only that branch
        builds a client. Older SDKs taking ``headers=`` never hit the defect, so they are left
        to the base class rather than re-copied here.
        """
        import inspect

        from mcp import ClientSession
        from rlm.mcp_base import _resolve_streamable_http

        transport = _resolve_streamable_http()
        if "http_client" not in inspect.signature(transport).parameters:
            return await super()._open_session(stack)

        url, extra_headers = await self._resolve_config()
        if not url:
            raise ValueError(f"{type(self).__name__} must set `url` or override `_open_session`")
        token = await self._resolve_token()
        # Extra configured headers first, Authorization last so it always wins.
        auth_header = {**extra_headers, "Authorization": f"Bearer {token}"}

        client = await stack.enter_async_context(_timed_http_client(auth_header))
        read, write, *_ = await stack.enter_async_context(transport(url, http_client=client))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session


shogym_stream = ShogymStream()

# Forward bare module access (`import shogym_stream; await shogym_stream.pull()`) to the
# instance, but NOT the names the kernel bootstrap probes -- forwarding `run` would make it
# treat the module as a callable skill and break tool dispatch.
_RESERVED = {"run", "__wrapped__", "__call__"}


def __getattr__(name):
    if name.startswith("_") or name in _RESERVED:
        raise AttributeError(name)
    return getattr(shogym_stream, name)

from asyncio import Semaphore
from typing import TYPE_CHECKING, Any, Dict, Optional

from hgym.agents import LLMAgent
from hgym.agents.openai.utils import (
    get_action,
    get_client_kwargs,
    get_tools,
    parse_observation,
    to_openai_tool_choice,
)
from hgym.models import CompletionRequest, ModelClient, OpenAICompatClient
from hgym.types import (
    Action,
    FunctionConfigs,
    MetricConfigs,
    Observation,
    ToolConfigs,
)

if TYPE_CHECKING:
    from hgym.harness import Harness


class OpenAIAgent(LLMAgent):
    def __init__(
        self,
        model_name: str,
        function_configs: FunctionConfigs,
        tool_configs: Optional[ToolConfigs] = None,
        metric_configs: Optional[MetricConfigs] = None,
        semaphore: Optional[Semaphore] = None,
        *,
        inference_params: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        client: Optional[ModelClient] = None,
        system_template: Optional[str] = None,
    ):
        super().__init__(
            function_configs=function_configs,
            tool_configs=tool_configs,
            metric_configs=metric_configs,
            semaphore=semaphore,
        )
        self._client_kwargs = get_client_kwargs(
            function_configs=self._function_configs,
            tool_configs=self._tool_configs,
        )
        # The inference seam (RFC 001 Section 6): a swappable ModelClient plus the
        # inference params (temperature, max_tokens, ...) that travel with the model.
        self._client: ModelClient = (
            client if client is not None else OpenAICompatClient(base_url=base_url)
        )
        self._model_name = model_name
        self._inference_params: Dict[str, Any] = dict(inference_params or {})
        # The instruction surface (RFC 002): an optional system-template override. When
        # set (a loaded harness supplied one), it replaces the function's own template
        # for rendering the system message; the env still owns the variables that fill
        # it. ``None`` means defer to each function's example_system_template.
        self._system_template = system_template

    @classmethod
    def from_harness(
        cls,
        harness: "Harness",
        function_configs: FunctionConfigs,
        tool_configs: Optional[ToolConfigs] = None,
        metric_configs: Optional[MetricConfigs] = None,
        semaphore: Optional[Semaphore] = None,
        *,
        base_url: Optional[str] = None,
        client: Optional[ModelClient] = None,
    ) -> "OpenAIAgent":
        """Build an agent that applies a loaded :class:`Harness`'s agent-side surfaces.

        Wires the **inference** surface (``model`` + ``inference_params``) and the
        **instruction** surface (``system_template``). The harness's ``extra_specs``
        (tool surface) and ``horizon`` (control surface) are applied at the env/runner,
        not here, so they are intentionally untouched by this constructor.
        """
        return cls(
            harness.model,
            function_configs,
            tool_configs=tool_configs,
            metric_configs=metric_configs,
            semaphore=semaphore,
            inference_params=harness.inference_params,
            base_url=base_url,
            client=client,
            system_template=harness.system_template,
        )

    async def act(self, obs: Observation) -> Action:
        function_config = self._function_configs[obs.function_name]
        messages = parse_observation(obs, function_config, self._system_template)

        base = self._client_kwargs[obs.function_name]
        tools = base.get("tools")
        tool_choice = base.get("tool_choice")
        parallel = base.get("parallel_tool_calls")

        # Dynamic tools from the observation override the static function config (this
        # is how runtime extras reach an agent whose static config only knew the
        # env-mandatory tools).
        if obs.tools is not None:
            tools = get_tools(tool_configs=obs.tools, function_config=None)
            tool_choice = to_openai_tool_choice(obs.tool_choice) or "auto"
            if tools:
                parallel = False

        request = CompletionRequest(
            model=self._model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
            params=self._inference_params,
        )
        response = await self.throttle(self._client.complete(request))
        return get_action(response.choices, function_config)

    def reset(self):
        pass

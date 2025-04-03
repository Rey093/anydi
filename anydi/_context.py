from __future__ import annotations

import contextlib
import threading
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Callable, ClassVar

import abc

from ._provider import CallableKind, Provider
from ._types import AnyInterface, Scope
from ._utils import get_full_qualname

if TYPE_CHECKING:
    from ._container import Container


class ScopedContext(abc.ABC):
    """ScopedContext base class."""

    scope: ClassVar[Scope]

    def __init__(self, container: Container) -> None:
        self.container = container
        self._instances: dict[Any, Any] = {}

    def set(self, interface: AnyInterface, instance: Any) -> None:
        """Set an instance of a dependency in the scoped context."""
        self._instances[interface] = instance

    def alias(self, interface: AnyInterface, alias: AnyInterface) -> None:
        """Alias an instance of a dependency in the scoped context."""
        if interface in self._instances:
            self._instances[alias] = self._instances[interface]

    @abc.abstractmethod
    def get_or_create(self, provider: Provider) -> tuple[Any, bool]:
        """Get or create an instance of a dependency from the scoped context."""

    @abc.abstractmethod
    async def aget_or_create(self, provider: Provider) -> tuple[Any, bool]:
        """Get or create an async instance of a dependency from the scoped context."""

    def _create_instance(self, provider: Provider) -> Any:
        """Create an instance using the provider."""
        if provider.kind == CallableKind.COROUTINE:
            raise TypeError(
                f"The instance for the coroutine provider `{provider}` cannot be "
                "created in synchronous mode."
            )
        args, kwargs = self._get_provider_params(provider)
        return provider.call(*args, **kwargs)

    async def _acreate_instance(self, provider: Provider) -> Any:
        """Create an instance asynchronously using the provider."""
        args, kwargs = await self._aget_provider_params(provider)
        if provider.kind == CallableKind.COROUTINE:
            return await provider.call(*args, **kwargs)
        return await run_async(provider.call, *args, **kwargs)

    def _resolve_parameter(
        self, provider: Provider, parameter: inspect.Parameter
    ) -> Any:
        self._validate_resolvable_parameter(parameter, call=provider.call)
        return self.container.resolve(parameter.annotation)

    async def _aresolve_parameter(
        self, provider: Provider, parameter: inspect.Parameter
    ) -> Any:
        self._validate_resolvable_parameter(parameter, call=provider.call)
        return await self.container.aresolve(parameter.annotation)

    def _validate_resolvable_parameter(
        self, parameter: inspect.Parameter, call: Callable[..., Any]
    ) -> None:
        """Ensure that the specified interface is resolved."""
        if parameter.annotation in self.container._unresolved_interfaces:  # noqa
            raise LookupError(
                f"You are attempting to get the parameter `{parameter.name}` with the "
                f"annotation `{get_full_qualname(parameter.annotation)}` as a "
                f"dependency into `{get_full_qualname(call)}` which is not registered "
                "or set in the scoped context."
            )

    async def _aget_provider_params(
        self, provider: Provider
    ) -> tuple[list[Any], dict[str, Any]]:
        """Asynchronously retrieve the arguments for a provider."""
        args: list[Any] = []
        kwargs: dict[str, Any] = {}

        for parameter in provider.parameters:
            if parameter.annotation in self.container._override_instances:  # noqa
                instance = self.container._override_instances[parameter.annotation]  # noqa
            elif parameter.annotation in self._instances:
                instance = self._instances[parameter.annotation]
            else:
                instance = await self._aresolve_parameter(provider, parameter)
            if parameter.kind == parameter.POSITIONAL_ONLY:
                args.append(instance)
            else:
                kwargs[parameter.name] = instance
        return args, kwargs

from typing import Any

from typing_extensions import Self

from ._utils import AsyncRLock, run_async


class InstanceContext:
    """A context to store instances."""

    __slots__ = ("_instances", "_stack", "_async_stack", "_lock", "_async_lock")

    def __init__(self) -> None:
        self._instances: dict[type[Any], Any] = {}
        self._stack = contextlib.ExitStack()
        self._async_stack = contextlib.AsyncExitStack()
        self._lock = threading.RLock()
        self._async_lock = AsyncRLock()

    def get(self, interface: type[Any]) -> Any | None:
        """Get an instance from the context."""
        return self._instances.get(interface)

    def set(self, interface: type[Any], value: Any) -> None:
        """Set an instance in the context."""
        self._instances[interface] = value

    def enter(self, cm: contextlib.AbstractContextManager[Any]) -> Any:
        """Enter the context."""
        return self._stack.enter_context(cm)

    async def aenter(self, cm: contextlib.AbstractAsyncContextManager[Any]) -> Any:
        """Enter the context asynchronously."""
        return await self._async_stack.enter_async_context(cm)

    def __setitem__(self, interface: type[Any], value: Any) -> None:
        self._instances[interface] = value

    def __getitem__(self, interface: type[Any]) -> Any:
        return self._instances[interface]

    def __contains__(self, interface: type[Any]) -> bool:
        return interface in self._instances

    def __delitem__(self, interface: type[Any]) -> None:
        self._instances.pop(interface, None)

    def __enter__(self) -> Self:
        """Enter the context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        """Exit the context."""
        return self._stack.__exit__(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        """Close the scoped context."""
        self._stack.__exit__(None, None, None)

    async def __aenter__(self) -> Self:
        """Enter the context asynchronously."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the context asynchronously."""
        return await run_async(
            self.__exit__, exc_type, exc_val, exc_tb
        ) or await self._async_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def aclose(self) -> None:
        """Close the scoped context asynchronously."""
        await self.__aexit__(None, None, None)

    def lock(self) -> threading.RLock:
        """Acquire the context lock."""
        return self._lock

    def alock(self) -> AsyncRLock:
        """Acquire the context lock asynchronously."""
        return self._async_lock

r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/common/tracer.py
   - Role: Global Pipeline Execution Call Tracer.
   - Purpose: Records and visualizes the exact sequence, source module paths,
     function names, and execution latencies (duration_ms) of pipeline components
     executed during system startup and per-query execution.
================================================================================
"""

from contextlib import contextmanager
from functools import wraps
import inspect
import time
from typing import Any, Callable, Dict, List, Optional


class TraceEvent:
    """Represents a single step in pipeline execution."""

    def __init__(
        self,
        step: int,
        stage_name: str,
        source_file: str,
        function_name: str,
        duration_ms: float,
    ) -> None:
        self.step = step
        self.stage_name = stage_name
        self.source_file = source_file
        self.function_name = function_name
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "stage_name": self.stage_name,
            "source_file": self.source_file,
            "function_name": self.function_name,
            "duration_ms": round(self.duration_ms, 2),
        }

    def __repr__(self) -> str:
        dur = f"{int(round(self.duration_ms))}ms"
        return f"* Step {self.step} [{self.stage_name:<20}] -> {self.source_file}:{self.function_name} ({dur})"


class PipelineTracer:
    """Tracks execution order and latency across pipeline components."""

    def __init__(self) -> None:
        self._events: List[TraceEvent] = []

    def clear(self) -> None:
        """Reset all recorded trace events."""
        self._events.clear()

    def record_event(
        self,
        stage_name: str,
        source_file: str,
        function_name: str,
        duration_ms: float,
    ) -> TraceEvent:
        """Record an execution step event."""
        step_num = len(self._events) + 1
        event = TraceEvent(
            step=step_num,
            stage_name=stage_name,
            source_file=source_file,
            function_name=function_name,
            duration_ms=duration_ms,
        )
        self._events.append(event)
        return event

    def get_trace(self) -> List[Dict[str, Any]]:
        """Return list of all recorded trace events as dictionaries."""
        return [e.to_dict() for e in self._events]

    def format_trace(self) -> str:
        """Generate formatted call trace representation."""
        if not self._events:
            return (
                "------------------------------------------------------------\n"
                "[PIPELINE EXECUTION CALL TRACE]\n"
                "* (No steps recorded)\n"
                "------------------------------------------------------------"
            )

        lines = [
            "------------------------------------------------------------",
            "[PIPELINE EXECUTION CALL TRACE]",
        ]
        for e in self._events:
            stage_str = f"Step {e.step} [{e.stage_name}]"
            call_str = f"{e.source_file}:{e.function_name}"
            dur_str = f"({int(round(e.duration_ms))}ms)"
            lines.append(f"* {stage_str:<28} -> {call_str} {dur_str}")

        lines.append("------------------------------------------------------------")
        return "\n".join(lines)

    def print_trace(self) -> None:
        """Print formatted trace to stdout."""
        print(self.format_trace())

    def trace_step(
        self,
        stage: str,
        module_path: str,
        func_name: Optional[str] = None,
    ):
        """Decorator or context manager to trace a pipeline step without altering signatures."""
        tracer_self = self

        class _StepTracerContext:
            def __init__(self, stage_name: str, path: str, fn_name: Optional[str] = None):
                self.stage_name = stage_name
                self.module_path = path
                self.func_name = fn_name or "execute"
                self.t0 = 0.0

            def __enter__(self):
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                dur_ms = (time.perf_counter() - self.t0) * 1000
                tracer_self.record_event(self.stage_name, self.module_path, self.func_name, dur_ms)

            def __call__(self, fn: Callable) -> Callable:
                resolved_fn_name = self.func_name if self.func_name != "execute" else fn.__name__

                @wraps(fn)
                def wrapper(*args, **kwargs):
                    t0 = time.perf_counter()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        dur_ms = (time.perf_counter() - t0) * 1000
                        tracer_self.record_event(
                            self.stage_name,
                            self.module_path,
                            resolved_fn_name,
                            dur_ms,
                        )

                return wrapper

        return _StepTracerContext(stage, module_path, func_name)


# Global singleton instance
global_tracer = PipelineTracer()

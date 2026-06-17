"""Processing Graph — non-destructive DAG-based editing pipeline.

Each processing step is a node in a directed acyclic graph.
Nodes store parameters, not pixel data. The graph is evaluated
from the base image through selected nodes to produce output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from numpy.typing import NDArray

log = logging.getLogger(__name__)


class NodeType(Enum):
    BASE = auto()
    PROCESS = auto()
    MASK = auto()
    BLEND = auto()


@dataclass
class ProcessNode:
    """A single node in the processing graph."""

    node_id: str
    node_type: NodeType = NodeType.PROCESS
    process_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    locked: bool = False
    parent_ids: list[str] = field(default_factory=list)

    _cached_output: NDArray | None = field(default=None, repr=False)
    _cache_valid: bool = False
    _cache_params_hash: int = 0


@dataclass
class ProcessingGraph:
    """A DAG of processing steps applied to an image."""

    base_image: NDArray | None = field(default=None, repr=False)
    nodes: dict[str, ProcessNode] = field(default_factory=dict)
    root_id: str = "base"

    def __post_init__(self):
        if self.root_id not in self.nodes:
            self.nodes[self.root_id] = ProcessNode(
                node_id=self.root_id,
                node_type=NodeType.BASE,
                process_name="base_image",
            )

    def set_base(self, image: NDArray):
        self.base_image = image.copy()
        self._invalidate_all()

    def add_node(
        self,
        process_name: str,
        params: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
    ) -> str:
        """Add a processing node. Returns node_id."""
        node_id = f"{process_name}_{len(self.nodes)}"
        if parent_ids is None:
            parent_ids = [self.root_id]
        node = ProcessNode(
            node_id=node_id,
            process_name=process_name,
            params=params or {},
            parent_ids=parent_ids,
        )
        self.nodes[node_id] = node
        return node_id

    def remove_node(self, node_id: str):
        """Remove a node and all its dependents."""
        if node_id == self.root_id:
            return
        dependents = self._find_dependents(node_id)
        for nid in dependents:
            self.nodes.pop(nid, None)
        self.nodes.pop(node_id, None)

    def update_params(self, node_id: str, params: dict[str, Any]) -> None:
        """Replace a node's params and invalidate its downstream cache."""
        node = self.nodes.get(node_id)
        if node is None or node_id == self.root_id:
            return
        node.params = params
        self.invalidate_downstream(node_id)

    def update_enabled(self, node_id: str, enabled: bool) -> None:
        """Toggle a node and invalidate its downstream cache."""
        node = self.nodes.get(node_id)
        if node is None or node_id == self.root_id:
            return
        node.enabled = enabled
        self.invalidate_downstream(node_id)

    def _find_dependents(self, node_id: str) -> set[str]:
        """Find all nodes that depend (directly or indirectly) on node_id."""
        dependents = set()
        for nid, node in self.nodes.items():
            if node_id in node.parent_ids:
                dependents.add(nid)
                dependents |= self._find_dependents(nid)
        return dependents

    def _invalidate_all(self):
        for node in self.nodes.values():
            node._cache_valid = False
            node._cached_output = None

    def invalidate_downstream(self, node_id: str):
        """Invalidate cache for node_id and all dependents. Skips locked nodes."""
        for nid in self._find_dependents(node_id):
            if nid in self.nodes and not self.nodes[nid].locked:
                self.nodes[nid]._cache_valid = False
                self.nodes[nid]._cached_output = None
        if node_id in self.nodes and not self.nodes[node_id].locked:
            self.nodes[node_id]._cache_valid = False
            self.nodes[node_id]._cached_output = None

    def evaluate(
        self,
        node_id: str | None = None,
        process_fn: Callable[[str, dict[str, Any], NDArray], NDArray] | None = None,
    ) -> NDArray | None:
        """Evaluate the graph from root to the given node.

        Args:
            node_id: Target node. If None, evaluates the last added node.
            process_fn: Callable(process_name, params, image) -> processed_image.
                       If None, returns the raw cached value (useful for root).

        Returns:
            Processed image, or None if base_image is not set.
        """
        if self.base_image is None:
            return None

        if node_id is None:
            all_ids = set(self.nodes.keys())
            dependent_ids = set()
            for node in self.nodes.values():
                for pid in node.parent_ids:
                    dependent_ids.add(pid)
            leaf_ids = all_ids - dependent_ids - {self.root_id}
            node_id = sorted(leaf_ids)[-1] if leaf_ids else self.root_id

        order = self._topological_sort(node_id)
        if order is None:
            return None

        current = self.base_image.copy()
        for nid in order:
            nnode = self.nodes.get(nid)
            if nnode is None:
                continue
            if not nnode.enabled:
                continue
            if nid == self.root_id:
                continue
            params_hash = hash(frozenset(nnode.params.items()))
            if (
                nnode._cache_valid
                and nnode._cached_output is not None
                and nnode._cache_params_hash == params_hash
            ):
                current = nnode._cached_output
                continue
            if process_fn:
                try:
                    current = process_fn(nnode.process_name, nnode.params, current)
                    nnode._cached_output = current.copy()
                    nnode._cache_valid = True
                    nnode._cache_params_hash = params_hash
                except Exception as e:
                    log.error("Processing node %s failed: %s", nid, e)
                    return None

        return current

    def _topological_sort(self, target_id: str) -> list[str] | None:
        """Return nodes in topological order from root to target."""
        visited: set[str] = set()
        order: list[str] = []

        def _visit(vid: str) -> bool:
            if vid in visited:
                return True
            vnode = self.nodes.get(vid)
            if vnode is None:
                return False
            for pid in vnode.parent_ids:
                if not _visit(pid):
                    return False
            visited.add(vid)
            order.append(vid)
            return True

        if not _visit(target_id):
            return None
        return order

    def to_dict(self) -> dict:
        """Serialize graph to dict for saving."""
        return {
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "node_type": n.node_type.name,
                    "process_name": n.process_name,
                    "params": n.params,
                    "enabled": n.enabled,
                    "locked": n.locked,
                    "parent_ids": n.parent_ids,
                }
                for nid, n in self.nodes.items()
            },
            "root_id": self.root_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProcessingGraph:
        graph = cls()
        graph.root_id = data.get("root_id", "base")
        for nid, ndata in data.get("nodes", {}).items():
            graph.nodes[nid] = ProcessNode(
                node_id=ndata["node_id"],
                node_type=NodeType[ndata["node_type"]],
                process_name=ndata["process_name"],
                params=ndata.get("params", {}),
                enabled=ndata.get("enabled", True),
                locked=ndata.get("locked", False),
                parent_ids=ndata.get("parent_ids", []),
            )
        return graph

    def list_history(self) -> list[dict]:
        """Return human-readable processing history."""
        history = []
        for nid, node in self.nodes.items():
            if nid == self.root_id:
                continue
            lock = "\U0001f512" if node.locked else " "
            status = "\u2713" if node.enabled else "\u2717"
            dep_count = len(self._find_dependents(nid))
            display = f"{lock}{status} {node.process_name.replace('_', ' ').title()}"
            if dep_count:
                display += f" \u2192 {dep_count} dependent"
            history.append({
                "id": nid,
                "name": node.process_name,
                "enabled": node.enabled,
                "dependents": dep_count,
                "display": display,
            })
        return history

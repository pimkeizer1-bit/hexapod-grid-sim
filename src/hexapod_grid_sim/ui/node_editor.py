"""Visual scripting node editor using DearPyGUI for constraint/control logic."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


try:
    import dearpygui.dearpygui as dpg

    HAS_DPG = True
except ImportError:
    HAS_DPG = False

# ---------------------------------------------------------------------------
# Sci-fi colour palette
# ---------------------------------------------------------------------------
UI_BG = (8, 12, 20, 255)
UI_ACCENT = (0, 230, 255, 255)
UI_ACCENT2 = (255, 160, 0, 255)
UI_TEXT = (180, 230, 255, 255)
UI_WARNING = (255, 50, 50, 255)

_PIN_COLOURS = {}  # populated after DataType is defined


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class NodeCategory(Enum):
    INPUT = auto()
    OUTPUT = auto()
    MATH = auto()
    LOGIC = auto()
    CONSTRAINT = auto()
    HEXAPOD = auto()
    CONTEXT_INPUT = auto()
    CONTEXT_OUTPUT = auto()


class DataType(Enum):
    EXEC = auto()
    FLOAT = auto()
    HEIGHT = auto()
    ANGLE = auto()
    BOOL = auto()
    PLATFORM = auto()


_PIN_COLOURS.update(
    {
        DataType.EXEC: (200, 200, 200, 255),
        DataType.FLOAT: (0, 230, 255, 255),
        DataType.HEIGHT: (80, 255, 120, 255),
        DataType.ANGLE: (255, 200, 60, 255),
        DataType.BOOL: (255, 80, 80, 255),
        DataType.PLATFORM: (180, 100, 255, 255),
    }
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Pin:
    id: str
    name: str
    data_type: DataType
    is_input: bool
    value: Any = 0.0
    connected_to: Optional[str] = None


@dataclass
class Node:
    id: str
    name: str
    category: NodeCategory
    pins: list[Pin] = field(default_factory=list)
    position: tuple[float, float] = (0.0, 0.0)
    compute_func: Optional[Callable[["Node", "NodeGraph"], None]] = None

    def input_pin(self, name: str) -> Optional[Pin]:
        return next((p for p in self.pins if p.is_input and p.name == name), None)

    def output_pin(self, name: str) -> Optional[Pin]:
        return next((p for p in self.pins if not p.is_input and p.name == name), None)


# ---------------------------------------------------------------------------
# Node graph evaluation
# ---------------------------------------------------------------------------
class NodeGraph:
    """Manages nodes, connections, and topological evaluation."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.connections: list[tuple[str, str]] = []  # (output_pin_id, input_pin_id)
        self._pin_to_node: dict[str, str] = {}

    # -- mutation -----------------------------------------------------------
    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        for pin in node.pins:
            self._pin_to_node[pin.id] = node.id

    def remove_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        pin_ids = {p.id for p in node.pins}
        self.connections = [
            (a, b)
            for a, b in self.connections
            if a not in pin_ids and b not in pin_ids
        ]
        for pid in pin_ids:
            self._pin_to_node.pop(pid, None)

    def connect(self, output_pin_id: str, input_pin_id: str) -> bool:
        out_pin = self._find_pin(output_pin_id)
        in_pin = self._find_pin(input_pin_id)
        if out_pin is None or in_pin is None:
            return False
        if out_pin.is_input or not in_pin.is_input:
            return False
        if not self._types_compatible(out_pin.data_type, in_pin.data_type):
            return False
        # remove existing connection on the input side
        self.connections = [(a, b) for a, b in self.connections if b != input_pin_id]
        in_pin.connected_to = output_pin_id
        self.connections.append((output_pin_id, input_pin_id))
        return True

    def disconnect(self, output_pin_id: str, input_pin_id: str) -> None:
        in_pin = self._find_pin(input_pin_id)
        if in_pin:
            in_pin.connected_to = None
        self.connections = [
            (a, b)
            for a, b in self.connections
            if not (a == output_pin_id and b == input_pin_id)
        ]

    # -- evaluation ---------------------------------------------------------
    def evaluate(self) -> None:
        order = self._topological_sort()
        for node_id in order:
            node = self.nodes[node_id]
            # propagate connected values
            for pin in node.pins:
                if pin.is_input and pin.connected_to:
                    src = self._find_pin(pin.connected_to)
                    if src is not None:
                        pin.value = src.value
            if node.compute_func is not None:
                node.compute_func(node, self)

    # -- helpers ------------------------------------------------------------
    def _find_pin(self, pin_id: str) -> Optional[Pin]:
        node_id = self._pin_to_node.get(pin_id)
        if node_id is None:
            return None
        node = self.nodes.get(node_id)
        if node is None:
            return None
        return next((p for p in node.pins if p.id == pin_id), None)

    @staticmethod
    def _types_compatible(src: DataType, dst: DataType) -> bool:
        if src == dst:
            return True
        numeric = {DataType.FLOAT, DataType.HEIGHT, DataType.ANGLE}
        return src in numeric and dst in numeric

    def _topological_sort(self) -> list[str]:
        in_edges: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        for out_pid, in_pid in self.connections:
            src_nid = self._pin_to_node.get(out_pid)
            dst_nid = self._pin_to_node.get(in_pid)
            if src_nid and dst_nid:
                in_edges[dst_nid].add(src_nid)

        order: list[str] = []
        ready = [nid for nid, deps in in_edges.items() if not deps]
        while ready:
            nid = ready.pop()
            order.append(nid)
            for other, deps in in_edges.items():
                deps.discard(nid)
                if not deps and other not in order and other not in ready:
                    ready.append(other)
        # append any remaining (cycles) so nothing is silently dropped
        for nid in self.nodes:
            if nid not in order:
                order.append(nid)
        return order


# ---------------------------------------------------------------------------
# Built-in compute functions
# ---------------------------------------------------------------------------
def _compute_constant(node: Node, _graph: NodeGraph) -> None:
    out = node.output_pin("Value")
    if out:
        pass  # value already set via property panel


def _compute_add(node: Node, _graph: NodeGraph) -> None:
    a = node.input_pin("A")
    b = node.input_pin("B")
    out = node.output_pin("Result")
    if a and b and out:
        out.value = float(a.value) + float(b.value)


def _compute_multiply(node: Node, _graph: NodeGraph) -> None:
    a = node.input_pin("A")
    b = node.input_pin("B")
    out = node.output_pin("Result")
    if a and b and out:
        out.value = float(a.value) * float(b.value)


def _compute_sin(node: Node, _graph: NodeGraph) -> None:
    inp = node.input_pin("X")
    out = node.output_pin("Result")
    if inp and out:
        out.value = math.sin(float(inp.value))


def _compute_time(node: Node, _graph: NodeGraph) -> None:
    out = node.output_pin("T")
    if out:
        out.value = time.time() % 1e6


def _compute_height_constraint(node: Node, _graph: NodeGraph) -> None:
    h = node.input_pin("Height")
    mn = node.input_pin("Min")
    mx = node.input_pin("Max")
    out = node.output_pin("Clamped")
    if h and mn and mx and out:
        out.value = max(float(mn.value), min(float(mx.value), float(h.value)))


def _compute_pitch_constraint(node: Node, _graph: NodeGraph) -> None:
    angle = node.input_pin("Angle")
    limit = node.input_pin("Limit")
    out = node.output_pin("Clamped")
    if angle and limit and out:
        lim = abs(float(limit.value))
        out.value = max(-lim, min(lim, float(angle.value)))


def _compute_anchor(node: Node, _graph: NodeGraph) -> None:
    locked = node.input_pin("Locked")
    out = node.output_pin("Active")
    if locked and out:
        out.value = bool(locked.value)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------
def _uid() -> str:
    return uuid.uuid4().hex[:12]


_TEMPLATES: dict[str, Callable[[], Node]] = {}


def _register(name: str, category: NodeCategory, inputs: list[tuple[str, DataType]],
              outputs: list[tuple[str, DataType]], compute: Optional[Callable]) -> None:
    def factory() -> Node:
        nid = _uid()
        pins: list[Pin] = []
        for pname, dtype in inputs:
            pins.append(Pin(id=f"{nid}_i_{pname}", name=pname, data_type=dtype, is_input=True))
        for pname, dtype in outputs:
            pins.append(Pin(id=f"{nid}_o_{pname}", name=pname, data_type=dtype, is_input=False))
        return Node(id=nid, name=name, category=category, pins=pins, compute_func=compute)

    _TEMPLATES[name] = factory


_register("Constant", NodeCategory.INPUT, [], [("Value", DataType.FLOAT)], _compute_constant)
_register("Add", NodeCategory.MATH, [("A", DataType.FLOAT), ("B", DataType.FLOAT)],
          [("Result", DataType.FLOAT)], _compute_add)
_register("Multiply", NodeCategory.MATH, [("A", DataType.FLOAT), ("B", DataType.FLOAT)],
          [("Result", DataType.FLOAT)], _compute_multiply)
_register("Sin", NodeCategory.MATH, [("X", DataType.FLOAT)], [("Result", DataType.FLOAT)], _compute_sin)
_register("Time", NodeCategory.INPUT, [], [("T", DataType.FLOAT)], _compute_time)
_register("HeightConstraint", NodeCategory.CONSTRAINT,
          [("Height", DataType.HEIGHT), ("Min", DataType.FLOAT), ("Max", DataType.FLOAT)],
          [("Clamped", DataType.HEIGHT)], _compute_height_constraint)
_register("PitchConstraint", NodeCategory.CONSTRAINT,
          [("Angle", DataType.ANGLE), ("Limit", DataType.FLOAT)],
          [("Clamped", DataType.ANGLE)], _compute_pitch_constraint)
_register("AnchorNode", NodeCategory.HEXAPOD,
          [("Locked", DataType.BOOL)], [("Active", DataType.BOOL)], _compute_anchor)


def create_node(template_name: str) -> Optional[Node]:
    factory = _TEMPLATES.get(template_name)
    return factory() if factory else None


# ---------------------------------------------------------------------------
# DearPyGUI-based node editor window
# ---------------------------------------------------------------------------
class NodeEditorWindow:
    """Interactive node editor UI backed by DearPyGUI."""

    def __init__(self, graph: Optional[NodeGraph] = None) -> None:
        if not HAS_DPG:
            raise RuntimeError("DearPyGUI is required: pip install dearpygui")
        self.graph = graph or NodeGraph()
        self._dpg_node_map: dict[str, int] = {}  # node.id -> dpg node tag
        self._dpg_pin_map: dict[str, int] = {}   # pin.id  -> dpg attribute tag
        self._selected_node_id: Optional[str] = None
        self._props_group: Optional[int] = None

    # -- public API ---------------------------------------------------------
    def setup(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Hexapod Node Editor", width=1280, height=720)
        dpg.setup_dearpygui()

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, UI_BG[:3], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text, UI_TEXT[:3], category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (20, 40, 60), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (20, 50, 70), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (30, 80, 110), category=dpg.mvThemeCat_Core)
        dpg.bind_theme(global_theme)

        self._build_palette_panel()
        self._build_editor_panel()
        self._build_properties_panel()

    def run(self) -> None:
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            self.graph.evaluate()
            self._refresh_properties()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    # -- UI builders --------------------------------------------------------
    def _build_palette_panel(self) -> None:
        with dpg.window(label="Node Palette", width=200, height=720, pos=(0, 0), no_close=True):
            by_cat: dict[NodeCategory, list[str]] = {}
            for name, factory in _TEMPLATES.items():
                node = factory()
                by_cat.setdefault(node.category, []).append(name)
            for cat, names in by_cat.items():
                with dpg.collapsing_header(label=cat.name, default_open=True):
                    for n in names:
                        dpg.add_button(label=n, callback=self._on_add_node, user_data=n, width=-1)

    def _build_editor_panel(self) -> None:
        with dpg.window(label="Canvas", width=780, height=720, pos=(200, 0), no_close=True):
            with dpg.node_editor(callback=self._on_link, delink_callback=self._on_delink) as self._editor_id:
                pass

    def _build_properties_panel(self) -> None:
        with dpg.window(label="Properties", width=300, height=720, pos=(980, 0), no_close=True):
            self._props_group = dpg.add_group()
            dpg.add_text("Select a node", parent=self._props_group)

    # -- callbacks ----------------------------------------------------------
    def _on_add_node(self, _sender: int, _data: Any, template_name: str) -> None:
        node = create_node(template_name)
        if node is None:
            return
        self.graph.add_node(node)
        self._spawn_dpg_node(node)

    def _spawn_dpg_node(self, node: Node) -> None:
        label_colour = UI_ACCENT[:3] if node.category in (
            NodeCategory.INPUT, NodeCategory.CONTEXT_INPUT) else UI_ACCENT2[:3]

        with dpg.node(label=node.name, parent=self._editor_id,
                      pos=list(node.position)) as dpg_node:
            self._dpg_node_map[node.id] = dpg_node
            dpg.set_item_user_data(dpg_node, node.id)

            for pin in node.pins:
                kind = dpg.mvNode_Attr_Input if pin.is_input else dpg.mvNode_Attr_Output
                shape = dpg.mvNode_PinShape_CircleFilled
                with dpg.node_attribute(attribute_type=kind, shape=shape) as attr:
                    self._dpg_pin_map[pin.id] = attr
                    colour = _PIN_COLOURS.get(pin.data_type, UI_TEXT[:3])
                    dpg.add_text(pin.name, color=colour)

        # click handler for selection
        with dpg.item_handler_registry() as handler:
            dpg.add_item_clicked_handler(callback=self._on_node_clicked, user_data=node.id)
        dpg.bind_item_handler_registry(dpg_node, handler)

    def _on_node_clicked(self, _sender: int, _data: Any, node_id: str) -> None:
        self._selected_node_id = node_id
        self._rebuild_properties()

    def _on_link(self, sender: int, link_data: tuple[int, int]) -> None:
        attr_out, attr_in = link_data
        out_pin_id = self._attr_to_pin_id(attr_out)
        in_pin_id = self._attr_to_pin_id(attr_in)
        if out_pin_id and in_pin_id:
            if self.graph.connect(out_pin_id, in_pin_id):
                dpg.add_node_link(attr_out, attr_in, parent=sender)

    def _on_delink(self, _sender: int, link_id: int) -> None:
        config = dpg.get_item_configuration(link_id)
        attr_out = config.get("attr_1")
        attr_in = config.get("attr_2")
        if attr_out is not None and attr_in is not None:
            out_pid = self._attr_to_pin_id(attr_out)
            in_pid = self._attr_to_pin_id(attr_in)
            if out_pid and in_pid:
                self.graph.disconnect(out_pid, in_pid)
        dpg.delete_item(link_id)

    # -- properties panel ---------------------------------------------------
    def _rebuild_properties(self) -> None:
        if self._props_group is None:
            return
        dpg.delete_item(self._props_group, children_only=True)
        node = self.graph.nodes.get(self._selected_node_id or "")
        if node is None:
            dpg.add_text("No selection", parent=self._props_group)
            return
        dpg.add_text(f"Node: {node.name}", parent=self._props_group, color=UI_ACCENT[:3])
        dpg.add_text(f"Category: {node.category.name}", parent=self._props_group)
        dpg.add_separator(parent=self._props_group)
        for pin in node.pins:
            if pin.is_input and pin.connected_to is None:
                dpg.add_input_float(
                    label=pin.name, default_value=float(pin.value),
                    callback=self._on_pin_value_changed, user_data=pin.id,
                    parent=self._props_group, width=150,
                )
            elif not pin.is_input:
                dpg.add_text(
                    f"{pin.name}: {pin.value:.4f}" if isinstance(pin.value, float) else f"{pin.name}: {pin.value}",
                    parent=self._props_group, tag=f"_prop_{pin.id}",
                )
        dpg.add_separator(parent=self._props_group)
        dpg.add_button(label="Delete Node", parent=self._props_group,
                       callback=self._on_delete_node, user_data=node.id)

    def _refresh_properties(self) -> None:
        node = self.graph.nodes.get(self._selected_node_id or "")
        if node is None:
            return
        for pin in node.pins:
            if not pin.is_input and dpg.does_item_exist(f"_prop_{pin.id}"):
                label = f"{pin.name}: {pin.value:.4f}" if isinstance(pin.value, float) else f"{pin.name}: {pin.value}"
                dpg.set_value(f"_prop_{pin.id}", label)

    def _on_pin_value_changed(self, _sender: int, value: float, pin_id: str) -> None:
        pin = self.graph._find_pin(pin_id)
        if pin:
            pin.value = value

    def _on_delete_node(self, _sender: int, _data: Any, node_id: str) -> None:
        dpg_tag = self._dpg_node_map.pop(node_id, None)
        if dpg_tag and dpg.does_item_exist(dpg_tag):
            dpg.delete_item(dpg_tag)
        self.graph.remove_node(node_id)
        self._selected_node_id = None
        self._rebuild_properties()

    # -- util ---------------------------------------------------------------
    def _attr_to_pin_id(self, attr_tag: int) -> Optional[str]:
        for pid, atag in self._dpg_pin_map.items():
            if atag == attr_tag:
                return pid
        return None


# ---------------------------------------------------------------------------
# Convenience launcher
# ---------------------------------------------------------------------------
def start_node_editor(*, threaded: bool = False, graph: Optional[NodeGraph] = None) -> Optional[threading.Thread]:
    """Launch the node editor, optionally in a background thread."""
    editor = NodeEditorWindow(graph)

    def _run() -> None:
        editor.setup()
        editor.run()

    if threaded:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t
    _run()
    return None


if __name__ == "__main__":
    start_node_editor()

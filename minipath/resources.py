"""Resource manager: virtual devices -> physical workers.

User code never names a physical worker. It asks for a slice of virtual
devices, and every dispatch resolves virtual to physical at send time. That
one level of indirection lets the controller swap a dead worker for a spare
without the training loop ever seeing a new device id.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class OutOfCapacity(RuntimeError):
    pass


@dataclass(frozen=True)
class Slice:
    """An ordered group of virtual devices. Rank i is ``vdevs[i]``."""

    vdevs: tuple
    name: str = ""

    def __len__(self) -> int:
        return len(self.vdevs)

    def __iter__(self):
        return iter(self.vdevs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Slice(self.vdevs[i], self.name)
        return Slice((self.vdevs[i],), self.name)


@dataclass
class ResourceManager:
    workers: list
    spares: list = field(default_factory=list)

    def __post_init__(self):
        self._lock = threading.Lock()
        self._free = list(self.workers)
        self._map: dict = {}
        self._next_vdev = 0
        self.dead: set = set()
        self.events: list = []

    def allocate(self, n: int, name: str = "") -> Slice:
        with self._lock:
            if n > len(self._free):
                raise OutOfCapacity(f"asked for {n} devices, {len(self._free)} free")
            vdevs = []
            for _ in range(n):
                v = self._next_vdev
                self._next_vdev += 1
                self._map[v] = self._free.pop(0)
                vdevs.append(v)
            return Slice(tuple(vdevs), name)

    def release(self, s: Slice) -> None:
        with self._lock:
            for v in s.vdevs:
                w = self._map.pop(v)
                if w not in self.dead:
                    self._free.append(w)

    def physical(self, v: int) -> int:
        return self._map[v]

    def physicals(self, s: Slice) -> list:
        return [self._map[v] for v in s.vdevs]

    def mark_dead(self, w: int) -> list:
        """Record a failure. Returns the virtual devices that were on ``w``."""
        with self._lock:
            self.dead.add(w)
            if w in self._free:
                self._free.remove(w)
            if w in self.spares:
                self.spares.remove(w)
            hit = [v for v, pw in self._map.items() if pw == w]
            self.events.append(("dead", w, hit))
            return hit

    def remap(self, v: int) -> int | None:
        """Move virtual device ``v`` onto a hot spare. Returns the new worker or None."""
        with self._lock:
            if not self.spares:
                return None
            w = self.spares.pop(0)
            old = self._map[v]
            self._map[v] = w
            self.events.append(("remap", v, old, w))
            return w

    def shrink(self, s: Slice) -> Slice:
        """Drop the virtual devices whose worker is dead (elastic shrink)."""
        with self._lock:
            keep = tuple(v for v in s.vdevs if self._map[v] not in self.dead)
            for v in s.vdevs:
                if v not in keep:
                    self._map.pop(v)
            self.events.append(("shrink", len(s), len(keep)))
            return Slice(keep, s.name)

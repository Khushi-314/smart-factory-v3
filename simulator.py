from __future__ import annotations

from dataclasses import dataclass, field
from math import gamma, log, sqrt
from typing import Dict, List, Optional, Tuple
import heapq
import itertools
import numpy as np


# -----------------------------
# Configuration / distributions
# -----------------------------

@dataclass(frozen=True)
class ProcessSpec:
    mean_s: float
    cv: float
    family: str = "truncnorm"  # truncnorm | lognormal

    @property
    def sd_s(self) -> float:
        return self.mean_s * self.cv


@dataclass(frozen=True)
class ReliabilitySpec:
    mttf_h: float
    weibull_k: float
    mean_repair_min: float
    repair_cv: float = 0.40

    @property
    def weibull_lambda_s(self) -> float:
        mttf_s = self.mttf_h * 3600.0
        return mttf_s / gamma(1.0 + 1.0 / self.weibull_k)


@dataclass(frozen=True)
class StationSpec:
    name: str
    process: ProcessSpec
    reliability: Optional[ReliabilitySpec] = None
    station_type: str = "machine"  # machine | human


@dataclass(frozen=True)
class ModelConfig:
    routes: Dict[str, Tuple[str, ...]]
    stations: Dict[str, StationSpec]
    buffer_capacity: int = 3
    convergence_buffer_capacity: int = 5
    convergence_stations: Tuple[str, ...] = ("WS4", "MC5", "MC10", "WS9")


ROUTES = {
    "PR1": ("MC1", "MC2", "WS4", "MC5", "MC6", "MC8", "MC10", "WS9"),
    "PR2": ("MC3", "MC4", "WS4", "MC5", "MC6", "MC9", "MC10", "WS9"),
    "PR3": ("MC1", "MC2", "WS4", "MC5", "MC7", "MC8", "MC10", "WS9"),
    "PR4": ("MC3", "MC4", "WS4", "MC5", "MC7", "MC9", "MC10", "WS9"),
}


def build_default_config() -> ModelConfig:
    # Class-derived assumptions:
    # CNC: CV=.08, MTTF=4.5 h, k=1.7
    # feeder/press: CV=.10, MTTF=5 h, k=1.6
    # inspection: CV=.10, MTTF=7 h, k=1.4
    # human: CV=.20, lognormal; no mechanical failure
    cnc_rel = ReliabilitySpec(mttf_h=4.5, weibull_k=1.7, mean_repair_min=9.0, repair_cv=0.40)
    press_rel = ReliabilitySpec(mttf_h=5.0, weibull_k=1.6, mean_repair_min=6.0, repair_cv=0.40)
    inspect_rel = ReliabilitySpec(mttf_h=7.0, weibull_k=1.4, mean_repair_min=5.0, repair_cv=0.35)

    stations = {
        "MC1": StationSpec("MC1", ProcessSpec(55.0, 0.08), cnc_rel),
        "MC2": StationSpec("MC2", ProcessSpec(58.0, 0.08), cnc_rel),
        "MC3": StationSpec("MC3", ProcessSpec(52.0, 0.08), cnc_rel),
        "MC4": StationSpec("MC4", ProcessSpec(47.0, 0.08), cnc_rel),
        "WS4": StationSpec("WS4", ProcessSpec(15.0, 0.20, "lognormal"), None, "human"),
        "MC5": StationSpec("MC5", ProcessSpec(12.0, 0.10), press_rel),
        # Excel nominal anchors:
        "MC6": StationSpec("MC6", ProcessSpec(25.0, 0.08), press_rel),
        "MC7": StationSpec("MC7", ProcessSpec(25.0, 0.08), press_rel),
        "MC8": StationSpec("MC8", ProcessSpec(5.0, 0.10), press_rel),
        "MC9": StationSpec("MC9", ProcessSpec(5.0, 0.10), press_rel),
        "MC10": StationSpec("MC10", ProcessSpec(10.0, 0.10), inspect_rel),
        "WS9": StationSpec("WS9", ProcessSpec(25.0, 0.20, "lognormal"), None, "human"),
    }
    return ModelConfig(routes=ROUTES, stations=stations)


def sample_positive_process(rng: np.random.Generator, spec: ProcessSpec) -> float:
    if spec.family == "lognormal":
        # Convert arithmetic mean/CV to log-space parameters.
        sigma2 = log(1.0 + spec.cv * spec.cv)
        sigma = sqrt(sigma2)
        mu = log(spec.mean_s) - 0.5 * sigma2
        return float(rng.lognormal(mu, sigma))

    # Tight industrial cycle times: resample Normal until positive.
    # Negative probability is negligible with the chosen CVs, but this is robust.
    sd = spec.sd_s
    for _ in range(100):
        x = float(rng.normal(spec.mean_s, sd))
        if x > 0.0:
            return x
    return max(spec.mean_s * 0.01, 1e-6)


def sample_weibull_life_s(rng: np.random.Generator, rel: ReliabilitySpec) -> float:
    # NumPy Weibull gives the unit-scale random variate.
    return float(rel.weibull_lambda_s * rng.weibull(rel.weibull_k))


def sample_lognormal_from_mean_cv(rng: np.random.Generator, mean: float, cv: float) -> float:
    sigma2 = log(1.0 + cv * cv)
    sigma = sqrt(sigma2)
    mu = log(mean) - 0.5 * sigma2
    return float(rng.lognormal(mu, sigma))


# -----------------------------
# Simulation entities
# -----------------------------

@dataclass
class Product:
    pid: int
    ptype: str
    route: Tuple[str, ...]
    route_idx: int = 0
    order_created_at: float = 0.0
    released_at: Optional[float] = None
    completed_at: Optional[float] = None
    waiting_time: float = 0.0
    source_waiting_time: float = 0.0
    first_admitted_at: Optional[float] = None
    queue_entered_at: Optional[float] = None
    service_time: float = 0.0
    failure_delay: float = 0.0
    blocked_delay: float = 0.0

    @property
    def current_station(self) -> Optional[str]:
        if self.route_idx >= len(self.route):
            return None
        return self.route[self.route_idx]


@dataclass
class StationRuntime:
    spec: StationSpec
    queue: List[Product] = field(default_factory=list)
    current: Optional[Product] = None
    current_remaining_s: float = 0.0
    current_token: int = 0
    failed: bool = False
    failure_life_remaining_s: Optional[float] = None
    blocked_product: Optional[Product] = None
    state: str = "STARVED"
    state_since: float = 0.0
    state_time: Dict[str, float] = field(default_factory=lambda: {"PROCESSING":0.0,"STARVED":0.0,"BLOCKED":0.0,"FAILED":0.0})
    failures: int = 0
    repair_time_s: float = 0.0
    processed_count: int = 0
    queue_area: float = 0.0
    queue_last_changed: float = 0.0
    queue_max: int = 0

    def update_queue_stats(self, now: float) -> None:
        self.queue_area += len(self.queue) * (now - self.queue_last_changed)
        self.queue_last_changed = now

    def transition(self, now: float, new_state: str) -> None:
        self.state_time[self.state] += now - self.state_since
        self.state = new_state
        self.state_since = now


@dataclass
class RunResult:
    makespan_s: float
    completed: int
    throughput_per_h: float
    product_results: List[dict]
    product_summary: Dict[str, dict]
    station_results: Dict[str, dict]
    event_log: List[dict]
    total_failures: int
    max_lead_time_decomposition_error_s: float
    max_order_fulfillment_decomposition_error_s: float


class FactorySimulation:
    """Event-driven multi-product factory simulation.

    Blocking semantics:
    - Each station has a finite input queue.
    - A completed product stays on its upstream station while the next station's
      queue is full; the upstream station is BLOCKED and cannot start another job.
    - On any downstream queue departure, blocked upstream completions are retried.

    Failure semantics:
    - Failure age accrues only during PROCESSING.
    - If failure life expires mid-job, processing pauses, repair occurs, and the
      same job resumes from remaining processing time (preempt-resume).
    - Perfect repair resets operating age / samples a new Weibull life.
    """

    def __init__(self, config: ModelConfig, order_counts: Dict[str, int], seed: int = 12345, log_events: bool = True):
        self.config = config
        self.order_counts = {k:int(v) for k,v in order_counts.items()}
        self.rng = np.random.default_rng(seed)
        self.now = 0.0
        self._counter = itertools.count()
        self.events: List[Tuple[float,int,str,tuple]] = []
        self.products: List[Product] = []
        self.completed_products: List[Product] = []
        self.log_events = log_events
        self.event_log: List[dict] = []
        self.stations = {name: StationRuntime(spec) for name,spec in config.stations.items()}

        for st in self.stations.values():
            if st.spec.reliability:
                st.failure_life_remaining_s = sample_weibull_life_s(self.rng, st.spec.reliability)

        pid = 0
        # Deterministic round-robin release ordering across requested product types.
        remaining = self.order_counts.copy()
        active = [p for p in ("PR1","PR2","PR3","PR4") if remaining.get(p,0) > 0]
        while active:
            next_active = []
            for ptype in active:
                if remaining[ptype] > 0:
                    pid += 1
                    self.products.append(Product(pid=pid, ptype=ptype, route=config.routes[ptype]))
                    remaining[ptype] -= 1
                if remaining[ptype] > 0:
                    next_active.append(ptype)
            active = next_active

    def _log(self, event: str, station: Optional[str] = None, product: Optional[Product] = None, **details) -> None:
        if not self.log_events:
            return
        row = {
            "time_s": float(self.now),
            "event": event,
            "station": station,
            "pid": product.pid if product is not None else None,
            "ptype": product.ptype if product is not None else None,
        }
        row.update(details)
        self.event_log.append(row)

    def _schedule(self, when: float, kind: str, payload: tuple) -> None:
        heapq.heappush(self.events, (when, next(self._counter), kind, payload))

    def _capacity(self, station_name: str) -> int:
        if station_name in self.config.convergence_stations:
            return self.config.convergence_buffer_capacity
        return self.config.buffer_capacity

    def _enqueue(self, product: Product, station_name: str) -> bool:
        st = self.stations[station_name]
        if len(st.queue) >= self._capacity(station_name):
            return False
        st.update_queue_stats(self.now)
        product.queue_entered_at = self.now
        st.queue.append(product)
        st.queue_max = max(st.queue_max, len(st.queue))
        self._log("QUEUE_ENTER", station_name, product, queue_length=len(st.queue))
        self._try_start(station_name)
        return True

    def _try_start(self, station_name: str) -> None:
        st = self.stations[station_name]
        if st.failed or st.current is not None or st.blocked_product is not None:
            return
        if not st.queue:
            if st.state != "STARVED":
                st.transition(self.now, "STARVED")
            return

        st.update_queue_stats(self.now)
        product = st.queue.pop(0)
        self._log("QUEUE_LEAVE", station_name, product, queue_length=len(st.queue))
        if product.queue_entered_at is not None:
            product.waiting_time += self.now - product.queue_entered_at
            product.queue_entered_at = None

        st.current = product
        st.current_remaining_s = sample_positive_process(self.rng, st.spec.process)
        st.current_token += 1
        token = st.current_token
        st.transition(self.now, "PROCESSING")
        self._log("PROCESS_START", station_name, product, sampled_process_s=st.current_remaining_s)
        self._schedule_next_processing_event(station_name, token)
        self._retry_blocked_upstream()  # queue slot just opened

    def _schedule_next_processing_event(self, station_name: str, token: int) -> None:
        st = self.stations[station_name]
        assert st.current is not None

        if st.spec.reliability is None:
            dt = st.current_remaining_s
            self._schedule(self.now + dt, "COMPLETE", (station_name, token, dt))
            return

        life = st.failure_life_remaining_s
        assert life is not None
        # Exact tie: complete the product first. Probability-zero continuous edge case.
        if st.current_remaining_s <= life + 1e-12:
            dt = st.current_remaining_s
            self._schedule(self.now + dt, "COMPLETE", (station_name, token, dt))
        else:
            dt = life
            self._schedule(self.now + dt, "FAIL", (station_name, token, dt))

    def _on_complete(self, station_name: str, token: int, dt: float) -> None:
        st = self.stations[station_name]
        if token != st.current_token or st.current is None or st.failed:
            return
        product = st.current
        st.current_remaining_s -= dt
        product.service_time += dt
        if st.spec.reliability:
            st.failure_life_remaining_s = max(0.0, st.failure_life_remaining_s - dt)  # type: ignore

        st.current_remaining_s = 0.0
        st.processed_count += 1
        self._log("PROCESS_COMPLETE", station_name, product)
        product.route_idx += 1

        if product.route_idx >= len(product.route):
            product.completed_at = self.now
            self.completed_products.append(product)
            self._log("PRODUCT_COMPLETE", station_name, product)
            st.current = None
            st.transition(self.now, "STARVED")
            self._try_start(station_name)
            return

        next_station = product.current_station
        assert next_station is not None
        if self._enqueue(product, next_station):
            st.current = None
            st.transition(self.now, "STARVED")
            self._try_start(station_name)
        else:
            # Product physically remains on upstream machine after service completion.
            st.blocked_product = product
            st.current = None
            st.transition(self.now, "BLOCKED")
            self._log("BLOCKED", station_name, product, next_station=next_station)

    def _on_fail(self, station_name: str, token: int, dt: float) -> None:
        st = self.stations[station_name]
        if token != st.current_token or st.current is None or st.failed:
            return

        product = st.current
        st.current_remaining_s -= dt
        product.service_time += dt
        st.failure_life_remaining_s = 0.0
        st.failed = True
        st.failures += 1
        st.transition(self.now, "FAILED")
        self._log("FAILURE", station_name, product, remaining_process_s=st.current_remaining_s)

        rel = st.spec.reliability
        assert rel is not None
        repair_s = sample_lognormal_from_mean_cv(self.rng, rel.mean_repair_min * 60.0, rel.repair_cv)
        st.repair_time_s += repair_s
        product.failure_delay += repair_s
        self._log("REPAIR_START", station_name, product, repair_duration_s=repair_s)
        self._schedule(self.now + repair_s, "REPAIR_DONE", (station_name, token))

    def _on_repair_done(self, station_name: str, token: int) -> None:
        st = self.stations[station_name]
        if token != st.current_token or st.current is None or not st.failed:
            return
        st.failed = False
        rel = st.spec.reliability
        assert rel is not None
        st.failure_life_remaining_s = sample_weibull_life_s(self.rng, rel)
        st.transition(self.now, "PROCESSING")
        self._log("REPAIR_COMPLETE", station_name, st.current)
        self._schedule_next_processing_event(station_name, token)

    def _retry_blocked_upstream(self) -> None:
        # Repeatedly release any completed products whose next queue now has room.
        changed = True
        while changed:
            changed = False
            for name, st in self.stations.items():
                p = st.blocked_product
                if p is None:
                    continue
                nxt = p.current_station
                if nxt is None:
                    raise RuntimeError("Blocked product has no next station")
                if len(self.stations[nxt].queue) < self._capacity(nxt):
                    blocked_duration = self.now - st.state_since
                    p.blocked_delay += max(0.0, blocked_duration)
                    # Transition before enqueue because enqueue can recursively start downstream.
                    st.blocked_product = None
                    st.transition(self.now, "STARVED")
                    self._log("UNBLOCKED", name, p, blocked_duration_s=max(0.0, blocked_duration), next_station=nxt)
                    ok = self._enqueue(p, nxt)
                    if not ok:
                        raise RuntimeError("Buffer availability changed unexpectedly")
                    self._try_start(name)
                    changed = True

    def run(self) -> RunResult:
        # All customer orders exist at t=0, but they remain outside the factory
        # until their first-station input buffer has room.  The in-factory
        # lead-time clock starts only at SOURCE_ADMIT.
        source_backlog: Dict[str, List[Product]] = {name: [] for name in self.stations}
        for p in self.products:
            self._log("ORDER_CREATED", p.current_station, p)
            source_backlog[p.current_station].append(p)  # type: ignore[index]

        def admit_backlog() -> bool:
            changed = False
            for stname, backlog in source_backlog.items():
                while backlog and len(self.stations[stname].queue) < self._capacity(stname):
                    p = backlog.pop(0)
                    if p.first_admitted_at is None:
                        p.first_admitted_at = self.now
                        p.released_at = self.now
                        p.source_waiting_time = self.now - p.order_created_at
                        self._log("SOURCE_ADMIT", stname, p, source_waiting_s=p.source_waiting_time)
                    if self._enqueue(p, stname):
                        changed = True
                    else:
                        raise RuntimeError("Source admission found no buffer space after capacity check")
            return changed

        admit_backlog()

        total = len(self.products)
        while len(self.completed_products) < total:
            if not self.events:
                # Before declaring deadlock, try source/backpressure admissions.
                admitted = admit_backlog()
                self._retry_blocked_upstream()
                if not self.events and not admitted:
                    raise RuntimeError("Simulation deadlocked with unfinished products")

            t, _, kind, payload = heapq.heappop(self.events)
            if t < self.now - 1e-9:
                raise RuntimeError("Event time moved backwards")
            self.now = t

            if kind == "COMPLETE":
                self._on_complete(*payload)
            elif kind == "FAIL":
                self._on_fail(*payload)
            elif kind == "REPAIR_DONE":
                self._on_repair_done(*payload)
            else:
                raise RuntimeError(f"Unknown event kind {kind}")

            admit_backlog()
            self._retry_blocked_upstream()

        makespan = self.now
        # close state accounting at makespan
        for st in self.stations.values():
            st.state_time[st.state] += makespan - st.state_since
            st.state_since = makespan
            st.update_queue_stats(makespan)

        station_results = {}
        for name, st in self.stations.items():
            station_results[name] = {
                "processed_count": st.processed_count,
                "failures": st.failures,
                "repair_time_s": st.repair_time_s,
                "processing_time_s": st.state_time["PROCESSING"],
                "blocked_time_s": st.state_time["BLOCKED"],
                "starved_time_s": st.state_time["STARVED"],
                "failed_time_s": st.state_time["FAILED"],
                "utilization": st.state_time["PROCESSING"] / makespan if makespan else 0.0,
                "blocked_fraction": st.state_time["BLOCKED"] / makespan if makespan else 0.0,
                "starved_fraction": st.state_time["STARVED"] / makespan if makespan else 0.0,
                "failed_fraction": st.state_time["FAILED"] / makespan if makespan else 0.0,
                "downtime_s": st.state_time["FAILED"],
                "downtime_fraction": st.state_time["FAILED"] / makespan if makespan else 0.0,
                "observed_mttr_s": (st.repair_time_s / st.failures) if st.failures else 0.0,
                "avg_queue_length": st.queue_area / makespan if makespan else 0.0,
                "max_queue_length": st.queue_max,
                "state_fraction_sum": sum(st.state_time.values()) / makespan if makespan else 0.0,
            }

        product_results = []
        lead_decomposition_errors = []
        fulfillment_decomposition_errors = []
        for p in self.products:
            if p.completed_at is None or p.released_at is None:
                factory_lead = None
                fulfillment = None
            else:
                factory_lead = p.completed_at - p.released_at
                fulfillment = p.completed_at - p.order_created_at

            # In-factory lead time begins at SOURCE_ADMIT, so external backlog
            # waiting is deliberately excluded from this decomposition.
            accounted_factory_lead = p.waiting_time + p.service_time + p.failure_delay + p.blocked_delay
            accounted_fulfillment = p.source_waiting_time + accounted_factory_lead
            lead_error = (factory_lead - accounted_factory_lead) if factory_lead is not None else None
            fulfillment_error = (fulfillment - accounted_fulfillment) if fulfillment is not None else None
            if lead_error is not None:
                lead_decomposition_errors.append(abs(lead_error))
            if fulfillment_error is not None:
                fulfillment_decomposition_errors.append(abs(fulfillment_error))

            product_results.append({
                "pid": p.pid,
                "ptype": p.ptype,
                "order_created_at_s": p.order_created_at,
                "released_at_s": p.released_at,
                "completed_at_s": p.completed_at,
                # Backward-compatible name: lead_time_s now means in-factory lead time.
                "lead_time_s": factory_lead,
                "factory_lead_time_s": factory_lead,
                "order_fulfillment_time_s": fulfillment,
                "source_waiting_time_s": p.source_waiting_time,
                "station_waiting_time_s": p.waiting_time,
                "waiting_time_s": p.waiting_time,
                "total_waiting_time_s": p.source_waiting_time + p.waiting_time,
                "service_time_s": p.service_time,
                "failure_delay_s": p.failure_delay,
                "blocked_delay_s": p.blocked_delay,
                "accounted_lead_time_s": accounted_factory_lead,
                "accounted_order_fulfillment_time_s": accounted_fulfillment,
                "lead_time_decomposition_error_s": lead_error,
                "order_fulfillment_decomposition_error_s": fulfillment_error,
            })

        product_summary = {}
        for ptype in self.config.routes:
            rows = [r for r in product_results if r["ptype"] == ptype]
            if not rows:
                continue
            lead = np.asarray([r["factory_lead_time_s"] for r in rows], dtype=float)
            fulfillment = np.asarray([r["order_fulfillment_time_s"] for r in rows], dtype=float)
            station_wait = np.asarray([r["station_waiting_time_s"] for r in rows], dtype=float)
            source_wait = np.asarray([r["source_waiting_time_s"] for r in rows], dtype=float)
            product_summary[ptype] = {
                "products": len(rows),
                "mean_lead_time_s": float(lead.mean()),
                "p50_lead_time_s": float(np.quantile(lead, 0.50)),
                "p95_lead_time_s": float(np.quantile(lead, 0.95)),
                "mean_order_fulfillment_time_s": float(fulfillment.mean()),
                "p50_order_fulfillment_time_s": float(np.quantile(fulfillment, 0.50)),
                "p95_order_fulfillment_time_s": float(np.quantile(fulfillment, 0.95)),
                "mean_source_waiting_time_s": float(source_wait.mean()),
                "mean_waiting_time_s": float(station_wait.mean()),
                "mean_service_time_s": float(np.mean([r["service_time_s"] for r in rows])),
                "mean_failure_delay_s": float(np.mean([r["failure_delay_s"] for r in rows])),
                "mean_blocked_delay_s": float(np.mean([r["blocked_delay_s"] for r in rows])),
            }

        return RunResult(
            makespan_s=makespan,
            completed=total,
            throughput_per_h=(total / (makespan / 3600.0)) if makespan else 0.0,
            product_results=product_results,
            product_summary=product_summary,
            station_results=station_results,
            event_log=self.event_log,
            total_failures=sum(st.failures for st in self.stations.values()),
            max_lead_time_decomposition_error_s=max(lead_decomposition_errors) if lead_decomposition_errors else 0.0,
            max_order_fulfillment_decomposition_error_s=max(fulfillment_decomposition_errors) if fulfillment_decomposition_errors else 0.0,
        )


def run_once(order_counts: Dict[str,int], seed: int = 12345, config: Optional[ModelConfig] = None,
             log_events: bool = True) -> RunResult:
    return FactorySimulation(config or build_default_config(), order_counts, seed=seed, log_events=log_events).run()


def run_monte_carlo(order_counts: Dict[str,int], n_runs: int = 100, seed: int = 2026,
                    config: Optional[ModelConfig] = None) -> dict:
    cfg = config or build_default_config()
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n_runs)
    makespans = []
    throughputs = []
    failures = []
    bottlenecks = []
    run_rows = []

    for run_idx, child in enumerate(children, start=1):
        run_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        result = run_once(order_counts, run_seed, cfg, log_events=False)
        makespans.append(result.makespan_s)
        throughputs.append(result.throughput_per_h)
        failures.append(result.total_failures)
        # utilization bottleneck for a simple first-pass classification
        bottleneck = max(result.station_results, key=lambda s: result.station_results[s]["utilization"])
        bottlenecks.append(bottleneck)
        run_rows.append({
            "run": run_idx,
            "seed": run_seed,
            "makespan_s": result.makespan_s,
            "makespan_h": result.makespan_s / 3600.0,
            "throughput_per_h": result.throughput_per_h,
            "total_failures": result.total_failures,
            "bottleneck": bottleneck,
        })

    arr = np.asarray(makespans, dtype=float)
    from collections import Counter
    counts = Counter(bottlenecks)
    summary_table = [
        {"metric": "Mean makespan (h)", "value": float(arr.mean() / 3600.0)},
        {"metric": "Median makespan (h)", "value": float(np.median(arr) / 3600.0)},
        {"metric": "P05 makespan (h)", "value": float(np.quantile(arr, 0.05) / 3600.0)},
        {"metric": "P95 makespan (h)", "value": float(np.quantile(arr, 0.95) / 3600.0)},
        {"metric": "Mean throughput (products/h)", "value": float(np.mean(throughputs))},
        {"metric": "Mean failures", "value": float(np.mean(failures))},
        {"metric": "P(at least one failure)", "value": float(np.mean(np.asarray(failures) > 0))},
        {"metric": "P(zero failures)", "value": float(np.mean(np.asarray(failures) == 0))},
    ]
    return {
        "n_runs": n_runs,
        "mean_makespan_s": float(arr.mean()),
        "median_makespan_s": float(np.median(arr)),
        "p05_makespan_s": float(np.quantile(arr, 0.05)),
        "p95_makespan_s": float(np.quantile(arr, 0.95)),
        "mean_throughput_per_h": float(np.mean(throughputs)),
        "mean_failures": float(np.mean(failures)),
        "prob_at_least_one_failure": float(np.mean(np.asarray(failures) > 0)),
        "prob_zero_failures": float(np.mean(np.asarray(failures) == 0)),
        "bottleneck_probability": {k: v / n_runs for k,v in sorted(counts.items())},
        "makespans_s": makespans,
        "run_results": run_rows,
        "summary_table": summary_table,
    }

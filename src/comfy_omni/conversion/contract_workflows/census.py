"""Pure census engine for the auto-contract scanner (P4b).

The census is the ``enforce``-disabled half of ``build_native_export_plan``
(design v2, section 8, P4b): it *observes* a native source checkpoint and
records what is there, without comparing anything against a pinned contract.
Observations only become decisions one level up, in
:mod:`comfy_omni.contract_auto.matcher` (three-level matching) and
:mod:`comfy_omni.contract_auto.generator` (draft + fail-closed diff).

Input boundary (design v2, section 7, review #12): a scan input is exactly
one of

* a single ``.safetensors`` file,
* an explicit list of ``.safetensors`` shards, or
* a directory constrained by ``model.safetensors.index.json``.

In the index mode the engine verifies that the index covers the observed
tensor set completely, that the directory holds no extra shard, that no
tensor name appears twice, and that the shard enumeration order is stable
(sorted by file name; :class:`comfy_omni.native_export.SafeTensorSources`
additionally rejects linked paths and duplicate cross-shard names).  The
component classification always runs over the *merged* name set of the whole
logical artifact, never per shard.

Reading is orchestrated through :class:`SafeTensorSources` (read-only reuse:
per-file sha256/size, header parsing via :mod:`comfy_omni.inspection`, triplet
discovery via :func:`comfy_omni.convrot.discover_convrot_groups`).  The
descriptor-level core :func:`census_tensors` is shared by the file path and
by callers that already hold descriptors (the P4b equivalence tests drive it
with exact-template-shape descriptor sets that would be tens of GiB as
files).

Layering: this package depends on ``inspection`` / ``convrot`` /
``native_export`` (read-only reuse) / ``oracle`` / ``h3.contracts``; nothing
below depends on it back.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from comfy_omni.artifacts import fileops
from comfy_omni.artifacts.sources import INDEX_NAME, SafeTensorSources
from comfy_omni.contracts.models import STORAGE_BF16_PLAIN, STORAGE_INT8_CONVROT, ContractError
from comfy_omni.conversion.contract_workflows.convrot import (
    COMFY_MARKER_SUFFIX,
    SCALE_SUFFIX,
    ConvRotError,
    ConvRotGroup,
    discover_convrot_groups,
)
from comfy_omni.domain.checkpoints import TensorDescriptor, classify_component


def schema_sha256(descriptors: Sequence[TensorDescriptor]) -> str:
    """Canonical sorted name/dtype/shape digest used by legacy enforced pins."""

    payload = [
        {"name": item.name, "dtype": item.dtype, "shape": list(item.shape)}
        for item in sorted(descriptors, key=lambda value: value.name)
    ]
    return hashlib.sha256(fileops.canonical_json(payload)).hexdigest()


#: Persisted schema identifier remains stable across the package rename.
CENSUS_SCHEMA = "h3_forge.contract_auto.census/v1"

#: Input modes (design v2, section 7 review #12 boundary).
INPUT_SINGLE_FILE = "single-file"
INPUT_EXPLICIT_SHARDS = "explicit-shards"
INPUT_INDEX_SHARDS = "index-shards"
#: Descriptor-level core entry (no file backing); used by the file paths'
#: shared core and by in-memory callers.
INPUT_DESCRIPTORS = "descriptors"


class ContractScanError(ContractError):
    """Fail-closed scan/draft error carrying structured evidence.

    ``evidence`` carries the machine-readable proof (census summary, census diff, stage tag)
    required for unknown sources to fail closed with evidence.
    """

    def __init__(self, message: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.evidence: dict[str, Any] = dict(evidence or {})


@dataclass(frozen=True)
class FileRecord:
    """One scanned source file: path, size, and whole-file sha256."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CensusReport:
    """Pure observation of one logical source artifact (no enforcement)."""

    files: tuple[FileRecord, ...]
    input_mode: str
    storage_kind: str
    tensor_count: int
    marker_count: int
    convrot_group_count: int
    #: ``group_size -> group count`` census (string keys for JSON stability).
    convrot_group_size_census: Mapping[str, int]
    #: ``"{rows}x{columns}" -> group count`` census of the scale tensors.
    scale_shape_census: Mapping[str, int]
    #: ``prefix -> (rows, columns, group_size)`` of every observed group.
    convrot_group_shapes: Mapping[str, tuple[int, int, int]]
    dtype_stats: Mapping[str, int]
    observed_schema_sha256: str
    component_hint: str
    component_evidence: tuple[str, ...]
    #: Full merged ``__metadata__`` of the artifact (observational).
    metadata: Mapping[str, str] = field(default_factory=dict, repr=False)
    #: Observed descriptors and discovered groups (not part of the JSON view).
    descriptors: tuple[TensorDescriptor, ...] = field(default=(), repr=False, compare=False)
    groups: tuple[ConvRotGroup, ...] = field(default=(), repr=False, compare=False)

    def census_summary(self) -> dict[str, Any]:
        """Compact census digest reused inside every evidence payload."""

        return {
            "input_mode": self.input_mode,
            "storage_kind": self.storage_kind,
            "tensor_count": self.tensor_count,
            "marker_count": self.marker_count,
            "convrot_group_count": self.convrot_group_count,
            "convrot_group_size_census": dict(self.convrot_group_size_census),
            "scale_shape_census": dict(self.scale_shape_census),
            "dtype_stats": dict(self.dtype_stats),
            "observed_schema_sha256": self.observed_schema_sha256,
            "file_count": len(self.files),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CENSUS_SCHEMA,
            "input_mode": self.input_mode,
            "files": [{"path": item.path, "size": item.size, "sha256": item.sha256} for item in self.files],
            "storage_kind": self.storage_kind,
            "tensor_count": self.tensor_count,
            "marker_count": self.marker_count,
            "convrot_group_count": self.convrot_group_count,
            "convrot_group_size_census": dict(sorted(self.convrot_group_size_census.items())),
            "scale_shape_census": dict(sorted(self.scale_shape_census.items())),
            "convrot_group_shapes": {
                prefix: list(shape) for prefix, shape in sorted(self.convrot_group_shapes.items())
            },
            "dtype_stats": dict(sorted(self.dtype_stats.items())),
            "observed_schema_sha256": self.observed_schema_sha256,
            "component_hint": self.component_hint,
            "component_evidence": list(self.component_evidence),
        }


def _observed_group_size_overrides(
    marker_payloads: Mapping[str, bytes],
) -> tuple[dict[str, int], dict[str, str]]:
    """Lenient *per-marker* pre-read of each marker's declared group size.

    The census must *observe* the group size a source declares instead of
    assuming 256 (the full Ref2VA source pins 64 for its 50 AdaLN groups).
    The strict decoder (:func:`comfy_omni.convrot.parse_convrot_marker`, called
    inside :func:`discover_convrot_groups`) re-validates every marker against
    the override derived here -- exact key set, format, ``convrot`` flag and
    group size.

    Per-marker failure handling (QA P4b finding 3): a marker whose payload
    cannot be pre-read is recorded individually with its reason and simply
    left without an override; the overrides of every *other* marker stay
    intact.  (The previous any-failure ``return {}`` wiped the whole
    override table, so one corrupt marker made all its group-size-64
    neighbors fail strict validation against the default 256 and the error
    misreported legitimate markers.)  Pre-read failures are handed to the
    strict decoder unchanged; :func:`census_tensors` attributes its error to
    exactly the failed markers.

    Returns ``(overrides, failures)`` where ``overrides`` maps group prefix
    to declared size and ``failures`` maps marker tensor name to reason.
    """

    overrides: dict[str, int] = {}
    failures: dict[str, str] = {}
    for name, raw in marker_payloads.items():
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures[name] = f"payload is not valid JSON: {exc}"
            continue
        if not isinstance(payload, dict):
            failures[name] = f"payload is {type(payload).__name__}, not a JSON object"
            continue
        group_size = payload.get("convrot_groupsize")
        if type(group_size) is not int:
            failures[name] = "payload declares no integer convrot_groupsize"
            continue
        overrides[name[: -len(COMFY_MARKER_SUFFIX)]] = group_size
    return overrides, failures


def _unsupported_marker_declarations(
    marker_payloads: Mapping[str, bytes],
) -> tuple[dict[str, int], tuple[str, ...]] | None:
    """Return bounded evidence for valid non-ConvRot marker declarations.

    Invalid or structurally ambiguous payloads return ``None`` so the existing
    strict ConvRot decoder remains their fail-closed authority. A syntactically
    valid declaration enters ConvRot discovery only when it explicitly carries
    ``format=int8_tensorwise`` and ``convrot=true``.
    """

    census: dict[str, int] = {}
    unsupported: list[str] = []
    for name, raw in sorted(marker_payloads.items()):
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("format"), str):
            return None
        marker_format = payload["format"]
        if marker_format == "int8_tensorwise" and payload.get("convrot") is True:
            continue
        declaration = "int8_tensorwise/non-convrot" if marker_format == "int8_tensorwise" else marker_format
        census[declaration] = census.get(declaration, 0) + 1
        unsupported.append(name)
    if not unsupported:
        return None
    return dict(sorted(census.items())), tuple(unsupported)


def _marker_free_storage_check(descriptors: Sequence[TensorDescriptor]) -> None:
    """A marker-free source cannot carry quantization leftovers (fail-closed)."""

    orphan_scales = sorted(descriptor.name for descriptor in descriptors if descriptor.name.endswith(SCALE_SUFFIX))
    if orphan_scales:
        raise ContractScanError(
            f"marker-free source carries weight_scale tensors without comfy_quant markers: {orphan_scales[:4]}",
            evidence={"stage": "storage-observation", "orphan_scales": orphan_scales[:8]},
        )
    int8_tensors = sorted(descriptor.name for descriptor in descriptors if descriptor.dtype == "I8")
    if int8_tensors:
        raise ContractScanError(
            f"marker-free source carries INT8 tensors without comfy_quant markers: {int8_tensors[:4]}",
            evidence={"stage": "storage-observation", "int8_tensors": int8_tensors[:8]},
        )


def _discover_storage(
    ordered: tuple[TensorDescriptor, ...], marker_payloads: Mapping[str, bytes]
) -> tuple[tuple[ConvRotGroup, ...], str, tuple[str, ...]]:
    marker_names = tuple(item.name for item in ordered if item.name.endswith(COMFY_MARKER_SUFFIX))
    if not marker_names:
        _marker_free_storage_check(ordered)
        return (), STORAGE_BF16_PLAIN, ()
    payloads = {name: marker_payloads[name] for name in marker_names if name in marker_payloads}
    declarations = _unsupported_marker_declarations(payloads) if len(payloads) == len(marker_names) else None
    if declarations is not None:
        declaration_census, unsupported = declarations
        summary = ", ".join(f"{name}={count}" for name, count in declaration_census.items())
        raise ContractScanError(
            f"unsupported comfy_quant storage declarations: {summary}; contract workflows support "
            "only strict int8 ConvRot markers and marker-free BF16 sources",
            evidence={
                "stage": "storage-observation",
                "reason_code": "unsupported-comfy-quant-storage",
                "marker_count": len(marker_names),
                "marker_declaration_census": declaration_census,
                "sample_unsupported_markers": list(unsupported[:8]),
            },
        )
    overrides, failures = _observed_group_size_overrides(payloads)
    try:
        groups = discover_convrot_groups(
            ordered,
            payloads,
            expected_groups=len(marker_names),
            expected_group_sizes=overrides,
        )
    except ConvRotError as exc:
        detail = str(exc)
        if failures:
            named = "; ".join(f"{name}: {reason}" for name, reason in sorted(failures.items()))
            detail = f"comfy_quant marker pre-read failed: {named} (strict decoder: {detail})"
        raise ContractScanError(
            detail,
            evidence={
                "stage": "convrot-discovery",
                "marker_count": len(marker_names),
                "marker_pre_read_failures": dict(sorted(failures.items())),
                "observed_group_size_override_count": len(overrides),
            },
        ) from exc
    return groups, STORAGE_INT8_CONVROT, marker_names


def _group_observations(
    groups: Sequence[ConvRotGroup],
) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[int, int, int]]]:
    group_sizes: dict[str, int] = {}
    scale_shapes: dict[str, int] = {}
    group_shapes: dict[str, tuple[int, int, int]] = {}
    for group in groups:
        size_key = str(group.group_size)
        scale_key = f"{group.scale.shape[0]}x{group.scale.shape[1]}"
        group_sizes[size_key] = group_sizes.get(size_key, 0) + 1
        scale_shapes[scale_key] = scale_shapes.get(scale_key, 0) + 1
        rows, columns = group.weight.shape
        group_shapes[group.prefix] = (rows, columns, group.group_size)
    return group_sizes, scale_shapes, group_shapes


def _component_observation(names: tuple[str, ...], metadata: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    try:
        component, evidence = classify_component(names, metadata)
    except ValueError as exc:
        raise ContractScanError(
            f"component classification is contradictory: {exc}",
            evidence={"stage": "component-classification"},
        ) from exc
    return component, tuple(evidence)


def census_tensors(
    descriptors: Sequence[TensorDescriptor],
    marker_payloads: Mapping[str, bytes],
    *,
    files: Sequence[FileRecord] = (),
    input_mode: str = INPUT_DESCRIPTORS,
    metadata: Mapping[str, str] | None = None,
) -> CensusReport:
    """Observe one descriptor set without consulting any pinned template."""

    ordered = tuple(sorted(descriptors, key=lambda descriptor: descriptor.name))
    if not ordered:
        raise ContractScanError("source contains no tensors", evidence={"stage": "census", "tensor_count": 0})
    groups, storage_kind, marker_names = _discover_storage(ordered, marker_payloads)
    group_sizes, scale_shapes, group_shapes = _group_observations(groups)
    dtype_stats: dict[str, int] = {}
    for descriptor in ordered:
        dtype_stats[descriptor.dtype] = dtype_stats.get(descriptor.dtype, 0) + 1
    merged_metadata = dict(metadata or {})
    component, component_evidence = _component_observation(
        tuple(descriptor.name for descriptor in ordered), merged_metadata
    )
    return CensusReport(
        files=tuple(files),
        input_mode=input_mode,
        storage_kind=storage_kind,
        tensor_count=len(ordered),
        marker_count=len(marker_names),
        convrot_group_count=len(groups),
        convrot_group_size_census=group_sizes,
        scale_shape_census=scale_shapes,
        convrot_group_shapes=group_shapes,
        dtype_stats=dtype_stats,
        observed_schema_sha256=schema_sha256(ordered),
        component_hint=component,
        component_evidence=component_evidence,
        metadata=merged_metadata,
        descriptors=ordered,
        groups=groups,
    )


def _load_shard_index(root: Path) -> dict[str, str]:
    """Parse ``model.safetensors.index.json`` strictly into a weight map."""

    index_path = root / INDEX_NAME
    try:
        payload, _ = fileops.read_file_pinned(index_path)
        document = fileops.parse_json_strict(payload)
    except (OSError, fileops.FsopsJsonError) as exc:
        raise ContractScanError(
            f"{index_path}: unreadable or invalid shard index: {exc}",
            evidence={"stage": "input-boundary", "index": str(index_path)},
        ) from exc
    if not isinstance(document, dict) or not isinstance(document.get("weight_map"), dict):
        raise ContractScanError(
            f"{index_path}: shard index must be an object with a weight_map object",
            evidence={"stage": "input-boundary", "index": str(index_path)},
        )
    weight_map = {
        str(name): str(shard)
        for name, shard in document["weight_map"].items()  # type: ignore[union-attr]
    }
    if not weight_map:
        raise ContractScanError(
            f"{index_path}: shard index weight_map is empty",
            evidence={"stage": "input-boundary", "index": str(index_path)},
        )
    return weight_map


def _safetensors_in_tree(root: Path) -> list[str]:
    """Every ``.safetensors`` file under ``root`` (root-relative, sorted).

    ``os.walk`` never follows directory links, so a linked subtree is never
    silently included; a declared linked shard is rejected by
    :class:`SafeTensorSources` and an undeclared one by the extra-shard
    check below.
    """

    found: list[str] = []
    for base, directories, files in os.walk(root):
        directories.sort()
        for name in sorted(files):
            if name.lower().endswith(".safetensors"):
                found.append(str((Path(base) / name).relative_to(root)))
    return found


def _resolve_single(path: Path) -> tuple[str, tuple[Path, ...], dict[str, str] | None]:
    """Resolve one scan input into (mode, shard paths, optional weight map)."""

    if path.is_file():
        if path.suffix.lower() != ".safetensors":
            raise ContractScanError(
                f"{path}: expected a .safetensors file, a shard list, or a directory holding {INDEX_NAME}",
                evidence={"stage": "input-boundary", "path": str(path)},
            )
        return INPUT_SINGLE_FILE, (path,), None
    if path.is_dir():
        if not (path / INDEX_NAME).is_file():
            raise ContractScanError(
                f"{path}: directory input requires {INDEX_NAME} (a bare directory is not a "
                "defined scan boundary: single file, explicit shard list, or index-constrained "
                "shard set)",
                evidence={
                    "stage": "input-boundary",
                    "observed_entries": sorted(item.name for item in path.iterdir())[:16],
                },
            )
        weight_map = _load_shard_index(path)
        declared = sorted(set(weight_map.values()))
        for shard in declared:
            candidate = Path(shard)
            if candidate.name != shard or candidate.is_absolute() or shard in {".", ".."}:
                raise ContractScanError(
                    f"{path}: shard index must reference plain shard file names, got {shard!r}",
                    evidence={"stage": "input-boundary", "shard": shard},
                )
        shard_paths = tuple(path / shard for shard in declared)
        missing = [shard for shard, shard_path in zip(declared, shard_paths, strict=True) if not shard_path.is_file()]
        if missing:
            raise ContractScanError(
                f"{path}: shard index declares shards that are missing: {missing[:4]}",
                evidence={"stage": "input-boundary", "missing_shards": missing[:8]},
            )
        present = _safetensors_in_tree(path)
        extra = sorted(set(present) - set(declared))
        if extra:
            raise ContractScanError(
                f"{path}: directory holds safetensors files outside the index-declared shard set: {extra[:4]}",
                evidence={"stage": "input-boundary", "extra_shards": extra[:8], "declared": declared},
            )
        return INPUT_INDEX_SHARDS, shard_paths, weight_map
    raise ContractScanError(
        f"{path}: scan input does not exist",
        evidence={"stage": "input-boundary", "path": str(path)},
    )


class CensusEngine:
    """Orchestrates the pure census over the three legal input boundaries."""

    def scan(self, path: Path | str) -> CensusReport:
        """Scan one input: a single file or an index-constrained directory."""

        mode, paths, weight_map = _resolve_single(Path(path))
        return self.scan_shards(paths, input_mode=mode, weight_map=weight_map)

    def scan_paths(self, paths: Sequence[Path | str]) -> CensusReport:
        """Scan an explicit path list (one path may be an index directory)."""

        items = tuple(Path(item) for item in paths)
        if not items:
            raise ContractScanError("scan input must not be empty", evidence={"stage": "input-boundary"})
        if len(items) == 1:
            return self.scan(items[0])
        for item in items:
            if item.is_dir():
                raise ContractScanError(
                    f"{item}: a directory input must be the sole scan path (index-constrained shard set)",
                    evidence={"stage": "input-boundary", "path": str(item)},
                )
        return self.scan_shards(items)

    def scan_shards(
        self,
        paths: Sequence[Path | str],
        *,
        input_mode: str = INPUT_EXPLICIT_SHARDS,
        weight_map: Mapping[str, str] | None = None,
    ) -> CensusReport:
        """Scan an explicit shard list (or pre-resolved index shard set)."""

        items = tuple(Path(item) for item in paths)
        if not items:
            raise ContractScanError("explicit shard list must not be empty", evidence={"stage": "input-boundary"})
        for item in items:
            if item.suffix.lower() != ".safetensors" or not item.is_file():
                raise ContractScanError(
                    f"{item}: every scan shard must be an existing .safetensors file",
                    evidence={"stage": "input-boundary", "path": str(item)},
                )
        if len(set(items)) != len(items):
            raise ContractScanError("duplicate shard path in scan input", evidence={"stage": "input-boundary"})
        try:
            with SafeTensorSources(items) as sources:
                return self._census_from_sources(sources, input_mode=input_mode, weight_map=weight_map)
        except ContractScanError:
            raise
        except (ContractError, fileops.FsopsError, OSError) as exc:
            raise ContractScanError(str(exc), evidence={"stage": "source-open", "shard_count": len(items)}) from exc

    def _census_from_sources(
        self,
        sources: SafeTensorSources,
        *,
        input_mode: str,
        weight_map: Mapping[str, str] | None,
    ) -> CensusReport:
        if weight_map is not None:
            _check_index_coverage(weight_map, sources)
        descriptors = tuple(located.descriptor for located in sources.tensors.values())
        marker_payloads = {
            name: sources.read_raw(located)
            for name, located in sources.tensors.items()
            if name.endswith(COMFY_MARKER_SUFFIX)
        }
        files = tuple(
            FileRecord(path=str(path), size=size, sha256=digest)
            for path, size, digest in zip(sources.paths, sources.sizes, sources.hashes, strict=True)
        )
        report = census_tensors(
            descriptors,
            marker_payloads,
            files=files,
            input_mode=input_mode,
            metadata=_merged_metadata(sources),
        )
        sources.verify_unchanged()
        return report


def _merged_metadata(sources: SafeTensorSources) -> dict[str, str]:
    merged: dict[str, str] = {}
    for file_metadata in sources.metadata:
        for key, value in file_metadata.items():
            if key in merged and merged[key] != value:
                raise ContractScanError(
                    f"contradictory __metadata__ across shards: {key!r}",
                    evidence={"stage": "metadata", "key": key},
                )
            merged[key] = value
    return merged


def _check_index_coverage(weight_map: Mapping[str, str], sources: SafeTensorSources) -> None:
    """Index mode guarantees: full coverage, correct placement, no extras."""

    declared = dict(weight_map)
    observed = set(sources.tensors)
    missing_from_shards = sorted(set(declared) - observed)
    undeclared_in_shards = sorted(observed - set(declared))
    if missing_from_shards or undeclared_in_shards:
        raise ContractScanError(
            f"shard index does not cover the observed tensor set: "
            f"missing_from_shards={missing_from_shards[:4]} "
            f"undeclared_in_shards={undeclared_in_shards[:4]}",
            evidence={
                "stage": "shard-index",
                "missing_from_shards": missing_from_shards[:8],
                "undeclared_in_shards": undeclared_in_shards[:8],
            },
        )
    misplaced = sorted(
        name
        for name, shard in declared.items()
        if Path(shard).name != sources.paths[sources.tensors[name].source_index].name
    )
    if misplaced:
        raise ContractScanError(
            f"shard placement disagrees with the index weight_map: {misplaced[:4]}",
            evidence={"stage": "shard-index", "misplaced_tensors": misplaced[:8]},
        )

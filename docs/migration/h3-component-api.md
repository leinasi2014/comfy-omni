# H3 component directory and API bootstrap

The component directory restores three existing H3 endpoints:

- `GET /v1/h3-forge/components`
- `GET /v1/h3-forge/components/{kind}`
- `POST /v1/h3-forge/components/scan`

The directory reports candidate identifiers and locations. Discovery does not
validate a model's tensors or claim that it can be loaded. It does not select
an active model, switch components, invoke LoRA/tools or run generation.

## Configuration and responses

`H3_FORGE_COMPONENT_ROOTS` retains the legacy forms: semicolon-separated
`zone=path` pairs, including repeated zones, or a JSON object mapping zones
to a path or list of paths. The zones are `comfy`, `official` and `servable`.
Comfy roots discover ordinary files beneath the five existing type directories;
official/servable roots discover nonempty component directories at the existing
package depths. No checkpoint header, payload or hash is read by this catalog.

The first request constructs and scans the catalog. Later GET requests reuse
that snapshot; POST scan refreshes the in-memory index. Failed scans preserve
the previous complete index. An unconfigured or failed initial build is not
cached, so a later request can pick up corrected configuration. A successfully
constructed catalog retains its configured roots.

Entry fields and `h3_forge.components.list/v1`,
`h3_forge.components.scan/v1` and `h3_forge.error/v1` remain unchanged.
Official entries stay visible but locked. `selection` and
`selection_candidates` describe request candidate IDs, not active/default
state. Code-defined schedules keep a null zone and empty path; discovered
entries do not invent a contract digest. Directory discovery retains the
legacy filesystem semantics rather than becoming an artifact verifier.

Missing configuration returns 503 `components_not_configured`; malformed
configuration returns 500 `components_misconfigured`; catalog failures return
500 `components_scan_failed`. Configuration is checked before an unknown kind
returns 404 `H3_COMPONENT_KIND_UNKNOWN`. No global exception handler replaces
the host's unrelated HTTP error behavior.

## Runtime and API phases

`comfy_omni.plugin:register` still delegates to one coordinator. Runtime
contributions remain lazy strings. Both preserved H3 architecture keys now
resolve to the existing package-verifying runtime dispatcher, so the dense
wire key no longer points at an absent module. This does not add a second
dense implementation or bypass its supported-package checks.

Only a root process with a resident host arms the single deferred API import
helper. Worker registration imports no API route or FastAPI module and scans
no directories. Forked workers discard inherited pending API hooks and reset
their process-local registration locks. The helper waits for the host API
module body to finish and checks that the completed object remains resident;
the coordinator then resolves declarative API contributions. The component
mount prefers the host's module-level router, which its app includes later,
and supports the existing app-shaped fallback.

The runtime and API phases share one coordinator lock. Deferred imports and
callback failures remain retryable; a successful runtime registration is not
repeated after an API failure. Existing target marking prevents duplicate
component mounts. LoRA/tools can later contribute to the same API phase;
there are no placeholder routes or recursive sub-plugin registrations.

## Source attribution and distribution

The Apache-2.0 behavior and characterization inputs below are from
`h3-forge@e9cb011d00b028c149db3978de246c54f6e34acc`.

| Source | Exact Git blob | New owner |
| --- | --- | --- |
| `src/h3_forge/plugin.py` | `304a776bf4daf1f7a28b1bc6192d320da30421fd` | phased integration bootstrap |
| `src/h3_forge/_import_hook.py` | `935e30d22558a3e9b9065421423e36cd101e35df` | single integration deferred-import helper |
| `src/h3_forge/component_catalog/catalog.py` | `322dd5b5e37722d82675d9d6c547901b296b759f` | domain component values and runtime catalog |
| `src/h3_forge/component_catalog/api.py` | `03597ba2952a6d7933fa174cdfe5b1073b234d9d` | thin component routes and host mounting |
| `tests/test_import_hook.py` | `d07d30caa47426614ba2726944ed37c120d0fa62` | real import completion/race characterization |
| `tests/test_component_catalog.py` | `77a6b54d1e72c1de9e9000d671c42ab5a4ebc40c` | directory and snapshot characterization |
| `tests/test_component_api.py` | `434f944bf5b581a7b8571595012285de8a85adb8` | HTTP wire/error characterization |

The actual host mounting shape comes from Apache-2.0 `vllm-omni` at
`17285c2f55a41bf15772676121814d59a60ace35`,
`vllm_omni/entrypoints/openai/api_server.py` blob
`57adaad08ff28160831f503e639425f250bf4313`. The host implementation remains an
external import and is not copied.

The bounded derived implementation and tests are distributed under this
repository's Apache-2.0 license with h3-forge contributor attribution. Legacy
public facades, full plugin chains, generation-request schemas and model
payloads are not distributed by this change. Acceptance decisions remain in
[Issue #10](https://github.com/leinasi2014/comfy-omni/issues/10#issuecomment-5550697907)
and the [H3 first-release goal](https://github.com/leinasi2014/comfy-omni/issues/4).

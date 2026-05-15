# sco-ldc API Reference

**API version:** 3.0
**Document version:** 2026-05-15
**Production base URL:** `https://sco-ldc.com`
**Fallback URL:** `https://sco-ldc.onrender.com`

This document describes the HTTP/JSON API of sco-ldc, a web service that returns quadratic limb-darkening coefficients (u1, u2) by trilinear interpolation of three published Claret coefficient tables, and that resolves exoplanet names to host-star stellar parameters from the NASA Exoplanet Archive (NEA) and ExoFOP-TESS.

This reference is language-neutral and intended for any HTTP client (Java, JavaScript, Python, curl, etc.). All requests are HTTPS GET. All response bodies are JSON. The API requires no authentication and accepts unauthenticated cross-origin requests (CORS allow-origin is `\*`).

## 1\. Conventions

### Base URL

All endpoints are served from `https://sco-ldc.com`. A fallback URL at `https://sco-ldc.onrender.com` serves the same application and can be used if the primary URL is unreachable for any reason. HTTPS is required; the service does not accept HTTP.

### Authentication

None. sco-ldc's API is open and does not require API keys, tokens, registration, or any other credential. Any HTTP client can call any endpoint at any time.

### Methods

All endpoints accept `GET` requests only. There is no `POST`, `PUT`, or `DELETE`. This reflects the read-only nature of the service: sco-ldc retrieves data and computes from it but does not store or modify state on behalf of callers.

### Parameters

All parameters are passed as URL query string parameters. There is no JSON request body. Parameter names are case-sensitive. Numeric parameters can be passed as integers (5778) or as decimals (5778.0); both are accepted. Filter codes are strings and case-sensitive.

### Response format

All successful responses are JSON with HTTP status 200. The response Content-Type is `application/json`. All field names use snake\_case (e.g., `filter\_code`, not `filterCode`). Numeric fields are always JSON numbers (not strings); string fields are always JSON strings (not numbers).

### URL encoding

Filter codes containing special characters (the SDSS codes use an asterisk; planet names typically contain spaces) must be URL-encoded in the request. Most HTTP client libraries handle this automatically when parameters are passed as a structured dictionary or map rather than concatenated into the URL string. If constructing URLs manually, ensure that asterisks become `%2A` and spaces become `%20` or `+`.

## 2\. Stability and versioning

The API described here is treated as a stable contract. Once a caller integrates with sco-ldc, the following are guaranteed:

* Existing parameter names will not be renamed
* Existing response field names will not be renamed
* Existing filter codes will not be reassigned to different filters
* Existing semantics will be preserved (which model is the default, which endpoint serves which path, etc.)

Additions are not considered breaking and may happen without coordination:

* New endpoints may be added at any time
* New optional parameters may be added to existing endpoints
* New response fields may be added to existing responses (callers should not fail if they encounter unknown fields)

If breaking changes ever become necessary, they will be exposed through versioned URL paths (e.g., `/api/v2/compute`) rather than by modifying the existing endpoints. Callers can continue using the current endpoints on their own timeline and migrate when convenient.

## 3\. Endpoints overview

|Endpoint|Purpose|
|-|-|
|`GET /api/health`|Liveness probe and cache freshness diagnostic|
|`GET /api/filters`|Catalog of supported filters and their model coverage|
|`GET /api/resolve`|Resolve an exoplanet name or TOI to host-star Teff/logg/\[Fe/H]|
|`GET /api/compute`|Compute (u1, u2) given Teff, logg, \[Fe/H], filter, and model|

Typical integration flow:

1. **(Optional, at startup or once per session.)** Call `/api/filters` once to discover what filters and models are available, with their grid ranges. Cache the result locally.
2. **(Optional, per target.)** Call `/api/resolve?planet=<NAME>` to fetch host-star Teff/logg/\[Fe/H] from a planet or TOI name.
3. **(Per LDC requested.)** Call `/api/compute?teff=...\&logg=...\&feh=...\&filter=...\&model=...` to obtain u1 and u2.

## 4\. `GET /api/health`

Returns a status snapshot. Useful for liveness checks and for verifying that the underlying data sources are fresh before relying on a lookup.

**Required parameters:** none.
**Optional parameters:** none.

**Response (HTTP 200):**

```
{
  "status": "ok",
  "version": "3.0.0",
  "freshness": "ok",
  "tables": {
    "tableab.dat": 179645,
    "table5.dat": 574,
    "CBBQUADRATIC.txt": 7695
  },
  "filter\_count": 25,
  "nea\_cache":    { "count": 6287, "refreshed\_utc": "2026-05-15 17:00 UTC" },
  "exofop\_cache": { "count": 7934, "refreshed\_utc": "2026-05-15 17:00 UTC" }
}
```

**Field semantics:**

|Field|Type|Notes|
|-|-|-|
|`status`|string|Always `"ok"` if the response was generated at all|
|`version`|string|Server-side application version identifier|
|`freshness`|string|`"ok"` if both planet caches refreshed within 48 h; else `"stale"`|
|`tables`|object|Row counts per source file. Keys are file names; values are integers|
|`filter\_count`|integer|Number of distinct filters loaded (currently 25)|
|`nea\_cache.count`|integer|Rows currently held in the NEA planet cache|
|`nea\_cache.refreshed\_utc`|string|Last successful live refresh; `null` if never refreshed|
|`exofop\_cache.count`|integer|Rows in the ExoFOP TOI cache|
|`exofop\_cache.refreshed\_utc`|string|Same convention as NEA|

**Notes for callers:**

* The `freshness` field is the recommended signal to monitor. Callers do not need to act on `"stale"` programmatically — the LDC computation continues to work — but it may be useful for diagnostic surfaces.
* Cache row counts grow over time as NEA and ExoFOP add new entries. Exact numbers are not guaranteed to be stable; they are informational.

## 5\. `GET /api/filters`

Returns the catalog of filters available and, per filter, which stellar atmosphere models are populated along with their parameter grid ranges. Calling `/api/filters` is the authoritative way to discover what's currently supported; new filters or new model coverage can appear in future versions.

**Required parameters:** none.
**Optional parameters:** none.

**Response (HTTP 200):**

```
{
  "filters": \[
    {
      "code": "V",
      "name": "Johnson V",
      "category": "Johnson-Cousins",
      "source": "CB2011",
      "citation": "Claret \& Bloemen (2011, A\&A 529, A75)",
      "models": \[
        {
          "model": "ATLAS",
          "model\_key": "ATLAS",
          "teff\_min": 3500.0, "teff\_max": 50000.0,
          "logg\_min": 0.0,    "logg\_max": 5.0,
          "feh\_min":  -5.0,   "feh\_max":  1.0,
          "feh\_fixed": false,
          "n\_points": 7813
        },
        {
          "model": "PHOENIX",
          "model\_key": "PHOENIX",
          "teff\_min": 2000.0, "teff\_max": 9800.0,
          "logg\_min": 3.5,    "logg\_max": 5.0,
          "feh\_min":  0.0,    "feh\_max":  0.0,
          "feh\_fixed": true,
          "n\_points": 116
        }
      ]
    }
  ]
}
```

**Field semantics (per filter):**

|Field|Type|Notes|
|-|-|-|
|`code`|string|Filter code used in `/api/compute`. Case-sensitive|
|`name`|string|Human-readable filter name|
|`category`|string|Filter family: `Johnson-Cousins`, `Sloan/SDSS`, `Strömgren`, `Space-based`, `Other`|
|`source`|string|Internal source tag: `CB2011`, `C2018`, or `CMG2022`|
|`citation`|string|Source's bibliographic citation|
|`models`|array|List of model entries (typically 1 or 2 per filter)|

**Field semantics (per model entry):**

|Field|Type|Notes|
|-|-|-|
|`model`|string|Display name used in `/api/compute`'s `model` parameter|
|`model\_key`|string|Internal storage key; identical to `model` except for TESS (`PHOENIX-COND` displays as such but is stored as `PHOENIX`)|
|`teff\_min/max`|number|Grid extent in K. Inclusive bounds. Inputs outside fail|
|`logg\_min/max`|number|Grid extent in cgs dex|
|`feh\_min/max`|number|Grid extent in dex. For PHOENIX models, a single value at 0.0|
|`feh\_fixed`|boolean|`true` if the grid has only one \[Fe/H] value (solar metallicity only)|
|`n\_points`|integer|Total grid points populated for this filter+model combination|

**Notes for callers:**

* Filter codes are case-sensitive. In particular, lowercase letters distinguish Strömgren from uppercase Johnson: `"V"` is Johnson V; `"v"` is Strömgren v. SDSS codes use a trailing asterisk: `u\*`, `g\*`, `r\*`, `i\*`, `z\*`.
* The complete current list of codes is: `U`, `B`, `V`, `R`, `I`, `J`, `H`, `K`, `u`, `v`, `b`, `y`, `u\*`, `g\*`, `r\*`, `i\*`, `z\*`, `Kp`, `C`, `S1`, `S2`, `S3`, `S4`, `TESS`, `CBB`.
* The `model` parameter to `/api/compute` accepts the display name from the `model` field. Accepted values are `"ATLAS"`, `"PHOENIX"`, and `"PHOENIX-COND"`. Not every filter supports every model; the `models` array per filter is the authoritative list.
* When the `feh\_fixed` flag is `true`, passing a `feh` value other than 0.0 to `/api/compute` will be rejected. The default `feh=0.0` works correctly.

## 6\. `GET /api/resolve`

Looks up an exoplanet name or TESS Object of Interest (TOI) candidate identifier and returns host-star stellar parameters from the authoritative source (NEA for confirmed planets; ExoFOP-TESS for TOI candidates not in NEA).

**Required parameters:**

|Name|Type|Notes|
|-|-|-|
|`planet`|string|Planet name or TOI identifier. Case-insensitive on lookup|

**Optional parameters:** none.

**Examples of valid input:**

* `WASP-23 b`
* `wasp-23 b` (case-insensitive)
* `HD 189733 b`
* `TRAPPIST-1 e`
* `TOI-700.01`
* `TOI 700.01` (TOI prefix can use a space or hyphen)

**Response when found (HTTP 200):**

```
{
  "found": true,
  "planet": "WASP-23 b",
  "hostname": "WASP-23",
  "teff": 5150.0,
  "logg": 4.4,
  "feh":  -0.05,
  "source": "NEA",
  "citation": "DOI: 10.26133/NEA13"
}
```

**Response when not found (HTTP 200, `found=false`):**

```
{
  "found": false,
  "planet": "<input as supplied>",
  "reason": "not\_in\_nea",
  "suggestions": \["WASP-23 b", "WASP-3 b", "WASP-43 b"]
}
```

**Response on upstream error (HTTP 200, `found=false`):**

This is only returned in a degraded mode (the server cache is empty and a live upstream query also failed). In normal operation this shape is not returned.

```
{
  "found": false,
  "planet": "<input as supplied>",
  "reason": "error",
  "error": "NEA query timed out"
}
```

**Field semantics:**

|Field|Type|Notes|
|-|-|-|
|`found`|boolean|`true` if the planet was resolved; `false` otherwise|
|`planet`|string|Canonical planet name on success; raw input on failure|
|`hostname`|string|Host star designation (e.g. `"WASP-23"`)|
|`teff`|number or null|Effective temperature in K, or `null` if the source has no value|
|`logg`|number or null|Surface gravity log g in cgs dex, or `null`|
|`feh`|number or null|Metallicity \[Fe/H] in dex, or `null`|
|`source`|string|`"NEA"` or `"ExoFOP"`. Tells the caller which database served this|
|`citation`|string|Citation for the source database|
|`reason`|string|On failure: `"not\_in\_nea"`, `"not\_in\_exofop"`, or `"error"`|
|`suggestions`|array of strings|On `not\_in\_nea` failure: up to 3 close-match planet names from NEA|
|`error`|string|On `error` reason: a short human-readable explanation|

**Citation strings returned:**

* For NEA lookups: `"DOI: 10.26133/NEA13"`
* For ExoFOP lookups: `"ExoFOP-TESS, NExScI/Caltech-IPAC"`

**Notes for callers:**

* Any of `teff`, `logg`, or `feh` may be `null` even on a `found=true` response. This is faithful to the source — some NEA/ExoFOP entries genuinely lack values for individual parameters. Callers should handle `null` and either prompt the user to supply a value or skip the LDC computation.
* The `source` field tells the caller whether the values came from NEA's curated composite parameters table or from ExoFOP's TOI catalog. These pipelines can disagree by \~100 K on Teff or \~0.1 dex on \[Fe/H] for the same star. This is expected and not an error.
* Lookups are typically served from cache; expect sub-100 ms response.
* The HTTP status code is `200` even on `found=false`. The body's `found` field is the success indicator. Callers should not branch on HTTP status alone for the resolve endpoint.
* For inputs that look like TOI identifiers (e.g., `TOI-700.01`), the lookup queries NEA first because some confirmed planets are catalogued in NEA under their TOI designation. Only if NEA has no entry under that name does the lookup fall through to ExoFOP. Callers do not need to implement this routing logic; just pass the user's input string and inspect the `source` field of the response to know where the values came from.

## 7\. `GET /api/compute`

Computes quadratic limb-darkening coefficients (u1, u2) by trilinear interpolation of the Claret table corresponding to the requested filter and atmosphere model. This is the main computational endpoint.

**Required parameters:**

|Name|Type|Notes|
|-|-|-|
|`teff`|number|Effective temperature in K|
|`logg`|number|Surface gravity log g in cgs dex|
|`filter`|string|Filter code from the `/api/filters` catalog. Case-sensitive|

**Optional parameters:**

|Name|Type|Notes|
|-|-|-|
|`feh`|number|Metallicity \[Fe/H] in dex. Defaults to `0.0` if omitted|
|`model`|string|`"ATLAS"`, `"PHOENIX"`, or `"PHOENIX-COND"`. Defaults to `"ATLAS"` if omitted|

**Response on success (HTTP 200):**

```
{
  "u1": 0.5124,
  "u2": 0.1558,
  "filter\_code": "i\*",
  "filter\_name": "SDSS i'",
  "model": "ATLAS",
  "citation": "Claret \& Bloemen (2011, A\&A 529, A75)",
  "grid": {
    "teff\_bracket": \[5000.0, 5250.0],
    "logg\_bracket": \[4.0, 4.5],
    "feh\_bracket":  \[-0.5, 0.0],
    "fractions":    { "teff": 0.4, "logg": 0.8, "feh": 0.9 },
    "on\_grid":      false
  }
}
```

(The numeric values above are illustrative; actual values from the live API may differ.)

**Response on invalid input (HTTP 400):**

```
{
  "detail": "Invalid Input (Teff = 1500.0 K): The PHOENIX model does not support values of Teff below 2000 K."
}
```

**Field semantics:**

|Field|Type|Notes|
|-|-|-|
|`u1`|number|First quadratic LDC|
|`u2`|number|Second quadratic LDC|
|`filter\_code`|string|The filter code that was used (echoes the input)|
|`filter\_name`|string|Human-readable filter name|
|`model`|string|The model used (echoes/canonicalizes the input)|
|`citation`|string|Source citation for the underlying Claret table|
|`grid.teff\_bracket`|\[num, num]|The two grid Teff values bracketing the input|
|`grid.logg\_bracket`|\[num, num]|Same for logg|
|`grid.feh\_bracket`|\[num, num]|Same for \[Fe/H]|
|`grid.fractions.teff`|number|Interpolation fraction in \[0, 1] along the Teff axis|
|`grid.fractions.logg`|number|Same for logg|
|`grid.fractions.feh`|number|Same for \[Fe/H]|
|`grid.on\_grid`|boolean|`true` if the input falls exactly on a grid point (no interpolation)|

**Citation strings by source:**

* CB2011 filters: `"Claret \& Bloemen (2011, A\&A 529, A75)"`
* C2018 (TESS only): `"Claret (2018, A\&A 618, A20)"`
* CMG2022 (CBB only): `"Claret, Mullen \& Gary (2022, RNAAS 6, 169)"`

**Notes for callers:**

* For PHOENIX models in CB2011, the \[Fe/H] grid is degenerate (solar metallicity only). Passing `feh=0.0` (the default) is correct; passing other values is rejected.
* For TESS in the C2018 table, the \[Fe/H] grid is similarly degenerate.
* Inputs outside the grid range raise HTTP 400 with a human-readable `detail` message. The error message often includes a suggestion for an alternative model (e.g., "Use the PHOENIX model instead" when an ATLAS request is below 3500 K).
* `u1` and `u2` are returned as full-precision IEEE 754 doubles. The Claret published values are themselves to 4 decimal places; sco-ldc does not artificially round the interpolation result.
* The `grid` block is informational. Callers that don't need interpolation diagnostics can ignore it.

## 8\. Error responses

A summary of every error case the API can return:

|Scenario|HTTP status|Body shape|
|-|-|-|
|Healthy successful request|200|Endpoint-specific success shape|
|`/api/resolve` planet not found|200|`{ "found": false, ... }` (see `/api/resolve` section)|
|`/api/compute` invalid input (out of range)|400|`{ "detail": "..." }`|
|`/api/compute` unknown filter or model|400|`{ "detail": "..." }`|
|Missing required query parameter|422|FastAPI default validation error shape|
|Render upstream outage (rare)|502 or 503|Render's load balancer page, not JSON|

**HTTP 422 note.** If a required query parameter is missing entirely (e.g., calling `/api/compute` without a `filter` parameter), the response is HTTP 422 with FastAPI's standard validation error shape, not HTTP 400. The body in that case is `{"detail": \[{"loc": \[...], "msg": "...", "type": "..."}]}`. Callers should handle 422 alongside 400 as "client input was wrong."

**HTTP 502/503 note.** If sco-ldc itself is unreachable (Render outage, cold-start in progress beyond the response timeout), the response may come from Render's load balancer rather than from the application. This response is HTML rather than JSON. Callers should not assume any non-200 response will be JSON; they should check the Content-Type header before parsing.

Callers should treat any non-2xx response as a generic failure unless they specifically want to surface the message from `detail`.

## 9\. Implementation notes

**Caching.** Both the NEA planet table and the ExoFOP TOI table are cached in-memory on the server and refreshed once daily at 17:00 UTC. Callers do not need to invalidate or refetch on a faster cadence. Repeated `/api/resolve` calls for the same planet within a session will return identical values until the next daily refresh.

**No rate limiting (currently).** The API has no enforced rate limits. The architecture handles dozens of requests per second comfortably. That said, individual workflows should not need more than a handful of calls per target, so this is not a real concern for normal use.

**Network failures.** If sco-ldc is unreachable (Render outage, DNS failure, network partition), the appropriate caller behavior is to surface this to the user and fall back to manual entry. The server should not be assumed to be always available.

**Connection reuse.** For occasional individual calls, connection management is a minor optimization. For higher-volume callers (a script processing many targets sequentially), it matters significantly: opening a fresh TLS connection per call adds 400-500 ms of handshake latency, while reusing an open connection eliminates this entirely. Most modern HTTP client libraries handle connection reuse via configured connection pools.

**Recommended timeouts.** Connection timeout 10 seconds, read timeout 30 seconds. Compute calls typically respond in 50-100 ms; resolve calls take a similar amount for cached planets and may take several seconds for live fallback queries when the cache misses. The 30-second read timeout gives ample headroom for slow network conditions without aborting on transient blips.

**Filter discovery.** Callers should ideally call `/api/filters` once per session (at startup, or lazily on first use) to discover the supported filters and grid ranges, rather than hardcoding the filter list. New filters may be added in future versions; calling `/api/filters` keeps the caller current automatically.

**Citation surfacing.** Each `/api/resolve` and `/api/compute` response includes a `citation` field. If a caller surfaces LDC values to a user, including the citation alongside (or in a tooltip) is the appropriate attribution and makes traceability straightforward. The two citations from a chained resolve+compute call are typically both needed for complete attribution in a publication.

## 10\. Example calls

### Health check

```
GET https://sco-ldc.com/api/health
```

```
curl https://sco-ldc.com/api/health
```

### Get filter catalog

```
GET https://sco-ldc.com/api/filters
```

```
curl https://sco-ldc.com/api/filters
```

### Resolve a planet name

```
GET https://sco-ldc.com/api/resolve?planet=WASP-23+b
```

```
curl 'https://sco-ldc.com/api/resolve?planet=WASP-23%20b'
```

### Resolve a TOI candidate

```
GET https://sco-ldc.com/api/resolve?planet=TOI-700.01
```

### Compute LDCs

```
GET https://sco-ldc.com/api/compute?teff=5150\&logg=4.4\&feh=-0.05\&filter=V\&model=ATLAS
```

```
curl 'https://sco-ldc.com/api/compute?teff=5150\&logg=4.4\&feh=-0.05\&filter=V\&model=ATLAS'
```

### Compute with default metallicity

```
GET https://sco-ldc.com/api/compute?teff=5150\&logg=4.4\&filter=V
```

Omitting `feh` defaults to 0.0; omitting `model` defaults to ATLAS.

### Full target-to-LDC chain

To produce LDCs for `WASP-23 b` in Johnson V:

1. `GET /api/resolve?planet=WASP-23+b` returns teff, logg, feh values from NEA.
2. `GET /api/compute?teff=<...>\&logg=<...>\&feh=<...>\&filter=V\&model=ATLAS` returns u1 and u2.

The two-step chain is the canonical pattern for integrators (such as AstroImageJ) where the user supplies a planet name and the integration tool needs to populate LDCs automatically.

## 11\. Contact

For questions about this API, feature requests, or breaking-change coordination, contact Ed Mullen, Sycamore Canyon Observatory.

Source code repository: https://github.com/SCO-SCI/sco-ldc


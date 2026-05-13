# Dictionary Collisions Clean-Room Specification

## 1. Scope

This document specifies the externally observable behavior of the `collisions` executable so that an independent implementation can be written without copying source structure.

It covers only the behavior of `collisions` itself. It does not prescribe any particular implementation strategy.

This specification covers:

1. Runtime file locations
2. Hash semantics
3. Command-line interfaces
4. Input file formats
5. Search behavior
6. Output behavior
7. Current observable quirks and edge cases

The current executable exposes three search modes:

1. Type mode (default)
2. Field mode (`--field`)
3. GBID mode (`--gbid`)

## 2. Runtime File Locations

At runtime, `collisions` expects the following data files to be discoverable by relative path from the current working directory when the corresponding feature is used:

1. `dict.txt`
2. `english_dict.txt`
3. `dict_expanded.txt`
4. `field_types.txt`
5. `unfound_hashes.txt`
6. `unfound_field_hashes.txt`

The following relative output directories are used when logging is enabled:

1. `type/`
2. `field/`
3. `gbid/`

Observed behavior:

1. These paths are opened as ordinary relative paths.
2. No absolute-path discovery convention is required.
3. When logging is enabled, output files are created beneath the mode-specific directory using the matched hash as the filename stem.

## 3. Hash Algorithms Used by `collisions`

All hashing is based on the same iterative update formula:

$$
h_{i+1} = (h_i \ll 5) + h_i + b_i
$$

Where:

1. Each candidate starts from $h_0 = 0$.
2. Arithmetic is 32-bit unsigned and therefore wraps modulo $2^{32}$.
3. Concatenated candidates behave as if their bytes were hashed in direct sequence, with no separators.

This is equivalent to multiplying by 33 and adding each byte value.

`collisions` uses three observable hash variants:

### 3.1 Type mode hash

1. Uses the bytes of each candidate exactly as constructed.
2. Applies the rolling hash with no final mask.

### 3.2 Field mode hash

1. Uses the same rolling hash as type mode.
2. Applies a final bitmask of `0x0FFFFFFF`.

Semantically, field mode exposes a 28-bit visible hash carried in a 32-bit integer.

### 3.3 GBID mode hash

1. Lowercases ASCII letters for hashing purposes before each byte update.
2. Applies no final mask.

In GBID mode, hash matching is therefore case-insensitive with respect to ASCII letters, even if the printed candidate text retains original casing from explicit token inputs.

## 4. `collisions` Purpose

The `collisions` executable enumerates candidate strings made from concatenated token pools and reports the candidates whose mode-specific hash matches at least one target hash.

Mode changes the hash semantics and some token-source behavior.

## 5. `collisions` Inputs

### 5.1 Target hashes

Target hashes may come from either or both of these sources:

1. Positional command-line arguments interpreted as hexadecimal integers
2. Standard input, when stdin is not a TTY, also interpreted as hexadecimal integers

Supplied hashes from both sources are unioned and deduplicated.

If no target hashes are supplied after reading command-line arguments and stdin, the default target set is:

1. Type mode: hashes from `unfound_hashes.txt`
2. Field mode: hashes from `unfound_field_hashes.txt`
3. GBID mode: no default target set

### 5.2 Filtering of supplied hashes

If the user explicitly supplies target hashes, supplied hashes are filtered unless `--force` is specified.

Filtering source by mode:

1. Type mode: retain only hashes present in `unfound_hashes.txt`
2. Field mode: retain only hashes present in `unfound_field_hashes.txt`
3. GBID mode: current behavior uses no repo-backed target set, so explicitly supplied hashes are removed unless `--force` is used

Observable behavior:

1. Removed hashes are announced on stdout as `removing already known hash: <hex>`.
2. After filtering, if no targets remain, the program exits successfully without performing a search.

### 5.3 Dictionary files

Dictionary files are plain text files with one token per line.

Rules:

1. Empty lines are ignored.
2. Lines beginning with `#` are ignored.
3. Every other line is treated as a literal token.

If a dictionary file cannot be opened, an error is printed to stderr in the form:

```text
Dictionary <path> not found.
```

The program does not abort on this condition; it continues with whatever other token sources remain available.

### 5.4 `field_types.txt`

`field_types.txt` is parsed as whitespace-separated pairs of hexadecimal integers:

1. First value: field hash
2. Second value: associated type hash

Multiple rows may map the same field hash to multiple type hashes.

This file is only relevant in field mode.

## 6. Candidate Token Sources

### 6.1 Main dictionary pool

Unless `--no-dict` is active, the main pool starts from file-backed dictionary content.

Observed dictionary selection behavior:

1. Default: load `dict.txt`
2. `--english`: load `dict.txt`, then load `english_dict.txt`
3. `--expanded`: load `dict_expanded.txt`
4. `--dict <value>`: load `<value>` as the selected dictionary path
5. Special case: `--dict english_dict.txt` first loads `dict.txt`, then loads `english_dict.txt`

Observable details:

1. `--no-dict` disables file-backed dictionary loading even if other dictionary-selection flags are also present.
2. The current repository does not include `dict_expanded.txt`, so `--expanded` currently produces a missing-dictionary warning unless that file is supplied externally.

### 6.2 Default built-in token pool

Unless both `--words-only` and file-backed dictionary loading are enabled, the program also appends built-in single-token candidates.

Built-in tokens:

1. Type mode and field mode: uppercase letters `A-Z`
2. All modes: lowercase letters `a-z`
3. All modes: digits `0-9`
4. All modes: `_`

Observable detail:

1. Built-in suppression is based on file-backed dictionary loading being enabled, not on whether the selected dictionary file actually opened successfully.

### 6.3 Dictionary normalization

When file-backed dictionary entries are loaded and `--literal` is not active:

1. Entries shorter than 2 characters are ignored unless `--words-only` is active.
2. Multi-character entries containing no lowercase ASCII letters are ignored by default.
3. `--allow-all-caps` disables that filter.
4. In type mode and field mode, an accepted fully lowercase entry is transformed to title case with only the first character uppercased.
5. In GBID mode, an accepted entry is lowercased.
6. Mixed-case accepted entries otherwise remain unchanged.
7. Accepted dictionary entries are deduplicated within the main dictionary pool.

Examples of current normalization:

1. `foo` from a dictionary file becomes `Foo` in type mode and field mode.
2. `foo` from a dictionary file becomes `foo` in GBID mode.
3. `NASA` is ignored by default, but retained with `--allow-all-caps`.

### 6.4 Literal dictionary mode

If `--literal` is active, dictionary entries are inserted exactly as read, without case normalization or the no-lowercase filter. Main-dictionary deduplication still applies.

## 7. Search Space Construction

The search engine builds candidate strings by concatenating one token at each logical position from position `0` to `length - 1`.

For a given position, the candidate source is chosen as follows:

1. If a suffix token list exists and this is the final position, use the suffix token list.
2. Else if a position-specific token list exists for this position, use that list.
3. Else use the main dictionary pool.

The candidate string is the direct concatenation of the chosen tokens, with no separators.

If any required position has an empty token pool for a searched length, that length produces no candidates.

## 8. Length Semantics

The program searches candidate lengths from `min` through `max`, inclusive.

Default effective bounds:

1. Minimum length: `1`
2. Maximum length: `64`

Observed CLI interpretation:

1. `--min N` only takes effect when `1 <= N < 64`
2. `--max N` only takes effect when `1 <= N < 64`
3. Values outside that accepted range are ignored silently

Observable diagnostic behavior:

1. An accepted `--min N` is reported as `Using min of <N-1>`
2. An accepted `--max N` is reported as `Using max of <N>`
3. These lines appear only when the corresponding option took effect

Example:

1. `--min 1 --max 1` prints `Using min of 0` and `Using max of 1`, then searches only length 1.

## 9. Command-Line Interface for `collisions`

### 9.1 Mode selection

1. `--field` -> use field hashing rules
2. `--gbid` -> use GBID hashing rules
3. Default -> use type hashing rules

### 9.2 Dictionary source controls

1. `--dict <value>` -> use the following argument as a dictionary path or identifier
2. `--english` -> load `dict.txt` and `english_dict.txt`
3. `--expanded` -> load `dict_expanded.txt`
4. `--no-dict` -> disable file-backed dictionary loading
5. `--literal` -> disable normalization of loaded dictionary entries
6. `--words-only` -> suppress the built-in single-character token pool when file-backed dictionary loading is enabled
7. `--allow-all-caps` -> retain multi-character dictionary entries that would otherwise be rejected for lacking lowercase ASCII letters

### 9.3 Position-specific token controls

1. `--prefix <tokens>` -> assign a token list to position 0
2. `--suffix <tokens>` -> assign a token list to the final position
3. `--subdict <position> <tokens>` -> assign a token list to an arbitrary zero-based position
4. `--no-prefix` -> disable automatic field-mode prefix inference

Token argument parsing for `--prefix`, `--suffix`, and `--subdict`:

1. The next command-line token is consumed as a single argument.
2. That argument is parsed as a whitespace-delimited list.
3. In ordinary shell usage this usually means one shell argument yields one token unless the caller deliberately passes embedded spaces via quoting.

### 9.4 Search size and execution controls

1. `--min <N>` -> request a minimum candidate length, subject to the accepted range described above
2. `--max <N>` -> request a maximum candidate length, subject to the accepted range described above
3. `--threads <N>` -> requested worker count, clamped into `[1, 64]`
4. `--log` -> append matches to per-hash files in the mode-specific output directory

### 9.5 Hash-set selection and filtering controls

1. `--force` -> keep explicitly supplied hashes even if they would otherwise be filtered out
2. `--paired` -> in type mode only, require candidate words to also imply at least one plausible related field hash

Observed behavior:

1. `--paired` is ignored outside type mode.

### 9.6 Field-type prefix heuristics

1. `--common` -> use common field-name prefixes only
2. `--uncommon` -> enable a broader prefix set

Observed behavior:

1. Common-prefix mode is already the default.
2. `--common` is therefore idempotent.

### 9.7 Unknown options and missing arguments

Any unrecognized option beginning with `-` causes immediate failure:

```text
Error: Unknown option: <option>
```

Missing required option arguments also cause immediate failure with an `Error:` message naming the option.

Exit status should be nonzero in these failure cases.

## 10. Mode-Specific Search Semantics

### 10.1 Type mode

Type mode uses the unmasked rolling hash and generally searches title-cased word-like tokens unless token inputs were supplied literally.

If `--paired` is active in type mode, a candidate is accepted only if at least one of these related field-name forms hashes to a member of `unfound_field_hashes.txt`:

1. `t<word>`
2. `pt<word>`
3. `ar<word>`
4. `t<word>s`
5. `pt<word>s`
6. `ar<word>s`

### 10.2 Field mode

Field mode uses the masked hash and can use `field_types.txt` plus a prefix-heuristic table to constrain the first token.

Observed behavior:

1. If `--no-prefix` is not active and no explicit position-0 token list is supplied, the program infers candidate position-0 prefixes from the target field hashes and their associated type hashes.
2. When a type has no specific prefix mapping, a fallback prefix set headed by `t` is used.
3. Supplying an explicit position-0 token list suppresses automatic prefix generation, but does not necessarily suppress later field/type gating.
4. If `--no-prefix` is active, `--literal` is not active, and no explicit position-0 token list is provided, the program synthesizes a position-0 token pool by lowercasing only the first character of every main-pool token.
5. If `--no-prefix` is active together with `--literal`, no such synthesized position-0 pool is created.
6. If every targeted field references the same single type hash, the field/type gate is disabled after prefix preparation.

Observable quirks:

1. The `--no-prefix` synthesized position-0 list is not deduplicated, so duplicate textual outputs are possible when different source tokens collapse to the same first-token spelling.
2. When a field hash maps to multiple type hashes, automatic prefix generation can draw prefixes from multiple associated types, but the final acceptance gate currently uses only the first listed associated type for that field hash.

### 10.3 GBID mode

GBID mode uses the lowercase-hash variant with no final mask.

Observed behavior:

1. Built-in uppercase single-character tokens are not added in this mode.
2. Non-literal dictionary entries are normalized to lowercase.
3. Field-type prefix inference is not used.
4. `--paired` has no effect.
5. There is no default repo-backed target list for this mode.
6. As a result, explicitly supplied GBID targets currently require `--force` if the caller wants them searched.

## 11. Matching Rules

A candidate is reported only if all applicable gates pass:

1. The candidate hash matches a target hash under the active mode's hash semantics.
2. The type-mode `--paired` gate passes, if enabled.
3. The field-mode type-prefix gate passes, if active.

If any applicable gate fails, the candidate is silently discarded.

## 12. Output Contract

### 12.1 Match output

Each accepted candidate is printed to stdout as:

```text
  <hex-hash>: <candidate>
```

Characteristics:

1. Two leading spaces are present before the hash.
2. Hashes are printed in lowercase hexadecimal.
3. Multiple matches for the same hash are printed as separate lines.
4. Duplicate textual matches may also appear as separate lines if they arise from different token combinations.

### 12.2 Progress and diagnostics

During a normal search run, the executable may write diagnostics such as the following to stderr:

1. `Using min of <N>` for an accepted minimum setting, where the printed value is one less than the user-facing minimum length
2. `Using max of <N>` for an accepted maximum setting
3. `Type prefix size: <N>`
4. `Prefix size: <N>`
5. `Dictionary size: <N>`
6. `Suffix size: <N>`
7. `Matching <N> hashes.`
8. `Using <N> workers.` when more than one worker is active
9. `Length: <N>` before each searched candidate length
10. Final hash-rate summary such as `Hash rate: 2M/s`

Observable detail:

1. If target selection yields an empty target set, the program exits before emitting the normal search diagnostics above.

### 12.3 Logging mode

If `--log` is enabled, every reported match is also appended to a file determined by mode and matched hash:

1. Type mode -> `type/<hash>.yml`
2. Field mode -> `field/<hash>.yml`
3. GBID mode -> `gbid/<hash>.yml`

File content format per appended line:

```text
<hex-hash>: <candidate>
```

This is YAML-like text, not a complete YAML document contract.

## 13. Concurrency and Termination

### 13.1 Worker model

The search may use multiple workers. The requested worker count is taken from `--threads`, clamped to 1 through 64.

Externally observable properties:

1. `--threads 1` performs a valid single-worker search.
2. Higher thread counts may improve throughput.
3. Match ordering is not guaranteed to be stable across runs when parallel execution is used.

### 13.2 Signal handling

The program stops early when interrupted by common termination signals.

Minimum compatibility requirement:

1. Respond to Ctrl+C / SIGINT by terminating the search loop reasonably promptly.
2. Respond similarly to SIGTERM.

## 14. Runtime Data Files

The runtime data files have the following observed roles:

1. `dict.txt` is a domain-specific token list.
2. `english_dict.txt` is an additional English word list.
3. `dict_expanded.txt`, when present, is an alternate expanded token list.
4. `unfound_hashes.txt` contains unresolved type hashes, one hex value per line.
5. `unfound_field_hashes.txt` contains unresolved field hashes, one hex value per line.
6. `field_types.txt` maps field hashes to candidate type hashes.

## 15. Known Observable Quirks

The current program exposes several quirks that are part of present behavior:

1. `--expanded` references `dict_expanded.txt`, but that file is absent from this repository.
2. The program accepts hashes from stdin and command-line arguments in the same run, effectively unioning them.
3. A missing dictionary file prints an error but does not force termination.
4. GBID mode has no default repo-backed target set, so explicit GBID targets are filtered out unless `--force` is supplied.
5. User-supplied `--min` and `--max` values accept `1..63`, even though the default maximum search length is `64`.
6. The special `--no-prefix` synthesized field-prefix pool is created only when `--literal` is not active.

## 16. Representative Black-Box Checks

The following checks summarize the externally observable behavior described above:

1. `collisions` with no targets in type mode loads targets from `unfound_hashes.txt`.
2. `collisions --field` with no targets loads targets from `unfound_field_hashes.txt`.
3. `collisions --gbid` with no explicit targets performs no search and exits successfully.
4. Supplying a target hash without `--force` causes filtering against the mode's repo-backed target set, if any.
5. Supplying an explicit GBID target without `--force` currently removes that target instead of searching it.
6. `--english` loads a broader dictionary pool than the default dictionary mode when both source files are present.
7. `--threads 1` performs a complete search without worker-thread requirements.
8. `--log` appends to the correct mode-specific per-hash file.
9. Field mode applies the 28-bit final mask when matching target hashes.
10. Type mode `--paired` suppresses candidates that do not imply one of the accepted related field-name forms.

## 17. Summary

At a black-box level, `collisions` is a dictionary-driven brute-force hash collision finder for three hash domains: type, field, and GBID. The most important behavioral requirements are the exact rolling-hash semantics, the mode-specific token normalization rules, the target-filtering behavior, the field-prefix heuristics and gates, and the mode-specific logging and default-target rules.
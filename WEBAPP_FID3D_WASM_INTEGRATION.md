# fid_3d WebAssembly Integration Guide

This guide is for another AI tool (or engineer) integrating the 3D processing API into a web app.

## Goal

Integrate the `fid_3d` wasm class so the app can run:
- full processing
- direct-only processing
- indirect-only processing

using in-memory inputs (no file I/O):
- Bruker text files as strings (`pulseprogram`, `acqus`, `acqu2s`, `acqu3s`, optional `nuslist`)
- Bruker raw FID/SER as binary bytes

## Exposed API (wasm)

Input/setup:
- `set_up_apodization_from_string(direct, indirect1, indirect2)`
- `run_zf(zfDirect, zfIndirect1, zfIndirect2)`
- `extract_region(fromNorm, toNorm)`
- `extract_region_ppm(fromPpm, toPpm)`
- `read_phase_correction_from_string(textWith6Numbers)`
- `set_time_domain_polynorminal_order(f2, f1, f3)`
- `set_frq_domain_polynorminal_order(f2, f1, f3)`
- `set_inverse(f2, f1, f3)`
- `set_delete_image(f2, f1, f3)`

In-memory Bruker load:
- `read_bruker_files_as_strings(contentsPulse, contentsAcqus, contentsAcqu2s, contentsAcqu3s, contentsNuslist, nusSerInflated)`
- `read_bruker_fid_data(vectorFloat)`
- `read_bruker_fid_data_bytes(vectorUChar)`

Run modes:
- `full_process()`
- `direct_only_process()`
- `indirect_only_process()`

Output access:
- `prepare_header_for_nmrpipe()`
- `get_nmrpipe_header_data()`
- `get_data_of_rrr()` ... `get_data_of_iii()`
- `get_ndata_frq()`
- `get_ndata_frq_indirect1()`
- `get_ndata_frq_indirect2()`

## Required call order

Use this exact order unless there is a strong reason not to.

1. Create instance: `const fid = new Module.fid_3d()`
2. Set processing flags/options:
- zf
- apodization
- extraction (optional)
- phase correction text (optional)
- polynomial orders
- inverse/delete-image triplets
3. Load Bruker metadata from strings:
- `read_bruker_files_as_strings(...)`
4. Load Bruker binary payload:
- preferred: `read_bruker_fid_data_bytes(...)`
5. Run one mode:
- full: `full_process()`
- direct-only: `direct_only_process()`
- indirect-only: `indirect_only_process()`
6. Export output pointers and dimensions:
- `prepare_header_for_nmrpipe()`
- get dims
- read header/data pointers from HEAP
7. Cleanup wasm objects (`delete`) to avoid memory leaks.

## Minimal integration skeleton (JavaScript)

```js
export function runFid3dWasm(Module, cfg, textInputs, fidBytes, mode) {
  const fid = new Module.fid_3d();

  try {
    fid.run_zf(cfg.zfDirect, cfg.zfIndirect1, cfg.zfIndirect2);

    fid.set_up_apodization_from_string(
      cfg.apodDirect,
      cfg.apodIndirect1,
      cfg.apodIndirect2
    );

    if (cfg.extNorm) {
      fid.extract_region(cfg.extNorm[0], cfg.extNorm[1]);
    } else if (cfg.extPpm) {
      fid.extract_region_ppm(cfg.extPpm[0], cfg.extPpm[1]);
    }

    if (cfg.phaseText) {
      fid.read_phase_correction_from_string(cfg.phaseText);
    }

    fid.set_time_domain_polynorminal_order(
      cfg.tdPolyOrder[0],
      cfg.tdPolyOrder[1],
      cfg.tdPolyOrder[2]
    );

    fid.set_frq_domain_polynorminal_order(
      cfg.frqPolyOrder[0],
      cfg.frqPolyOrder[1],
      cfg.frqPolyOrder[2]
    );

    fid.set_inverse(cfg.inverse[0], cfg.inverse[1], cfg.inverse[2]);
    fid.set_delete_image(cfg.deleteImage[0], cfg.deleteImage[1], cfg.deleteImage[2]);

    fid.read_bruker_files_as_strings(
      textInputs.pulse || "",
      textInputs.acqus,
      textInputs.acqu2s,
      textInputs.acqu3s,
      textInputs.nuslist || "",
      !!cfg.nusSerInflated
    );

    const v = new Module.VectorUChar();
    for (let i = 0; i < fidBytes.length; i++) v.push_back(fidBytes[i]);
    fid.read_bruker_fid_data_bytes(v);
    v.delete();

    if (mode === "full") {
      fid.full_process();
    } else if (mode === "direct-only") {
      fid.direct_only_process();
    } else if (mode === "indirect-only") {
      fid.indirect_only_process();
    } else {
      throw new Error("Unsupported mode: " + mode);
    }

    fid.prepare_header_for_nmrpipe();

    const nx = fid.get_ndata_frq();
    const ny = fid.get_ndata_frq_indirect1();
    const nz = fid.get_ndata_frq_indirect2();
    const n = nx * ny * nz;

    const headerPtr = fid.get_nmrpipe_header_data();
    const rrrPtr = fid.get_data_of_rrr();

    // Views into wasm memory (copy if you need persistence)
    const headerF32 = Module.HEAPF32.subarray(headerPtr >> 2, (headerPtr >> 2) + 512);
    const rrrF32 = Module.HEAPF32.subarray(rrrPtr >> 2, (rrrPtr >> 2) + n);

    return { dims: { nx, ny, nz }, headerF32, rrrF32 };
  } finally {
    fid.delete();
  }
}
```

## Common pitfalls

- Do not call `read_bruker_fid_data_bytes` before `read_bruker_files_as_strings`.
- If using NUS:
- pass nus list text in `read_bruker_files_as_strings`.
- set `nusSerInflated` correctly.
- Keep `mode` semantics correct:
- `direct-only` stops after direct axis processing.
- `indirect-only` expects a post-direct/pre-indirect stage state.
- Typed-array views into wasm memory become invalid after `fid.delete()`.
- If you need persistent output, copy arrays before delete.

## Prompt template for another AI tool

Use this prompt with your other AI coding tool:

"""
Integrate the `fid_3d` wasm API from this repository into my web app.

Constraints:
1) Use in-memory data only, no filesystem reads.
2) Inputs are:
   - pulseprogram/acqus/acqu2s/acqu3s text as strings
   - optional nuslist text as string
   - fid/ser binary as Uint8Array
3) Follow this exact call order:
   - configure flags/options
   - read_bruker_files_as_strings
   - read_bruker_fid_data_bytes
   - run selected mode (full/direct-only/indirect-only)
   - prepare_header_for_nmrpipe and extract pointers/dims
4) Expose one high-level function:
   runFid3d(mode, config, textInputs, fidBytes)
5) Return dimensions and real spectrum data (RRR), with optional other components.
6) Manage wasm object lifetimes correctly (delete temporary vectors and fid object).

Please generate production-quality TypeScript code with input validation and clear error messages.
"""

## Notes

- The C++ binding names and methods are implemented in `fid_3d` under the `dp_module_fid_3d` embind module.
- The app can default to using only `RRR` output for display and load additional components only when needed.

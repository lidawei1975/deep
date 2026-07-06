# DEEP-Picker WebAssembly Binding Reference

## Complete List of Bound Functions and Variables

This document provides a comprehensive reference for all WebAssembly-bound functions available in the DEEP-Picker library.

---

## 1. Class Hierarchy

```
fid_base (base class)
├── fid_2d (2D FID processing)
│   └── spectrum_phasing (automatic phase correction)
└── [other derived classes...]
```

---

## 2. fid_2d Class - Parameter Configuration

| Function | Signature | Returns | Purpose | Example |
|----------|-----------|---------|---------|---------|
| **set_aqseq** | `set_aqseq(mode: string)` | `bool` | Set acquisition sequence mode | `fid2d.set_aqseq("321")` |
| **set_negative** | `set_negative(flag: bool)` | `void` | Set negative imaginary flag | `fid2d.set_negative(true)` |
| **set_first_only** | `set_first_only(flag: bool)` | `void` | Process first spectrum only | `fid2d.set_first_only(false)` |
| **run_zf** | `run_zf(zf_direct: int, zf_indirect: int)` | `bool` | Set zero filling factors | `fid2d.run_zf(2, 2)` |

**Parameter Details:**

- **set_aqseq**: `mode` must be `"321"` or `"312"` depending on pulse sequence
- **set_negative**: `true` = imaginary data is negative; `false` = positive
- **set_first_only**: `true` = limit processing to first spectrum in pseudo-3D data
- **run_zf**: Typical values are `1, 2, 4, 8, 16` (multiplication factors)

---

## 3. fid_3d Class - Parameter Configuration

| Function | Signature | Returns | Purpose | Example |
|----------|-----------|---------|---------|---------|
| **run_zf** | `run_zf(zf_direct: int, zf_indirect1: int, zf_indirect2: int)` | `bool` | Set zero filling factors for all three axes | `fid3d.run_zf(2, 2, 2)` |

**Parameter Details:**

- **run_zf**: Typical values are `1, 2, 4, 8, 16` (multiplication factors)

---

## 4. fid_3d Class - Data Input Methods

| Function | Signature | Returns | Purpose | Notes |
|----------|-----------|---------|---------|-------|
| **read_bruker_files** | `read_bruker_files(pulse_prog: string, acqu2s: string, acqu3s: string, acqus: string, fid_file: string, nuslist_contents?: string, simulate_nus?: bool, nus_ser_inflated?: bool)` | `bool` | Read Bruker data from file paths | Traditional file-based approach |
| **read_bruker_files_as_strings** | `read_bruker_files_as_strings(pulse: string, acqus: string, acqu2s: string, acqu3s: string, nuslist_contents?: string, simulate_nus?: bool, nus_ser_inflated?: bool)` | `bool` | Read Bruker parameters from text | In-memory approach (recommended for web) |
| **read_bruker_fid_data_bytes** | `read_bruker_fid_data_bytes(fid_bytes: Uint8Array)` | `bool` | Read Bruker FID binary data | Companion to read_bruker_files_as_strings |
| **read_nmrpipe_file_from_buffer** | `read_nmrpipe_file_from_buffer(nmrpipe_bytes: Uint8Array)` | `bool` | Read NMRPipe binary format | Complete file read in one call |

**Usage Notes:**

- **read_bruker_files**: Reads directly from filesystem (not suitable for web)
- **read_bruker_files_as_strings + read_bruker_fid_data_bytes**: Two-step in-memory approach
  - First call: parameters (from uploaded files)
  - Second call: FID binary data (from uploaded files)
- **simulate_nus**: When `true`, the loader reads as fully sampled and zeros traces not listed in the nuslist after unpacking
- **nus_ser_inflated**: When `true`, the Bruker raw data is already on the full NusTD grid and should not be expanded
- **read_nmrpipe_file_from_buffer**: Single-step approach for NMRPipe data

**WebAssembly Note:** The 3D loader owns NUS handling internally. For simulate-NUS workflows, pass the nuslist contents, `simulate_nus=true`, and `nus_ser_inflated=false` if the source data is not already inflated.

---

## 5. fid_2d Class - Data Processing Methods

| Function | Signature | Returns | Purpose | Parameters |
|----------|-----------|---------|---------|-----------|
| **full_process** | `full_process(remove_di_direct?: bool, remove_di_indirect?: bool)` | `bool` | Perform FFT in both dimensions | `true` = remove Bruker digitizer filter artifacts |
| **polynorminal_baseline** | `polynorminal_baseline(order: int)` | `bool` | Apply polynomial baseline correction | `order`: 0 (constant), 1 (linear), 2 (quadratic), 3 (cubic) |
| **extract_region** | `extract_region(from: double, to: double)` | `bool` | Extract spectral region | `from, to` in range [0, 1] where 0=left edge, 1=right edge |

**Processing Chain:**

1. Read data (one of the input methods above)
2. Configure parameters (set_aqseq, set_negative, run_zf, etc.)
3. **full_process** - FFT along direct dimension → transpose → FFT along indirect dimension
4. **polynorminal_baseline** (optional) - Remove baseline
5. **extract_region** (optional) - Focus on specific frequency range
6. Export results (see output methods below)

---

## 6. fid_2d Class - Data Output Methods

| Function | Signature | Returns | Purpose | File Format |
|----------|-----------|---------|---------|-------------|
| **write_nmrpipe_ft2** | `write_nmrpipe_ft2(filename: string)` | `bool` | Write spectrum to file | NMRPipe ft2 binary format |
| **write_nmrpipe_ft2_to_buffer** | `write_nmrpipe_ft2_to_buffer()` | `uintptr_t` | Export spectrum to byte array buffer, returns address | In-memory (recommended for web) |
| **write_nmrpipe_ft2_to_buffer_index** | `write_nmrpipe_ft2_to_buffer_index(spec_index: int)` | `uintptr_t` | Export specific plane of pseudo-3D spectrum to buffer | In-memory (recommended for web) |
| **get_ft2_size_in_float32** | `get_ft2_size_in_float32()` | `int` | Get size of exported ft2 buffer in float32 elements | Helper for reading heap directly |
| **get_nspectra** | `get_nspectra()` | `int` | Get total number of spectra/planes in loaded dataset | Helper for pseudo-3D loops |
| **write_nmrpipe_intermediate_to_buffer** | `write_nmrpipe_intermediate_to_buffer(output: VectorUChar)` | `bool` | Export half-Fourier data | In-memory intermediate format |
| **write_nmrpipe_fid_to_buffer** | `write_nmrpipe_fid_to_buffer(output: VectorUChar)` | `bool` | Export FID time-domain data | In-memory FID format |
| **write_pipe_to_buffer** | `write_pipe_to_buffer(output: VectorUChar, real_only?: bool)` | `bool` | Generic NMRPipe export | `real_only=true` → real part only |
| **write_json** | `write_json(filename: string)` | `bool` | Write metadata to JSON file | JSON format |
| **write_pseudo3d_json** | `write_pseudo3d_json(filename: string)` | `bool` | Write pseudo-3D metadata | JSON format (number of spectra, etc.) |

**Output Recommendations:**

- **File-based**: `write_nmrpipe_ft2()`, `write_json()`, `write_pseudo3d_json()`
- **In-memory (Web)**: `write_nmrpipe_ft2_to_buffer()` → download or transmit

---

## 7. fid_2d Class - Apodization Methods

| Function | Signature | Returns | Purpose | Notes |
|----------|-----------|---------|---------|-------|
| **set_up_apodization** | `set_up_apodization(a1: ptr, a2: ptr)` | `bool` | Setup apodization windows | `a1`=direct dim, `a2`=indirect dim |
| **set_up_apodization_from_string** | `set_up_apodization_from_string(direct: string, indirect: string)` | `bool` | Setup from text descriptions | Text format for window functions |

**Placeholder Note:** Actual apodization class binding and string format to be determined based on `apodization` class binding in embind.

---

## 8. spectrum_phasing Class - Specialized Methods

| Function | Signature | Returns | Purpose | Notes |
|----------|-----------|---------|---------|-------|
| **auto_phase_correction_v2** | `auto_phase_correction_v2()` | `bool` | Automatic phase correction | Entropy-based optimization algorithm |
| **save_phase_correction_result** | `save_phase_correction_result(filename: string)` | `bool` | Save phase values to file | Format: `p0_direct p1_direct p0_indirect p1_indirect` |
| **save_phase_correction_result_as_string** | `save_phase_correction_result_as_string()` | `string` | Get phase values as string | Returns space-separated values |

**Phase Correction Details:**

- `auto_phase_correction_v2()` automatically finds optimal p0 and p1 values
- Results stored in protected members: `final_p0_direct`, `final_p1_direct`, `final_p0_indirect`, `final_p1_indirect`
- Can be retrieved by getter methods or saved directly

---

## 9. spectrum_phasing Class - Inherited Methods from fid_2d

`spectrum_phasing` inherits all methods from `fid_2d`, including:
- All parameter configuration methods (set_aqseq, set_negative, etc.)
- All data input methods (read_bruker_files_as_strings, etc.)
- All data processing methods (full_process, polynorminal_baseline, etc.)
- All output methods (write_nmrpipe_ft2_to_buffer, write_json, etc.)

---

## 10. Helper Types and Utilities

### Vector Types (Registered in Emscripten)

| Type | JavaScript Name | Usage |
|------|-----------------|-------|
| `vector<unsigned char>` | `VectorUChar` | For binary file I/O |
| `vector<string>` | `VectorString` | For file lists |
| `vector<float>` | `VectorFloat` | For pre-decoded numeric data |
| `vector<int>` | `VectorInt` | For integer arrays |

### Creating and Using Vectors

```javascript
// Create vector
const byteVector = new Module.VectorUChar();
const stringVector = new Module.VectorString();

// Add items
byteVector.push_back(0xFF);  // Add single byte
stringVector.push_back("file1.txt");
stringVector.push_back("file2.txt");

// Get size
const size = byteVector.size();

// Access element
const byte = byteVector.get(0);

// Clean up
byteVector.delete();
stringVector.delete();
```

---

## 11. Recommended Workflow Patterns

### Pattern A: Bruker In-Memory (Recommended for Web)

```
1. Load 3 text files:
   - Pulse program (optional)
   - acqus file (direct dimension parameters)
   - acqu2s file (indirect dimension parameters)

2. Load binary file:
   - ser or fid file (raw FID data)

3. Call in order:
   fid2d.read_bruker_files_as_strings(pulse_text, acqus_text, acqu2s_text);
   fid2d.read_bruker_fid_data_bytes(fid_bytes);

4. Configure (optional):
   fid2d.set_aqseq("321");
   fid2d.run_zf(2, 2);
   fid2d.set_up_apodization_from_string(direct_apod, indirect_apod);

5. Process:
   fid2d.full_process(false, false);

6. Optional:
   fid2d.polynorminal_baseline(1);

7. Phase correct (use spectrum_phasing instead):
   phasing.auto_phase_correction_v2();

8. Export:
   const address = phasing.write_nmrpipe_ft2_to_buffer();
   const size = phasing.get_ft2_size_in_float32();
   phasing.save_phase_correction_result("phase.txt");
```

### Pattern B: NMRPipe In-Memory

```
1. Load binary file:
   - nmrPipe ft2 or fid file

2. Call:
   fid2d.read_nmrpipe_file_from_buffer(nmrpipe_bytes);

3. Optional extraction:
   fid2d.extract_region(0.2, 0.8);

4. Export:
   const address = fid2d.write_nmrpipe_ft2_to_buffer();
   const size = fid2d.get_ft2_size_in_float32();
```

### Pattern C: Bruker File-Based (Legacy)

```
1. Prepare file paths:
   - pulse_program/path
   - acqus/path
   - acqu2s/path
   - [fid/path, ...] (can be multiple)

2. Call:
   fid2d.read_bruker_files(pulse_path, acqu2s_path, acqus_path, fid_paths);

3. Rest same as Pattern A from step 4 onward
```

---

## 11. Data Type Mappings

| C++ Type | JavaScript Type | Notes |
|----------|-----------------|-------|
| `bool` | `boolean` | true/false |
| `int` | `number` | Integer values |
| `double` | `number` | Floating-point (full precision) |
| `float` | `number` | Floating-point (32-bit) |
| `string` | `string` | UTF-8 text |
| `vector<unsigned char>` | `VectorUChar` | Byte array via embind vector |
| `vector<string>` | `VectorString` | String array via embind vector |
| `apodization*` | Special | Requires apodization class binding |

---

## 12. Error Handling

Most functions return `bool`:
- `true` = operation succeeded
- `false` = operation failed

Check return values and log errors:

```javascript
const success = fid2d.read_bruker_files_as_strings(pulse, acqus, acqu2s);
if (!success) {
  console.error("Failed to read Bruker files");
  // Handle error appropriately
}
```

---

## 13. Memory Management

**Important:** Clean up WASM objects when done:

```javascript
const fid2d = new Module.fid_2d();
const phasing = new Module.spectrum_phasing();

// ... do work ...

// Always delete when finished
fid2d.delete();
phasing.delete();

// For vectors:
const byteVector = new Module.VectorUChar();
// ... use vector ...
byteVector.delete();
```

---

## 14. Example: Quick Start

```javascript
// Initialize
const phasing = new Module.spectrum_phasing();

// Load file contents (from HTML file input)
const pulseContent = await readFileAsString(pulseFile);
const acqusContent = await readFileAsString(acqusFile);
const acqu2sContent = await readFileAsString(acqu2sFile);
const fidBytes = await readFileAsBytes(fidFile);

// Read Bruker data
phasing.read_bruker_files_as_strings(pulseContent, acqusContent, acqu2sContent);
phasing.read_bruker_fid_data_bytes(fidBytes);

// Process
phasing.set_aqseq("321");
phasing.run_zf(2, 2);
phasing.full_process(false, false);

// Phase correct
phasing.auto_phase_correction_v2();

// Export
const address = phasing.write_nmrpipe_ft2_to_buffer();
const size = phasing.get_ft2_size_in_float32();
const byteArray = new Uint8Array(Module.HEAPF32.buffer, address, size * 4).slice();

// Download
downloadFile(new Blob([byteArray]), "processed.ft2");

// Cleanup
phasing.delete();
```

---

## 15. Additional Resources

- See `example_wasm_workflow.js` for complete working examples
- See C++ header files for detailed function documentation
- Emscripten binding definitions are in C++ source files (EMSCRIPTEN_BINDINGS macro)

---

## Notes for JavaScript Developers

1. **File I/O**: Use HTML5 File API to read files, then pass contents to WASM functions
2. **Byte Arrays**: Convert between Uint8Array and VectorUChar as needed
3. **Async Operations**: WASM functions are synchronous; wrap in worker if needed for long operations
4. **Memory**: Always call `.delete()` on WASM objects to avoid memory leaks
5. **Error Handling**: Check return values and wrap in try-catch as appropriate
6. **Performance**: In-memory approach is recommended over file-based for web applications


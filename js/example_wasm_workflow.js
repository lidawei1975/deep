/**
 * DEEP-Picker WebAssembly Binding - Complete Workflow Example
 * 
 * This file demonstrates how to use all the WebAssembly-bound functions
 * for 2D NMR processing with fid_2d and spectrum_phasing classes.
 * 
 * Workflow Overview:
 * 1. Initialize WASM module
 * 2. Read Bruker NMR data (in-memory approach recommended)
 * 3. Configure processing parameters
 * 4. Perform spectral processing
 * 5. Apply automatic phase correction
 * 6. Output results
 */

// ============================================================================
// PART 1: MODULE INITIALIZATION
// ============================================================================

// Placeholder: Load WASM module (compiled from deep-picker)
// In a real application, you'd use dynamic import or <script> tag
let Module = {}; // This will be populated by Emscripten

async function initializeWasmModule() {
  // Placeholder: Load the compiled wasm module
  // Example: Module = await import('./deep_picker.js');
  
  console.log("WASM Module initialized");
  return Module;
}

// ============================================================================
// PART 2: HELPER FUNCTIONS FOR FILE I/O
// ============================================================================

/**
 * Read a file as string (for text-based Bruker parameter files)
 * @param {File} file - HTML5 File object
 * @returns {Promise<string>} - File contents as string
 */
async function readFileAsString(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
}

/**
 * Read a file as byte array (for binary files like ser, fid, ft2)
 * @param {File} file - HTML5 File object
 * @returns {Promise<Uint8Array>} - File contents as unsigned char vector
 */
async function readFileAsBytes(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const arrayBuffer = e.target.result;
      const uint8Array = new Uint8Array(arrayBuffer);
      resolve(uint8Array);
    };
    reader.onerror = (e) => reject(e);
    reader.readAsArrayBuffer(file);
  });
}

/**
 * Convert Uint8Array to downloadable Blob
 * @param {Uint8Array} byteArray - Byte array to convert
 * @param {string} mimeType - MIME type (e.g., 'application/octet-stream')
 * @returns {Blob} - Blob object for download
 */
function byteArrayToBlob(byteArray, mimeType = 'application/octet-stream') {
  return new Blob([byteArray], { type: mimeType });
}

// ============================================================================
// PART 3: PARAMETER SETUP FUNCTIONS
// ============================================================================

/**
 * Demonstrates how to configure FID processing parameters
 * These are typically called before full_process()
 */
function setupProcessingParameters(fid2dInstance) {
  console.log("\n--- Setting up processing parameters ---");
  
  // PLACEHOLDER: Set acquisition sequence mode
  // Options: "321" or "312" - depends on your NMR experiment type
  const aqseq_mode = "321"; // PLACEHOLDER: Change based on your experiment
  const success_aqseq = fid2dInstance.set_aqseq(aqseq_mode);
  console.log(`set_aqseq('${aqseq_mode}'): ${success_aqseq ? 'Success' : 'Failed'}`);

  // PLACEHOLDER: Set whether imaginary data is negative
  // true = imaginary data along indirect dimension is negative
  const imaginary_is_negative = true; // PLACEHOLDER
  fid2dInstance.set_negative(imaginary_is_negative);
  console.log(`set_negative(${imaginary_is_negative}): Done`);

  // PLACEHOLDER: Process only first spectrum in pseudo-3D
  // Set to true to limit processing to first spectrum
  const process_first_only = false; // PLACEHOLDER
  fid2dInstance.set_first_only(process_first_only);
  console.log(`set_first_only(${process_first_only}): Done`);

  // PLACEHOLDER: Set zero filling factors
  // Typical values: 1, 2, 4, 8, 16 (multiplication factor)
  const zf_direct = 2;        // PLACEHOLDER: Zero filling for direct dimension
  const zf_indirect = 2;      // PLACEHOLDER: Zero filling for indirect dimension
  const success_zf = fid2dInstance.run_zf(zf_direct, zf_indirect);
  console.log(`run_zf(${zf_direct}, ${zf_indirect}): ${success_zf ? 'Success' : 'Failed'}`);

  // OPTIONAL: Set up apodization windows
  // Note: This typically requires creating apodization objects via embind
  // For now, this is a placeholder - implementation depends on apodization class binding
  console.log("set_up_apodization: [Typically set up via class method - not in this basic example]");
}

// ============================================================================
// PART 4: WORKFLOW A - BRUKER FILE-BASED APPROACH (Traditional)
// ============================================================================

/**
 * WORKFLOW A: Read Bruker data from files (traditional approach)
 * This approach reads directly from file paths
 */
async function workflowA_BrukerFileBased(fid2dInstance, fileInputsObj) {
  console.log("\n========== WORKFLOW A: Bruker File-Based Approach ==========");
  
  // PLACEHOLDER: Provide file paths for Bruker data
  const pulseProgram = fileInputsObj.pulseProgram;      // e.g., "pulse_program.txt" or ""
  const acqusFile = fileInputsObj.acqusFile;            // e.g., "acqus" (direct dimension params)
  const acqus2File = fileInputsObj.acqus2File;          // e.g., "acqu2s" (indirect dimension params)
  const fidFiles = fileInputsObj.fidFiles;              // e.g., ["fid"] (can be multiple files for pseudo-3D)

  // PLACEHOLDER: Call read_bruker_files
  // This function reads raw Bruker FID data from files
  console.log("\n1. Reading Bruker files...");
  const success_read = fid2dInstance.read_bruker_files(
    pulseProgram,
    acqus2File,
    acqusFile,
    fidFiles
  );
  
  if (!success_read) {
    console.error("Failed to read Bruker files");
    return false;
  }
  console.log("Bruker files read successfully");

  return true;
}

// ============================================================================
// PART 5: WORKFLOW B - IN-MEMORY BRUKER APPROACH (Web-Recommended)
// ============================================================================

/**
 * WORKFLOW B: Read Bruker data from in-memory buffers (recommended for web)
 * This approach accepts file content as strings/bytes
 * More suitable for web applications where files are uploaded
 */
async function workflowB_BrukerInMemory(fid2dInstance, filesObj) {
  console.log("\n========== WORKFLOW B: In-Memory Bruker Approach (Recommended) ==========");

  // PLACEHOLDER: These would come from HTML file inputs or File API
  // Example: 
  //   filesObj.pulseProgram = await readFileAsString(pulseFile);
  //   filesObj.acqus = await readFileAsString(acqusFile);
  //   filesObj.acqu2s = await readFileAsString(acqu2sFile);
  //   filesObj.fidBytes = await readFileAsBytes(fidFile);

  const pulseProgram_content = filesObj.pulseProgram || "";  // Text content
  const acqus_content = filesObj.acqus || "";                // Text content
  const acqu2s_content = filesObj.acqu2s || "";              // Text content
  const fid_bytes = filesObj.fidBytes || [];                 // Uint8Array

  // Step 1: Read Bruker parameter files as strings
  console.log("\n1. Reading Bruker parameter files (as strings)...");
  const success_params = fid2dInstance.read_bruker_files_as_strings(
    pulseProgram_content,  // Pulse program file content
    acqus_content,         // Direct dimension acquisition parameters
    acqu2s_content         // Indirect dimension acquisition parameters
  );

  if (!success_params) {
    console.error("Failed to read Bruker parameter files");
    return false;
  }
  console.log("Bruker parameter files processed successfully");

  // Step 2: Read FID data as bytes
  console.log("\n2. Reading Bruker FID data (as byte buffer)...");
  const success_fid = fid2dInstance.read_bruker_fid_data_bytes(fid_bytes);

  if (!success_fid) {
    console.error("Failed to read FID data bytes");
    return false;
  }
  console.log("FID data processed successfully");

  return true;
}

// ============================================================================
// PART 6: WORKFLOW C - IN-MEMORY NMRPIPE APPROACH
// ============================================================================

/**
 * WORKFLOW C: Read NMRPipe binary files from in-memory buffer
 * This approach accepts nmrPipe format data as byte array
 */
async function workflowC_NMRPipeInMemory(fid2dInstance, nmrpipeBytes) {
  console.log("\n========== WORKFLOW C: In-Memory NMRPipe Approach ==========");

  // PLACEHOLDER: nmrpipeBytes would come from File API
  // Example: nmrpipeBytes = await readFileAsBytes(nmrpipeFile);

  console.log("\n1. Reading NMRPipe file from buffer...");
  const success = fid2dInstance.read_nmrpipe_file_from_buffer(nmrpipeBytes);

  if (!success) {
    console.error("Failed to read NMRPipe file from buffer");
    return false;
  }
  console.log("NMRPipe file processed successfully");

  return true;
}

// ============================================================================
// PART 7: SPECTRAL PROCESSING
// ============================================================================

/**
 * Perform full 2D spectral processing
 * Includes FFT in both dimensions and data transformations
 */
function performSpectralProcessing(fid2dInstance) {
  console.log("\n========== Spectral Processing ==========");

  // PLACEHOLDER: Set digital filter removal flags
  // false = keep digitizer filter artifacts, true = remove them
  const remove_di_direct = false;      // PLACEHOLDER: Digital filter removal (direct dim)
  const remove_di_indirect = false;    // PLACEHOLDER: Digital filter removal (indirect dim)

  console.log("\n1. Running full spectral processing...");
  const success = fid2dInstance.full_process(remove_di_direct, remove_di_indirect);

  if (!success) {
    console.error("Full process failed");
    return false;
  }
  console.log("Spectral processing completed successfully");

  return true;
}

/**
 * Apply polynomial baseline correction
 * This removes low-frequency polynomial baseline from the spectrum
 */
function applyBaselineCorrection(fid2dInstance) {
  console.log("\n========== Baseline Correction ==========");

  // PLACEHOLDER: Polynomial order
  // Typical values: 0 (constant), 1 (linear), 2 (quadratic), 3 (cubic)
  const polynomial_order = 1; // PLACEHOLDER: Change based on baseline shape

  console.log(`\n1. Applying polynomial baseline correction (order=${polynomial_order})...`);
  const success = fid2dInstance.polynorminal_baseline(polynomial_order);

  if (!success) {
    console.error("Baseline correction failed");
    return false;
  }
  console.log("Baseline correction completed successfully");

  return true;
}

/**
 * Extract a spectral region of interest
 * Useful for zooming into a specific frequency range
 */
function extractSpectralRegion(fid2dInstance) {
  console.log("\n========== Spectral Region Extraction ==========");

  // PLACEHOLDER: Normalized frequency range [0, 1]
  // from=0.2, to=0.8 means extract the middle 60% of the spectrum
  const from = 0.2;  // PLACEHOLDER: Start position (0.0 to 1.0)
  const to = 0.8;    // PLACEHOLDER: End position (0.0 to 1.0)

  console.log(`\n1. Extracting spectral region from ${from} to ${to}...`);
  const success = fid2dInstance.extract_region(from, to);

  if (!success) {
    console.error("Region extraction failed");
    return false;
  }
  console.log(`Region extracted successfully [${from}, ${to}]`);

  return true;
}

// ============================================================================
// PART 8: AUTOMATIC PHASE CORRECTION (spectrum_phasing class)
// ============================================================================

/**
 * Apply automatic phase correction using spectrum_phasing
 * This requires a spectrum_phasing instance (inherits from fid_2d)
 */
function applyAutomaticPhaseCorrection(phasingInstance) {
  console.log("\n========== Automatic Phase Correction ==========");

  console.log("\n1. Running entropy-based automatic phase correction...");
  const success = phasingInstance.auto_phase_correction_v2();

  if (!success) {
    console.error("Auto phase correction failed");
    return false;
  }
  console.log("Automatic phase correction completed successfully");

  return true;
}

// ============================================================================
// PART 9: OUTPUT/EXPORT FUNCTIONS
// ============================================================================

/**
 * Output metadata to JSON format
 */
function exportMetadataJSON(fid2dInstance, filename) {
  console.log("\n========== JSON Metadata Export ==========");

  console.log(`\n1. Writing metadata to JSON: ${filename}...`);
  const success = fid2dInstance.write_json(filename);

  if (!success) {
    console.error("JSON write failed");
    return false;
  }
  console.log("JSON metadata written successfully");

  return true;
}

/**
 * Output pseudo-3D metadata to JSON format
 * (For experiments with multiple spectra/pseudo-3D data)
 */
function exportPseudo3DMetadataJSON(fid2dInstance, filename) {
  console.log("\n========== Pseudo-3D JSON Metadata Export ==========");

  console.log(`\n1. Writing pseudo-3D metadata to JSON: ${filename}...`);
  const success = fid2dInstance.write_pseudo3d_json(filename);

  if (!success) {
    console.error("Pseudo-3D JSON write failed");
    return false;
  }
  console.log("Pseudo-3D metadata written successfully");

  return true;
}

/**
 * Export processed spectrum to NMRPipe ft2 format (file-based)
 */
function exportNMRPipeFT2(fid2dInstance, filename) {
  console.log("\n========== NMRPipe FT2 Export ==========");

  console.log(`\n1. Writing spectrum to NMRPipe ft2 format: ${filename}...`);
  const success = fid2dInstance.write_nmrpipe_ft2(filename);

  if (!success) {
    console.error("NMRPipe ft2 write failed");
    return false;
  }
  console.log("NMRPipe ft2 file written successfully");

  return true;
}

/**
 * Export processed spectrum to buffer (in-memory, recommended for web)
 * Returns byte array that can be downloaded or transmitted
 */
function exportNMRPipeFT2ToBuffer(fid2dInstance) {
  console.log("\n========== NMRPipe FT2 Export to Buffer ==========");

  console.log("\n1. Exporting spectrum to byte buffer...");
  
  const address = fid2dInstance.write_nmrpipe_ft2_to_buffer();

  if (address === 0) {
    console.error("Buffer export failed");
    return null;
  }
  
  const size_in_float32 = fid2dInstance.get_ft2_size_in_float32();
  console.log(`Spectrum exported to buffer successfully (size: ${size_in_float32} float32 elements)`);
  
  // Convert to Uint8Array for download or transmission
  const total_bytes = size_in_float32 * 4;
  const byteArray = new Uint8Array(Module.HEAPF32.buffer, address, total_bytes).slice();
  
  return byteArray;
}

/**
 * Export phase correction results
 * Includes p0_direct, p1_direct, p0_indirect, p1_indirect values
 */
function exportPhaseCorrection(phasingInstance, filename) {
  console.log("\n========== Phase Correction Export ==========");

  console.log(`\n1. Saving phase correction results to: ${filename}...`);
  const success = phasingInstance.save_phase_correction_result(filename);

  if (!success) {
    console.error("Phase correction save failed");
    return false;
  }
  console.log("Phase correction results saved successfully");

  return true;
}

// ============================================================================
// PART 10: COMPLETE WORKFLOW EXAMPLES
// ============================================================================

/**
 * COMPLETE EXAMPLE 1: Bruker in-memory workflow with automatic phase correction
 * Recommended workflow for web applications
 */
async function completeWorkflowExample1_BrukerWithPhasing() {
  console.log("\n" + "=".repeat(70));
  console.log("COMPLETE WORKFLOW EXAMPLE 1: Bruker + Automatic Phase Correction");
  console.log("=".repeat(70));

  // Initialize WASM module
  await initializeWasmModule();

  // =======================================================================
  // STEP 1: Create fid_2d and spectrum_phasing instances
  // =======================================================================
  console.log("\nSTEP 1: Creating WASM instances...");
  
  // NOTE: Constructor names depend on Emscripten bindings
  // Typically: new Module.fid_2d() and new Module.spectrum_phasing()
  const fid2d = new Module.fid_2d();
  const phasing = new Module.spectrum_phasing();
  console.log("Instances created successfully");

  // =======================================================================
  // STEP 2: Prepare file contents (PLACEHOLDERS - from HTML file inputs)
  // =======================================================================
  console.log("\nSTEP 2: Preparing file contents...");
  
  const filesObj = {
    pulseProgram: "PLACEHOLDER_PULSE_PROGRAM_TEXT",  // From uploaded pulse program file
    acqus: "PLACEHOLDER_ACQUS_TEXT",                 // From uploaded acqus file
    acqu2s: "PLACEHOLDER_ACQU2S_TEXT",               // From uploaded acqu2s file
    fidBytes: new Uint8Array([])                     // From uploaded ser or fid file
  };
  console.log("Note: In real app, load these from file inputs");

  // =======================================================================
  // STEP 3: Read Bruker data (in-memory)
  // =======================================================================
  console.log("\nSTEP 3: Reading Bruker data...");
  const success_read = await workflowB_BrukerInMemory(fid2d, filesObj);
  if (!success_read) { console.error("Failed to read data"); return; }

  // =======================================================================
  // STEP 4: Setup processing parameters
  // =======================================================================
  console.log("\nSTEP 4: Setting up processing parameters...");
  setupProcessingParameters(fid2d);

  // =======================================================================
  // STEP 5: Perform spectral processing
  // =======================================================================
  console.log("\nSTEP 5: Processing spectrum...");
  const success_process = performSpectralProcessing(fid2d);
  if (!success_process) { console.error("Processing failed"); return; }

  // =======================================================================
  // STEP 6: Apply baseline correction (optional)
  // =======================================================================
  console.log("\nSTEP 6: Applying baseline correction...");
  const success_baseline = applyBaselineCorrection(fid2d);
  if (!success_baseline) { console.error("Baseline correction failed"); return; }

  // =======================================================================
  // STEP 7: Apply automatic phase correction
  // =======================================================================
  console.log("\nSTEP 7: Applying automatic phase correction...");
  
  // Use phasing object (which inherits from fid_2d)
  const success_phasing = applyAutomaticPhaseCorrection(phasing);
  if (!success_phasing) { console.error("Phase correction failed"); return; }

  // =======================================================================
  // STEP 8: Export results
  // =======================================================================
  console.log("\nSTEP 8: Exporting results...");

  // Export spectrum to buffer (for download)
  const spectrum_buffer = exportNMRPipeFT2ToBuffer(phasing);
  if (spectrum_buffer) {
    const blob = byteArrayToBlob(spectrum_buffer);
    // PLACEHOLDER: Trigger download in browser
    // downloadFile(blob, 'processed_spectrum.ft2');
    console.log("Spectrum buffer ready for download (size: " + spectrum_buffer.length + " bytes)");
  }

  // Export phase correction results
  exportPhaseCorrection(phasing, "phase_correction_results.txt");

  // Cleanup
  fid2d.delete();
  phasing.delete();

  console.log("\n" + "=".repeat(70));
  console.log("WORKFLOW COMPLETED SUCCESSFULLY");
  console.log("=".repeat(70));
}

/**
 * COMPLETE EXAMPLE 2: NMRPipe file workflow with region extraction
 * For processing existing NMRPipe binary data
 */
async function completeWorkflowExample2_NMRPipeWithRegion() {
  console.log("\n" + "=".repeat(70));
  console.log("COMPLETE WORKFLOW EXAMPLE 2: NMRPipe + Region Extraction");
  console.log("=".repeat(70));

  // Initialize WASM module
  await initializeWasmModule();

  // =======================================================================
  // STEP 1: Create instances
  // =======================================================================
  console.log("\nSTEP 1: Creating WASM instances...");
  const fid2d = new Module.fid_2d();
  console.log("Instance created successfully");

  // =======================================================================
  // STEP 2: Read NMRPipe file
  // =======================================================================
  console.log("\nSTEP 2: Reading NMRPipe file...");
  
  // PLACEHOLDER: Load nmrPipe file as bytes
  const nmrpipe_bytes = new Uint8Array([]); // From file input
  
  const success_read = await workflowC_NMRPipeInMemory(fid2d, nmrpipe_bytes);
  if (!success_read) { console.error("Failed to read NMRPipe file"); return; }

  // =======================================================================
  // STEP 3: Extract region of interest
  // =======================================================================
  console.log("\nSTEP 3: Extracting spectral region...");
  const success_extract = extractSpectralRegion(fid2d);
  if (!success_extract) { console.error("Region extraction failed"); return; }

  // =======================================================================
  // STEP 4: Export results
  // =======================================================================
  console.log("\nSTEP 4: Exporting results...");
  
  const spectrum_buffer = exportNMRPipeFT2ToBuffer(fid2d);
  if (spectrum_buffer) {
    console.log("Extracted region ready for download (size: " + spectrum_buffer.length + " bytes)");
  }

  // Cleanup
  fid2d.delete();

  console.log("\n" + "=".repeat(70));
  console.log("WORKFLOW COMPLETED SUCCESSFULLY");
  console.log("=".repeat(70));
}

// ============================================================================
// PART 11: HTML INTEGRATION EXAMPLE
// ============================================================================

/**
 * Example HTML form handler for file upload
 * This shows how to integrate with a web form
 */
async function handleFormSubmit(event) {
  event.preventDefault();

  console.log("\n" + "=".repeat(70));
  console.log("PROCESSING USER FILES");
  console.log("=".repeat(70));

  try {
    // PLACEHOLDER: Get files from form inputs
    // const pulseFile = document.getElementById('pulse-input').files[0];
    // const acqusFile = document.getElementById('acqus-input').files[0];
    // const acqu2sFile = document.getElementById('acqu2s-input').files[0];
    // const fidFile = document.getElementById('fid-input').files[0];

    // Read files to memory
    const filesObj = {
      // pulseProgram: await readFileAsString(pulseFile),
      // acqus: await readFileAsString(acqusFile),
      // acqu2s: await readFileAsString(acqu2sFile),
      // fidBytes: await readFileAsBytes(fidFile),
    };

    // Run workflow
    await completeWorkflowExample1_BrukerWithPhasing();

  } catch (error) {
    console.error("Error during processing:", error);
  }
}

// ============================================================================
// PART 12: SUMMARY OF ALL BOUND FUNCTIONS
// ============================================================================

/**
 * REFERENCE: Summary of all available WASM-bound functions
 * 
 * fid_2d class methods:
 * ─────────────────────────────────────────────────────────────
 * • set_aqseq(mode: string) → bool
 *   Set acquisition sequence mode ("321" or "312")
 * 
 * • set_negative(flag: bool) → void
 *   Set whether imaginary data is negative
 * 
 * • set_first_only(flag: bool) → void
 *   Process only first spectrum in pseudo-3D
 * 
 * • set_up_apodization(a1: ptr, a2: ptr) → bool
 *   Setup apodization windows for both dimensions
 * 
 * • run_zf(zf_direct: int, zf_indirect: int) → bool
 *   Set zero filling factors (1, 2, 4, 8, etc.)
 * 
 * • extract_region(from: double, to: double) → bool
 *   Extract spectral region [from, to] in normalized coords
 * 
 * • read_bruker_files(pulse_prog, acqu2s, acqus, fid_files) → bool
 *   Read Bruker data from file paths (traditional)
 * 
 * • read_bruker_files_as_strings(pulse, acqus, acqu2s) → bool
 *   Read Bruker parameter files as text strings (in-memory)
 * 
 * • read_bruker_fid_data_bytes(fid_bytes: uint8[]) → bool
 *   Read Bruker FID data as byte array (in-memory)
 * 
 * • read_nmrpipe_file_from_buffer(nmrpipe_bytes: uint8[]) → bool
 *   Read NMRPipe binary format from byte array
 * 
 * • full_process(remove_di_direct?, remove_di_indirect?) → bool
 *   Perform complete FFT in both dimensions
 * 
 * • polynorminal_baseline(order: int) → bool
 *   Apply polynomial baseline correction (order 0-3)
 * 
 * • write_json(filename: string) → bool
 *   Export metadata to JSON file
 * 
 * • write_pseudo3d_json(filename: string) → bool
 *   Export pseudo-3D metadata to JSON file
 * 
 * • write_nmrpipe_ft2(filename: string) → bool
 *   Export spectrum to NMRPipe ft2 file
 * 
 * • write_nmrpipe_ft2_to_buffer() → uintptr_t (address)
 *   Export spectrum to byte buffer (in-memory)
 * 
 * • get_ft2_size_in_float32() → int
 *   Get size of the FT2 buffer in float32 elements
 * 
 * spectrum_phasing class methods (inherits from fid_2d):
 * ─────────────────────────────────────────────────────────────
 * • auto_phase_correction_v2() → bool
 *   Apply entropy-based automatic phase correction
 * 
 * • save_phase_correction_result(filename: string) → bool
 *   Save phase correction values to file
 */

// ============================================================================
// ENTRY POINT
// ============================================================================

console.log(`
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  DEEP-Picker WebAssembly Binding - Complete Workflow Example  ║
║                                                                ║
║  This file demonstrates all WebAssembly-bound functions       ║
║  for 2D NMR spectral processing and automatic phase           ║
║  correction.                                                  ║
║                                                                ║
║  See the function definitions and examples above for:         ║
║  - How to call each function                                  ║
║  - Parameter meanings and acceptable values                   ║
║  - Typical workflow sequence                                  ║
║  - File I/O handling (both file-based and in-memory)          ║
║                                                                ║
║  Main entry points:                                           ║
║  - completeWorkflowExample1_BrukerWithPhasing()               ║
║  - completeWorkflowExample2_NMRPipeWithRegion()               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
`);

// Uncomment to run example:
// completeWorkflowExample1_BrukerWithPhasing();

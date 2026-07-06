/**
 * Short documentation and workflow for 3D peak picking and fitting in JavaScript (WASM)
 */

async function run3DPickAndFit(wasmModule, ft3Buffer) {
    // 1. Initialize the spectrum_fit_3d object
    const app = new wasmModule.spectrum_fit_3d();

    // 2. Configure noise and scales
    // set_noise_scales(noise_level, scale1, scale2)
    // If noise_level is 0.0, it will be estimated from the spectrum.
    app.set_noise_scales(0.0, 6.0, 3.5);
    app.set_noise_level_for_nus(false);

    // 3. Load the stacked .ft3 file data from a buffer
    // read_ft3_from_buffer_raw(pointer, size)
    const size = ft3Buffer.length;
    const ptr = wasmModule._malloc(size);
    wasmModule.HEAPU8.set(ft3Buffer, ptr);
    
    const readOk = app.read_ft3_from_buffer_raw(ptr, size);
    wasmModule._free(ptr);
    
    if (!readOk) {
        console.error("Failed to read spectrum data");
        return;
    }

    // 4. Run peak picking
    // Returns true on success, saves peaks internally.
    app.peak_picking("3d_peaks.txt");

    // 5. Partition signal regions
    app.partition_signal_regions("3d_signal_regions.txt");

    // 6. Run iterative fitting on top-N partitions
    // top_n = 50 (fit top 50 strongest partitions)
    app.iterative_fit_all_partitions(50);

    // 7. Retrieve the results as a formatted string
    const resultString = app.get_fitted_peaks_string();
    
    console.log("Fitted Peaks Output:\n", resultString);
    
    // Clean up
    app.delete();
}

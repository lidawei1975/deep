

/**
 * Define a spectrum class to hold all spectrum information
 * Example of levels_length, polygon_length and points
 * levels_length=[0,3,5] means there are 2 levels, first level has 3 polygons, second level has 2 polygons: total 5 polygons
 * polygon_length=[0,3,6,8,14,16] means there are 5 polygons, first polygon has 3 points,
 * second polygon has 3 points, third polygon has 2 points, fourth polygon has 6 points, fifth polygon has 2 points: total 16 points
 * points=[x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14,x15,y15,x16,y16]
 * 
 * In case we have more than one contour plots (overlay of two contour plots),
 * we will have overlays= [0, 2, 4] means there are 2 overlayed contour plots
 * first plot has 2 levels, second plot has 2 levels in the levels_length array
 * if overlays =[] means all levels are in one plot (no overlay), this is the default, equal to [0, levels_length.length]
 */
class spectrum {
    constructor() {
        this.spectrum_format = "ft2"; //ft2 is the default format
        this.header = new Float32Array(512); //header of the spectrum, 512 float32 numbers
        this.raw_data = new Float32Array(0); //raw data, real real
        this.raw_data_ri = new Float32Array(0); //raw data for real (along indirect dimension) and imaginary (along indirect dimension) part
        this.raw_data_ir = new Float32Array(0); //raw data for imaginary (along indirect dimension) and real (along indirect dimension) part
        this.raw_data_ii = new Float32Array(0); //raw data for imaginary (along indirect dimension) and imaginary (along indirect dimension) part
        this.noise_level = 0.001; //noise level of the input spectrum
        this.levels = [0.001, 0.002, 0.003]; //levels of the contour plot
        this.spectral_max = Number.MAX_VALUE; //maximum value of the spectrum
        this.n_direct = 4096; //size of direct dimension of the input spectrum. integer
        this.n_indirect = 1204; //size of indirect dimension of the input spectrum. integer
        this.x_ppm_start = 12.0; //start ppm of direct dimension
        this.x_ppm_width = 12.0; //width of direct dimension
        this.x_ppm_step = -12.0 / 4096; //step of direct dimension
        this.y_ppm_start = 120.0; //start ppm of indirect dimension
        this.y_ppm_width = 120.0; //width of indirect dimension
        this.y_ppm_step = -120.0 / 1024; //step of indirect dimension
        this.x_ppm_ref = 0.0; //reference ppm of direct dimension
        this.y_ppm_ref = 0.0; //reference ppm of indirect dimension
        this.frq1 = 850.0; //spectrometer frequency of direct dimension
        this.frq2 = 80.0; //spectrometer frequency of indirect dimension
        this.picked_peaks_object = new cpeaks(); //picked peaks object
        this.fitted_peaks_object = new cpeaks(); //fitted peaks object
        /**
         * spectrum origin: 
         * -4: unknown,
         * -3" removed spectrum
         * -2: experimental spectrum from fid, 
         * -1: experimental spectrum uploaded,  
         * n(n>=0 and n<10000): reconstructed from experimental spectrum n
         * n(n>=10000): pseudo 3D spectrum, whose first plane spectrum is n-10000
         */
        this.spectrum_origin = -4;

        this.reconstructed_indices = []; //array of reconstructed spectra indices.

        this.spectrum_index = -1; //index of the spectrum in the hsqc_spectra array. integer and >=0

        /**
         * Default median sigmax, sigmay, gammax, gammay
         */
        this.median_sigmax = 1.0;
        this.median_sigmay = 1.0;
        this.median_gammax = 1.0;
        this.median_gammay = 1.0;

        /**
         * Control the display of the spectrum
         */
        this.visible = true; //visible or not

        /**
         * fid process parameters is only valid when spectrum_origin is -2 (from fid)
         * and it is the first plane (of a pseudo 3D spectrum if it is a pseudo 3D fid)
         */
        this.fid_process_parameters = null;
        /**
         * If the spectrum is 1st plane of a pseudo 3D spectrum, 
         * pseudo3d_children will hold the indices of children spectra (other planes)
         */
        this.pseudo3d_children = [];

        /**
         * By default, there is only one spectrum
         */
        this.nspect = 1; //number of spectra, default is 1

    };

    /**
     * Function to create a shallow copy of the spectrum object
     * Keep all the properties, except header, raw_data, raw_data_ri, raw_data_ir, raw_data_ii (Float32Array)
     * And add new properties: header_length, raw_data_length, raw_data_ri_length, raw_data_ir_length, raw_data_ii_length
     * which are the length of the corresponding Float32Array
     */
    create_shallow_copy_wo_float32() {
        let new_spectrum = new Object();
        for (var key in this) {
            if (this.hasOwnProperty(key)) {
                if (key === "header" || key === "raw_data" || key === "raw_data_ri" || key === "raw_data_ir" || key === "raw_data_ii") {
                    new_spectrum[key + "_length"] = this[key].length;
                }
                else {
                    new_spectrum[key] = this[key];
                }
            }
        }
        return new_spectrum;
    };


    update_x_ppm_ref(x_ppm_ref) {
        let delta_ppm = x_ppm_ref - this.x_ppm_ref;
        this.x_ppm_ref = x_ppm_ref;
        this.header[101] += delta_ppm * this.frq1;
        this.ref1 = this.header[101];

        /**
        * Update peaks as well if there are any
        */
        if (this.picked_peaks_object != null) {
            this.picked_peaks_object.update_x_ppm_ref(delta_ppm);
        }
        if (this.fitted_peaks_object != null) {
            this.fitted_peaks_object.update_x_ppm_ref(delta_ppm);
        }
    };

    update_y_ppm_ref(y_ppm_ref) {
        let delta_ppm = y_ppm_ref - this.y_ppm_ref;
        this.y_ppm_ref = y_ppm_ref;
        this.header[249] += delta_ppm * this.frq2;
        this.ref2 = this.header[249];

        /**
         * Update peaks as well if there are any
         */
        if (this.picked_peaks_object != null) {
            this.picked_peaks_object.update_y_ppm_ref(delta_ppm);
        }
        if (this.fitted_peaks_object != null) {
            this.fitted_peaks_object.update_y_ppm_ref(delta_ppm);
        }
    };


    /**
     * Process the raw file data of a 2D FT spectrum (.txt from Topspin totxt command)
     * @param {*} file_text: raw file data as a string
     * @param {*} file_name: name of the file
     * @param {*} field_strength: field strength of the spectrometer
     * Note: for topspin file, spectrum_origin is always -1 (user uploaded frequency file)
     *  
     */
    process_topspin_file(file_text, file_name, field_strength) {

        this.spectrum_format = "Topspin"
        this.spectrum_origin = -1;
        this.filename = file_name;


        let lines = file_text.split(/\r\n|\n/);

        /**
         * Loop all lines
         * 1. find line contained both "F1LEFT" and "F1RIGHT" to get the x_ppm_start and x_ppm_width
         * 2. find line contained both "F2LEFT" and "F2RIGHT" to get the y_ppm_start and y_ppm_width
         * 3. find line contained NROWS to get the n_indirect
         * 4. find line contained NCOLS to get the n_direct
         * Quit the loop if all 4 values are found. 
         */
        let number_of_info_retrieved = 0;
        let data_start = 0;
        this.n_direct = undefined;
        this.n_indirect = undefined;
        this.x_ppm_start = undefined;
        this.y_ppm_start = undefined;
        this.x_ppm_width = undefined;
        this.y_ppm_width = undefined;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes("F1LEFT") && lines[i].includes("F1RIGHT")) {
                /**
                 * Separate the line by any number of space(s) and get the location of F1LEFT and F1RIGHT
                 */
                let temp = lines[i].split(/\s+/); //split by any number of space(s)
                let left = temp.indexOf("F1LEFT");
                let right = temp.indexOf("F1RIGHT");
                this.y_ppm_start = parseFloat(temp[left + 2]);
                this.y_ppm_width = parseFloat(temp[right + 2]) - parseFloat(temp[left + 2]);
                number_of_info_retrieved++;
            }
            if (lines[i].includes("F2LEFT") && lines[i].includes("F2RIGHT")) {
                let temp = lines[i].split(/\s+/);
                let left = temp.indexOf("F2LEFT");
                let right = temp.indexOf("F2RIGHT");
                this.x_ppm_start = parseFloat(temp[left + 2]);
                this.x_ppm_width = parseFloat(temp[right + 2]) - parseFloat(temp[left + 2]);
                number_of_info_retrieved++;
            }
            if (lines[i].includes("NROWS")) {
                let temp = lines[i].split(/\s+/);
                let location = temp.indexOf("NROWS");
                this.n_indirect = parseInt(temp[location + 2]);
                number_of_info_retrieved++;
            }
            if (lines[i].includes("NCOLS")) {
                let temp = lines[i].split(/\s+/);
                let location = temp.indexOf("NCOLS");
                this.n_direct = parseInt(temp[location + 2]);
                number_of_info_retrieved++;
            }
            if (number_of_info_retrieved >= 4 && this.n_direct !== undefined && this.n_indirect !== undefined && this.x_ppm_start !== undefined && this.y_ppm_start !== undefined) {
                number_of_info_retrieved = -1; //this is a flag to show that we have found all 4 values
                data_start = i + 1;
                break;
            }
        }

        if (number_of_info_retrieved !== -1) {
            this.error = "Cannot find all necessary information in the file";
            return result;
        }

        /**
         * Loop remaining lines to get the data
         */
        this.raw_data = new Float32Array(this.n_direct * this.n_indirect);
        let col_counter = 0;
        let row_counter = 0;
        for (let i = data_start; i < lines.length; i++) {
            /**
             * If line start with "# row = ", this is a start of a new row, we reset column counter to 0
             * and get row counter from the line immediately after "# row = "
             */
            if (lines[i].startsWith("# row = ")) {
                col_counter = 0;
                row_counter = parseInt(lines[i].substring(8));
                continue;
            }
            /**
             * Other lines start with # are comments, we skip them
             * and also skip empty lines
             */
            else if (lines[i].startsWith("#") || lines[i] === "") {
                continue;
            }
            /**
             * Others are data line, only one number per line in Topspin totxt format
             */
            else {
                this.raw_data[row_counter * this.n_direct + col_counter] = parseFloat(lines[i]);
                col_counter++;
            }
        }

        /**
         * At the end, we should have col_counter = this.n_direct and row_counter = this.n_indirect-1
         */
        if (col_counter !== this.n_direct || row_counter !== this.n_indirect - 1) {
            this.error = "Cannot find all data in the file";
            return result;
        }

        /**
         * Make x_ppm_width and y_ppm_width positive.
         * Set this.x_ppm_step and this.y_ppm_step. Note that the ppm_step is negative
         */
        this.x_ppm_width = Math.abs(this.x_ppm_width);
        this.y_ppm_width = Math.abs(this.y_ppm_width);
        this.x_ppm_step = -this.x_ppm_width / this.n_direct;
        this.y_ppm_step = -this.y_ppm_width / this.n_indirect;

        /**
         * Set default ref to 0
         */
        this.x_ppm_ref = 0;
        this.y_ppm_ref = 0;

        /**
         * Noise level, spectral_max and min, projection, levels, negative_levels code.
         */
        this.process_spectrum_common_task();

        /**
         * Setup a fake nmrPipe header, so that we can use the same code to 
         * run DEEP_Picker, VoigtFit, etc.
         */
        this.header = new Float32Array(512); //empty array
        this.header[0] = 0.0; //magic number for nmrPipe header

        this.header[99] = this.n_direct; //size of direct dimension of the input spectrum
        this.header[219] = this.n_indirect; //size of indirect dimension of the input spectrum
        this.header[221] = 0; // not transposed
        this.header[56] = 1; //real data along both dimensions
        this.header[55] = 1; //real data along both dimensions

        this.header[24] = 2; //direct dimension is the second dimension
        this.header[25] = 1; //indirect dimension is the first dimension


        this.header[220] = 1; // frequency domain data (1 for frequency domain, 0 for time domain), indirect dimension
        this.header[222] = 1; // frequency domain data (1 for frequency domain, 0 for time domain), direct dimension

        /**
         * Suppose filed strength is 850 along direct dimension
         * and indirection dimension obs is 85.0
         */
        this.header[119] = field_strength; //observed frequency of direct dimension
        this.header[218] = 85.0; //observed frequency of indirect dimension

        this.header[100] = this.x_ppm_width * field_strength; //spectral width of direct dimension
        this.header[229] = this.y_ppm_width * 85.0; //spectral width of indirect dimension

        /**
         * Per topspin convention, First point is inclusive, last point is exclusive in [ppm_start, ppm_start+ppm_width)]
         * Notice x_ppm_step and y_ppm_step are negative
         */
        this.header[101] = (this.x_ppm_start - this.x_ppm_width - this.x_ppm_step) * field_strength; //origin of direct dimension (last point frq in Hz)
        this.header[249] = (this.y_ppm_start - this.y_ppm_width - this.y_ppm_step) * 85.0; //origin of indirect dimension (last point frq in Hz)
        this.ref1 = this.header[101];
        this.ref2 = this.header[249];

        /**
         * We did not fill carrier frequency, because we do not need it for DEEP_Picker, VoigtFit, etc.
         * They are needed in fid processing, not in spectrum processing.
         */

        this.frq1 = this.header[119];
        this.frq2 = this.header[218];

    };


    /**
     * Process the raw file data of a 2D FT spectrum (nmrPipe .ft2 format)
     * @param {arrayBuffer} arrayBuffer: raw file data
     * @param {string} file_name: name of the file
     * @param {string} spectrum_origin: index to origin of the spectrum for reconstructed spectra, -1 for ft2, -2 for FID and -3 for removed spectra
     * @returns hsqc_spectra object
     */
    process_ft_file(arrayBuffer, file_name, spectrum_origin) {


        this.spectrum_format = "ft2";

        this.spectrum_origin = spectrum_origin;

        this.header = new Float32Array(arrayBuffer, 0, 512);

        this.n_indirect = this.header[219]; //size of indirect dimension of the input spectrum
        this.n_direct = this.header[99]; //size of direct dimension of the input spectrum

        this.tp = this.header[221];

        /**
         * if transposed, set this.error and return
         */
        if (this.tp !== 0) {
            this.error = "Transposed data, please un-transpose the data before loading";
            return result;
        }

        /**
         * Datatype of the direct and indirect dimension
         * 0: complex
         * 1: real
         */
        this.datatype_direct = this.header[56];
        this.datatype_indirect = this.header[55];

        /**
         * this.datatype_direct: 1 means real, 0 means complex
         * this.datatype_indirect: 1 means real, 0 means complex
         */
        if (this.datatype_direct == 0 && this.datatype_indirect == 0) {
            console.log("Complex data along both dimensions.");
            this.n_indirect /= 2; //complex data along both dimensions, per nmrPipe format, so we divide by 2
        }
        else if (this.datatype_direct == 0 && this.datatype_indirect == 1) {
            console.log("Complex data along direct dimension, real data along indirect dimension.");
        }
        else if (this.datatype_direct == 1 && this.datatype_indirect == 0) {
            console.log("Complex data along indirect dimension, real data along direct dimension.");
        }
        else if (this.datatype_direct == 1 && this.datatype_indirect == 1) {
            console.log("Real data along both dimensions.");
        }
        console.log("n_direct: ", this.n_direct);
        console.log("n_indirect: ", this.n_indirect);

        this.direct_ndx = this.header[24]; //must be 2
        this.indirect_ndx = this.header[25]; //must be 1 or 3
        /**
         * direct_ndx must be 1, otherwise set error and return
         */
        if (this.direct_ndx !== 2) {
            this.error = "Direct dimension must be the second dimension";
            return result;
        }
        /**
         * indirect_ndx must be 1 or 3, otherwise set error and return
         */
        if (this.indirect_ndx !== 1 && this.indirect_ndx !== 3) {
            this.error = "Indirect dimension must be the first or third dimension";
            return result;
        }

        /**
         * this.sw, this.frq,this.ref are the spectral width, frequency and reference of the direct dimension
         * All are array of length 4
         */
        this.sw = [];
        this.frq = [];
        this.ref = [];

        this.sw[0] = this.header[229];
        this.sw[1] = this.header[100];
        this.sw[2] = this.header[11];
        this.sw[3] = this.header[29];

        this.frq[0] = this.header[218];
        this.frq[1] = this.header[119];
        this.frq[2] = this.header[10];
        this.frq[3] = this.header[28];

        this.ref[0] = this.header[249];
        this.ref[1] = this.header[101];
        this.ref[2] = this.header[12];
        this.ref[3] = this.header[30];

        /**
         * Get ppm_start, ppm_width, ppm_step for both direct and indirect dimensions
         */
        this.sw1 = this.sw[this.direct_ndx - 1];
        this.sw2 = this.sw[this.indirect_ndx - 1];
        this.frq1 = this.frq[this.direct_ndx - 1];
        this.frq2 = this.frq[this.indirect_ndx - 1];
        this.ref1 = this.ref[this.direct_ndx - 1];
        this.ref2 = this.ref[this.indirect_ndx - 1];


        this.x_ppm_start = (this.ref1 + this.sw1) / this.frq1;
        this.x_ppm_width = this.sw1 / this.frq1;
        this.y_ppm_start = (this.ref2 + this.sw2) / this.frq2;
        this.y_ppm_width = this.sw2 / this.frq2;
        this.x_ppm_step = -this.x_ppm_width / this.n_direct;
        this.y_ppm_step = -this.y_ppm_width / this.n_indirect;

        /**
         * shift by half of the bin size because the contour plot is defined by the center of each bin
         */
        this.x_ppm_start -= this.x_ppm_width / this.n_direct / 2;
        this.y_ppm_start -= this.y_ppm_width / this.n_indirect / 2;

        this.x_ppm_ref = 0.0;
        this.y_ppm_ref = 0.0;

        const spectral_data = new Float32Array(arrayBuffer);

        let data_size = arrayBuffer.byteLength / 4 - 512;

        /**
         * Let assess data_size, if it is not equal to n_direct * n_indirect * number_of_data_type, set error and return
         * number_of_data_type =1, if both direct and indirect dimensions are real
         * number_of_data_type =2, if either direct or indirect dimension is complex
         * number_of_data_type =4, if both direct and indirect dimensions are complex
         */
        let data_size_per_point = 1;
        if (this.datatype_direct === 1 && this.datatype_indirect === 1) {
            data_size_per_point = 1;
        }
        else if (this.datatype_direct === 0 && this.datatype_indirect === 1) {
            data_size_per_point = 2;
        }
        else if (this.datatype_direct === 1 && this.datatype_indirect === 0) {
            data_size_per_point = 2;
        }
        else if (this.datatype_direct === 0 && this.datatype_indirect === 0) {
            data_size_per_point = 4;
        }

        if (data_size !== this.n_direct * this.n_indirect * data_size_per_point) {
            this.error = "Data size does not match the size of the spectrum";
        }

        /**
         * this.raw_data is a Float32Array of the spectrum data, real (along indirect) real (along direct)
         * this.raw_data_ri is a Float32Array of the spectrum data, real (along indirect) imaginary (along direct)
         * this.raw_data_ir is a Float32Array of the spectrum data, imaginary (along indirect) real (along direct)
         * this.raw_data_ii is a Float32Array of the spectrum data, imaginary (along indirect) imaginary (along direct)
         * 
         * They are all row major, size is  n_indirect (rows) * n_direct (columns).
         * Order of data in arrayBuffer, after the 512 float (4 bytes) header
         * raw_data row1, raw_data_ri row1, raw_data_ir row1, raw_data_ii row1, 
         * raw_data row2, raw_data_ri row2, raw_data_ir row2, raw_data_ii row2, ...
         */
        let current_position = 512;

        /**
         * Initialize this.raw_data, this.raw_data_ri, this.raw_data_ir, this.raw_data_ii
         * If no initialization here, it will have default size of 0 (empty array)
         */
        this.raw_data = new Float32Array(this.n_indirect * this.n_direct);
        if (this.datatype_direct === 0) {
            this.raw_data_ri = new Float32Array(this.n_indirect * this.n_direct);
        }
        if (this.datatype_indirect === 0) {
            this.raw_data_ir = new Float32Array(this.n_indirect * this.n_direct);
        }
        if (this.datatype_direct === 0 && this.datatype_indirect === 0) {
            this.raw_data_ii = new Float32Array(this.n_indirect * this.n_direct);
        }


        for (let i = 0; i < this.n_indirect; i++) {
            /**
             * Copy from spectral_data (from current_position, for a length of  this.n_direct)
             * to this.raw_data (from i*this.n_direct, for a length of this.n_direct)
             */
            this.raw_data.set(spectral_data.subarray(current_position, current_position + this.n_direct), i * this.n_direct);
            current_position += this.n_direct;

            if (this.datatype_direct === 0) {
                this.raw_data_ri.set(spectral_data.subarray(current_position, current_position + this.n_direct), i * this.n_direct);
                current_position += this.n_direct;
            }
            if (this.datatype_indirect === 0) {
                this.raw_data_ir.set(spectral_data.subarray(current_position, current_position + this.n_direct), i * this.n_direct);
                current_position += this.n_direct;
            }
            if (this.datatype_direct === 0 && this.datatype_indirect === 0) {
                this.raw_data_ii.set(spectral_data.subarray(current_position, current_position + this.n_direct), i * this.n_direct);
                current_position += this.n_direct;
            }
        }

        /**
         * Keep original file name
         */
        this.filename = file_name;

        this.process_spectrum_common_task();

    };


    process_sparky_file(arrayBuffer, file_name, spectrum_origin) {

        /**
         * Get the 11th bytes and convert to integer (one byte integer). 
         */
        let n_dim = new Int8Array(arrayBuffer, 10, 1)[0];
        let n_complicity = new Int8Array(arrayBuffer, 11, 1)[0]; // 1: real, 0: complex
        // let n_version = new Int8Array(arrayBuffer, 13, 1)[0]; 

        /**
         * Make sure n_dim is 2 and n_complicity is 1 (real)
         */
        if (n_dim !== 2 || n_complicity !== 1) {
            console.log('n_dim: ', n_dim, 'n_complicity: ', n_complicity, 'n_version: ', n_version, ' . Only 2D real data is supported');
            return;
        }

        this.spectrum_format = "ucsf";

        this.spectrum_origin = spectrum_origin;

        /**
         * @IMPORTANT
         * Please notice that Sparky stores data in big endian. DataView's get* methods use big endian by default.
         */

        let indirect_parameters = new DataView(arrayBuffer, 188, 24);
        let direct_parameters = new DataView(arrayBuffer, 316, 24);


        this.n_indirect = indirect_parameters.getInt32(0);
        let indirect_tile_size = indirect_parameters.getInt32(8); //required to read spectral data below.
        this.frq2 = indirect_parameters.getFloat32(12); //observed frequency of indirect dimension MHz
        this.sw2 = indirect_parameters.getFloat32(16); //spectral width of indirect dimension, Hz
        this.center2 = indirect_parameters.getFloat32(20);  //ppm of the center of the spectrum
        this.ref2 = this.center2 * this.frq2 - this.sw2 / 2; //end frequency of the spectrum (lowest frequency)
        this.y_ppm_ref = 0.0; //reference correction (initially set to 0)
        this.y_ppm_start = this.center2 + this.sw2 / this.frq2 / 2.0; //ppm of the start of the spectrum
        this.y_ppm_width = this.sw2 / this.frq2; //width of the spectrum in ppm
        this.y_ppm_step = -this.y_ppm_width / this.n_indirect; //step size in ppm


        this.n_direct = direct_parameters.getInt32(0);
        let direct_tile_size = direct_parameters.getInt32(8);
        this.frq1 = direct_parameters.getFloat32(12); //observed frequency of direct dimension MHz
        this.sw1 = direct_parameters.getFloat32(16); //spectral width of direct dimension, Hz
        this.center1 = direct_parameters.getFloat32(20);   //ppm of the center of the spectrum
        this.ref1 = this.center1 * this.frq1 - this.sw1 / 2; //end frequency of the spectrum (lowest frequency)
        this.x_ppm_ref = 0.0; //reference correction (initially set to 0)
        this.x_ppm_start = this.center1 + this.sw1 / this.frq1 / 2.0; //ppm of the start of the spectrum
        this.x_ppm_width = this.sw1 / this.frq1; //width of the spectrum in ppm
        this.x_ppm_step = -this.x_ppm_width / this.n_direct; //step size in ppm

        this.raw_data = new Float32Array(this.n_direct * this.n_indirect);

        /**
         * Read spectral data into this.raw_data (float32array)
         * In sparky, data is stored in tiles, each tile is  direct_tile_size * indirect_tile_size
         */
        let n_direct_tile = Math.ceil(this.n_direct / direct_tile_size);
        /**
         * size of the last tile in direct dimension, set to direct_tile_size if n_direct is a multiple of direct_tile_size
         */
        let last_tile_size = this.n_direct % direct_tile_size;
        if (last_tile_size === 0) {
            last_tile_size = direct_tile_size;
        }
        let n_indirect_tile = Math.ceil(this.n_indirect / indirect_tile_size);


        let current_file_position = 436; //start of the spectral data. Aligned to 4 bytes boundary, so there is not need to use DataView

        /**
         * Because Sparky stores data in big endian, we need to swap the byte order
         * from location current_file_position to the end of the arrayBuffer
         * for all float32 data (4 bytes), swap the byte order
         */
        for (let i = 0; i < (arrayBuffer.byteLength - current_file_position) / 4; i++) {
            let temp = new Uint8Array(arrayBuffer, current_file_position + i * 4, 4);
            temp.reverse();
        }


        for (let i = 0; i < n_indirect_tile; i++) {
            for (let j = 0; j < n_direct_tile; j++) {
                for (let m = 0; m < indirect_tile_size; m++) {
                    /**
                     * Copy from arrayBuffer at location current_file_position
                     *  to this.raw_data, for a length of direct_tile_size
                     */
                    let new_data = new Float32Array(arrayBuffer, current_file_position, direct_tile_size);
                    current_file_position += direct_tile_size * 4;

                    if (i * indirect_tile_size + m < this.n_indirect) {
                        if (j === n_direct_tile - 1) {
                            /**
                             * Copy from new_data to this.raw_data, for a length of last_tile_size.
                             * at position (i*indirect_tile_size+m)*this.n_direct+j*direct_tile_size
                             */
                            this.raw_data.set(new_data.subarray(0, last_tile_size), (i * indirect_tile_size + m) * this.n_direct + j * direct_tile_size);
                        }
                        else {
                            /**
                             * Copy from new_data to this.raw_data, for a length of direct_tile_size.
                             * at position (i*indirect_tile_size+m)*this.n_direct+j*direct_tile_size
                             */
                            this.raw_data.set(new_data, (i * indirect_tile_size + m) * this.n_direct + j * direct_tile_size);
                        }
                    }
                }
            }
        }

        /**
         * Setup fake nmrPipe header, so that we can use the same code to run DEEP_Picker, etc.
         */
        this.header = new Float32Array(512); //empty array
        this.header[0] = 0.0; //magic number for nmrPipe header

        this.header[99] = this.n_direct; //size of direct dimension of the input spectrum
        this.header[219] = this.n_indirect; //size of indirect dimension of the input spectrum
        this.header[221] = 0; // not transposed
        this.header[9] = 2; // # of dimensions
        this.header[56] = 1; //real data along both dimensions
        this.header[55] = 1; //real data along both dimensions

        this.header[24] = 2; //direct dimension is the second dimension
        this.header[25] = 1; //indirect dimension is the first dimension

        this.header[220] = 1; // frequency domain data (1 for frequency domain, 0 for time domain), indirect dimension
        this.header[222] = 1; // frequency domain data (1 for frequency domain, 0 for time domain), direct dimension

        /**
         * Suppose filed strength is 850 along direct dimension
         * and indirection dimension obs is 85.0
         */
        this.header[119] = this.frq1; //observed frequency of direct dimension
        this.header[218] = this.frq2; //observed frequency of indirect dimension

        this.header[100] = this.sw1; //spectral width of direct dimension
        this.header[229] = this.sw2; //spectral width of indirect dimension

        /**
         * Per topspin convention, First point is inclusive, last point is exclusive in [ppm_start, ppm_start+ppm_width)]
         * Notice x_ppm_step and y_ppm_step are negative
         */
        this.header[101] = (this.x_ppm_start - this.x_ppm_width - this.x_ppm_step) * this.frq1; //origin of direct dimension (last point frq in Hz)
        this.header[249] = (this.y_ppm_start - this.y_ppm_width - this.y_ppm_step) * this.frq2; //origin of indirect dimension (last point frq in Hz)
        this.ref1 = this.header[101];
        this.ref2 = this.header[249];

        this.filename = file_name;

        /**
         * Noise level, spectral_max and min, projection, levels, negative_levels code.
         */
        this.process_spectrum_common_task();

    };


    /**
     * Shared code to process the spectrum data to get
     * noise_level, spectral_max, spectral_min, projection_direct, projection_indirect, projection_direct_max, projection_direct_min,
     * projection_indirect_max, projection_indirect_min, levels, negative_levels
     * @param {*} result 
     * @returns 
     */
    process_spectrum_common_task() {
        /**
         * Get median of abs(z). If data_size is > 1024*1024, we will sample 1024*1024 points by stride
         */
        this.noise_level = mathTool.estimate_noise_level(this.n_direct, this.n_indirect, this.raw_data);

        /**
         * Get max and min of z (z is sorted)
         */
        [this.spectral_max, this.spectral_min] = mathTool.find_max_min(this.raw_data);

        /**
         * If noise_level < spectral_max/1.4^30, set noise_level to spectral_max/1.4^30
         */
        if (this.noise_level < this.spectral_max / Math.pow(1.4, 30)) {
            this.noise_level = this.spectral_max / Math.pow(1.4, 30);
        }


        /**
         * raw_data is row major, size is  n_indirect (rows) * n_direct (columns).
         * Get projection of the spectrum along direct and indirect dimensions
         */
        this.projection_direct = new Float32Array(this.n_direct);
        this.projection_indirect = new Float32Array(this.n_indirect);

        for (let i = 0; i < this.n_direct; i++) {
            let sum = 0.0;
            for (let j = 0; j < this.n_indirect; j++) {
                sum += this.raw_data[j * this.n_direct + i];
            }
            this.projection_direct[i] = sum;
        }

        for (let i = 0; i < this.n_indirect; i++) {
            let sum = 0.0;
            for (let j = 0; j < this.n_direct; j++) {
                sum += this.raw_data[i * this.n_direct + j];
            }
            this.projection_indirect[i] = sum;
        }

        /**
         * Get max,min of the projection
         */
        [this.projection_direct_max, this.projection_direct_min] = mathTool.find_max_min(this.projection_direct);
        [this.projection_indirect_max, this.projection_indirect_min] = mathTool.find_max_min(this.projection_indirect);

        /**
         * In case of reconstructed spectrum from fitting or from NUS, noise_level is usually 0.
         * In that case, we define noise_level as spectral_max/power(1.5,40)
         */
        if (this.noise_level <= Number.MIN_VALUE) {
            this.noise_level = this.spectral_max / Math.pow(1.5, 40);
        }

        /**
         * Calculate positive contour levels 
         */
        this.levels = new Array(40);
        this.levels[0] = 2.25 * 5.5 * this.noise_level;
        for (let i = 1; i < this.levels.length; i++) {
            this.levels[i] = 1.5 * this.levels[i - 1];
            if (this.levels[i] > this.spectral_max) {
                this.levels = this.levels.slice(0, i + 1);
                break;
            }
        }
        this.positive_contour_type = "logarithmic";

        /**
         * Calculate negative contour levels
         */
        this.negative_levels = new Array(40);
        this.negative_levels[0] = -2.25 * 5.5 * this.noise_level;
        for (let i = 1; i < this.negative_levels.length; i++) {
            this.negative_levels[i] = 1.5 * this.negative_levels[i - 1];
            if (this.negative_levels[i] < this.spectral_min) {
                this.negative_levels = this.negative_levels.slice(0, i + 1);
                break;
            }
        }
        this.negative_contour_type = "logarithmic";
    };


    run_peak_fitting(peak_fitting_parameters) {

        /**
         * Clear current fitted peak object
         */
        this.fitted_peaks_object = null;

        /**
         * 0  gaussian_type,
         * 1  voigt_type,
         * 2  exact_type,
         * 3  voigt_lorentz_type, // voigt along x and lorentzian along y
         * 4  null_type //this is an error!
         */


        this.user_scale = peak_fitting_parameters.user_scale ? peak_fitting_parameters.user_scale : 5.5;
        this.user_scale2 = peak_fitting_parameters.user_scale2 ? peak_fitting_parameters.user_scale2 : 3.0;
        this.peak_shape = peak_fitting_parameters.peak_shape ? peak_fitting_parameters.peak_shape : 0; //default is gaussian
        this.too_near_cutoff = peak_fitting_parameters.too_near_cutoff ? peak_fitting_parameters.too_near_cutoff : 0.01; //default is 0.1 ppm
        this.removal_cutoff = peak_fitting_parameters.removal_cutoff ? peak_fitting_parameters.removal_cutoff : 0.01; //default is 0.1 ppm
        this.maxround = peak_fitting_parameters.maxround ? peak_fitting_parameters.maxrouns : 10; //default is 10 rounds

        let x_ppm = this.picked_peaks_object.get_column_by_header("X_PPM");
        let y_ppm = this.picked_peaks_object.get_column_by_header("Y_PPM");
        this.p_intensity = this.picked_peaks_object.get_column_by_header("HEIGHT");
        this.peak_assignments = this.picked_peaks_object.get_column_by_header("ASS");

        this.sigmax = this.picked_peaks_object.get_column_by_header("XW").map((x) => x / 2.355);
        this.sigmay = this.picked_peaks_object.get_column_by_header("YW").map((y) => y / 2.355);


        /**
         * User this.x_ppm_start and this.x_ppm_step to convert x_ppm to index
         * index = (x_ppm - this.x_ppm_start) / this.x_ppm_step
         */
        let p1 = x_ppm.map((x) => Math.round((x - this.x_ppm_start) / this.x_ppm_step));
        let p2 = y_ppm.map((y) => Math.round((y - this.y_ppm_start) / this.y_ppm_step));

        /**
         * Get peak widths from the picked peaks object
         */
        let x_width = this.picked_peaks_object.get_column_by_header("XW");
        let y_width = this.picked_peaks_object.get_column_by_header("YW");

        /**
         * Get median of x_width and y_width
         */
        this.median_width_x = mathTool.get_median(x_width);
        this.median_width_y = mathTool.get_median(y_width);
        this.wx = this.median_width_x * 1.6;
        this.wy = this.median_width_y * 1.6;

        // Allocate arrays
        this.peak_map = new Array(this.n_direct * this.n_indirect).fill(-1);
        this.peak_map2 = new Array(this.n_direct * this.n_indirect).fill(0);
        this.peak_map3 = new Array(this.n_direct * this.n_indirect).fill(0);

        // Populate peak_map
        for (let i = 0; i < p1.length; i++) {
            let xx = Math.round(p1[i]);
            let yy = Math.round(p2[i]);
            if (xx >= this.n_direct || xx < 0 || yy < 0 || yy >= this.n_indirect) {
                console.log("Something is wrong with the coordinates in peak_partition.");
            }
            this.peak_map[xx * this.n_indirect + yy] = i;
        }

        // Populate peak_map2 and peak_map3 for peak regions
        for (let i = 0; i < p1.length; i++) {
            let x_from = Math.round(p1[i] - this.wx * 1.5);
            let x_to = Math.round(p1[i] + this.wx * 1.5);
            let y_from = Math.round(p2[i] - this.wy * 1.5) + 1;
            let y_to = Math.round(p2[i] + this.wy * 1.5) + 1;

            if (x_from < 0) x_from = 0;
            if (x_to > this.n_direct) x_to = this.n_direct;
            if (y_from < 0) y_from = 0;
            if (y_to > this.n_indirect) y_to = this.n_indirect;

            for (let m = x_from; m < x_to; m++) {
                for (let n = y_from; n < y_to; n++) {
                    if (m >= this.n_direct || m < 0 || n < 0 || n >= this.n_indirect) {
                        console.log("Something is wrong with the coordinates in peak_partition.");
                    }
                    if (this.p_intensity[i] > 0) {
                        this.peak_map2[m * this.n_indirect + n] = 1;
                    } else {
                        this.peak_map3[m * this.n_indirect + n] = 1;
                    }
                }
            }
        }

        this.cluster_counter = 0;
        this.peak_partition_core(0); // positive peaks
        // this.peak_partition_core(1); // negative peaks
    };

    async peak_partition_core(flag) {
        // === Peak map is used to map the peak position to the data point index ===
        let peak_segment_b = Array(this.n_indirect).fill(0).map(() => []);
        let peak_segment_s = Array(this.n_indirect).fill(0).map(() => []);
        let used = Array(this.n_indirect).fill(0).map(() => []);
        let clusters = [];

        const lowest_level = this.noise_level * this.user_scale2;

        if (flag === 0) {
            // Process positive peaks only
            for (let j = 0; j < this.n_indirect; j++) {
                if (this.raw_data[j * this.n_direct + 0] >= lowest_level && this.peak_map2[j] === 1) {
                    peak_segment_b[j].push(0);
                }
                for (let i = 1; i < this.n_direct; i++) {
                    if ((this.raw_data[j * this.n_direct + i - 1] < lowest_level || this.peak_map2[j + (i - 1) * this.n_indirect] === 0)
                        && (this.raw_data[j * this.n_direct + i] >= lowest_level && this.peak_map2[j + i * this.n_indirect] === 1)) {
                        peak_segment_b[j].push(i);
                    }
                    if ((this.raw_data[j * this.n_direct + i - 1] >= lowest_level && this.peak_map2[j + (i - 1) * this.n_indirect] === 1)
                        && (this.raw_data[j * this.n_direct + i] < lowest_level || this.peak_map2[j + i * this.n_indirect] === 0)) {
                        peak_segment_s[j].push(i);
                    }
                }
                if (peak_segment_s[j].length < peak_segment_b[j].length) {
                    peak_segment_s[j].push(this.n_direct);
                }
                for (let i = 0; i < peak_segment_s[j].length; i++) {
                    used[j].push(0);
                }
            }
        } else {
            // Process negative peaks only
            for (let j = 0; j < this.n_indirect; j++) {
                if (this.raw_data[j * this.n_direct + 0] <= -lowest_level && this.peak_map3[j] === 1) {
                    peak_segment_b[j].push(0);
                }
                for (let i = 1; i < this.n_direct; i++) {
                    if ((this.raw_data[j * this.n_direct + i - 1] > -lowest_level || this.peak_map3[j + (i - 1) * this.n_indirect] === 0)
                        && (this.raw_data[j * this.n_direct + i] <= -lowest_level && this.peak_map3[j + i * this.n_indirect] === 1)) {
                        peak_segment_b[j].push(i);
                    }
                    if ((this.raw_data[j * this.n_direct + i - 1] <= -lowest_level && this.peak_map3[j + (i - 1) * this.n_indirect] === 1)
                        && (this.raw_data[j * this.n_direct + i] > -lowest_level || this.peak_map3[j + i * this.n_indirect] === 0)) {
                        peak_segment_s[j].push(i);
                    }
                }
                if (peak_segment_s[j].length < peak_segment_b[j].length) {
                    peak_segment_s[j].push(this.n_direct);
                }
                for (let i = 0; i < peak_segment_s[j].length; i++) {
                    used[j].push(0);
                }
            }
        }

        // === Breadth-first search to find all clusters ===
        let work = [];

        for (let j = 0; j < this.n_indirect; j++) {
            for (let i = 0; i < used[j].length; i++) {
                if (used[j][i] === 0) {
                    used[j][i] = 1;
                    work.length = 0;
                    work.push([j, i]);

                    let position = 0;
                    while (position < work.length) {
                        const [c_row, c_idx] = work[position];
                        position++;

                        for (let jj = Math.max(0, c_row - 1); jj < Math.min(this.n_indirect, c_row + 2); jj++) {
                            if (jj === c_row) continue;
                            for (let ii = 0; ii < used[jj].length; ii++) {
                                if (used[jj][ii] === 1) continue;
                                if (peak_segment_s[jj][ii] >= peak_segment_b[c_row][c_idx] &&
                                    peak_segment_b[jj][ii] <= peak_segment_s[c_row][c_idx]) {
                                    work.push([jj, ii]);
                                    used[jj][ii] = 1;
                                }
                            }
                        }
                    }

                    clusters.push(work.slice());
                }
            }
        }

        if (flag === 0) {
            console.log(`Total ${clusters.length} positive peak clusters.`);
        } else {
            console.log(`Total ${clusters.length} negative peak clusters.`);
        }

        // === Part II: prepare data for fitting ===
        for (let i0 = 0; i0 < clusters.length; i0++) {
            let min1 = Infinity, min2 = Infinity;
            let max1 = -Infinity, max2 = -Infinity;

            for (const [j, k] of clusters[i0]) {
                const begin = peak_segment_b[j][k];
                const stop = peak_segment_s[j][k];
                if (begin < min1) min1 = begin;
                if (stop > max1) max1 = stop;
                if (j < min2) min2 = j;
                if (j > max2) max2 = j;
            }
            max1++;
            max2++;

            if (max1 - min1 < 3 || max2 - min2 < 3) continue;

            const xydim = (max1 - min1) * (max2 - min2);
            const spect_parts = Array(this.nspect * xydim).fill(0);
            const aas = [];
            const xx = [], yy = [], sx = [], sy = [], gx = [], gy = [], ori_index = [], region_peak_cannot_move_flag = [];

            for (const [j, k] of clusters[i0]) {
                const begin = peak_segment_b[j][k];
                const stop = peak_segment_s[j][k];
                for (let kk = begin; kk < stop; kk++) {
                    for (let kIdx = 0; kIdx < this.nspect; kIdx++) {
                        spect_parts[kIdx * xydim + (kk - min1) * (max2 - min2) + (j - min2)] =
                            this.raw_data[kk + j * this.n_direct];
                    }

                    const peak_ndx = this.peak_map[kk * this.n_indirect + j];
                    if (peak_ndx >= 0) {
                        xx.push(kk - min1);
                        yy.push(j - min2);
                        sx.push(this.sigmax[peak_ndx]);
                        sy.push(this.sigmay[peak_ndx]);
                        gx.push(0.00001);
                        gy.push(0.00001);
                        ori_index.push(peak_ndx);
                        region_peak_cannot_move_flag.push(0);
                        aas.push(this.p_intensity[peak_ndx]);
                    }
                }
            }

            if (xx.length > 0) {
                const workerInput =
                {
                    webassembly_job: "gaussian_fitting",
                    spectrum_index: this.spectrum_index,
                    peak_shape: this.peak_shape,
                    maxround: this.maxround,
                    cluster_counter: this.cluster_counter,
                    peak_assignments: this.peak_assignments,
                    peak_sign: flag === 0 ? 1 : -1,
                    min1, min2,
                    size1: max1 - min1,
                    size2: max2 - min2,
                    nspect: this.nspect,
                    spect_parts,
                    xx, yy, aas, sx, sy, gx, gy, ori_index, region_peak_cannot_move_flag,
                    median_width_x: this.median_width_x,
                    median_width_y: this.median_width_y,
                    wx: this.wx * 1.5,
                    wy: this.wy * 1.5,
                    noise_level: this.noise_level,
                    user_scale2: this.user_scale2,
                    too_near_cutoff: this.too_near_cutoff,
                    step1: this.x_ppm_step,
                    step2: this.y_ppm_step,
                    removal_cutoff: this.removal_cutoff
                };
                this.cluster_counter++;

                // Call the worker, wait for result:
                webassembly_1d_worker_2.postMessage(workerInput);
            }
        }

        return true;
    };

    /**
     * result is from web worker, which use web assembly to do gaussian fitting
     * @param {*} result 
     */
    process_gaussian_fitting_result(result) {
        console.log("Processing gaussian fitting result...");
        /**
         * result is an object with the following properties:
         *  webassembly_job: event.data.webassembly_job,
            spectrum_index: event.data.spectrum_index,
            peak_assignments: event.data.peak_assignments,
            p1: p1,
            p2: p2,
            group: group,
            nround: nround,
            p_intensity: p_intensity,
            sigmax: sigmax,
            sigmay: sigmay,
            peak_index: peak_index,
            err: err,
            num_sums: num_sums,
            gammax: gammax,
            gammay: gammay,
            p_intensity_all_spectra: p_intensity_all_spectra,
         */

        if (this.fitted_peaks_object === null) {
            this.fitted_peaks_object = new cpeaks();
            /**
             * Add column_headers and column_formats to fitted_peaks_object 
             */
            this.fitted_peaks_object.column_headers = [
                "INDEX", 'X_AXIS', "Y_AXIS",
                "X_PPM", "Y_PPM", "XW", "YW", "HEIGHT", "ASS", "SIGMAX", "SIGMAY", "GAMMAX", "GAMMAY",
                "GROUP", "NROUND"
            ];
            this.fitted_peaks_object.column_formats = [
                "%5d", "%9.4f", "9.4f",
                "%8.4f", "8.4f", "%7.3f", "%7.3f", "%e", "%s", "%f", "%f", "%f", "%f",
                "%5d", "%4d"
            ];
            /**
             * DEfine this.fitted_peaks_object.columns as array of arrays, and out layer size is the same as column_headers
             */
            this.fitted_peaks_object.columns = [];
            for (let i = 0; i < this.fitted_peaks_object.column_headers.length; i++) {
                this.fitted_peaks_object.columns.push([]);
            }
        }
        /**
         * Add current result to fitted_peaks_object, one row per peak
         */
        for (let i = 0; i < result.p1.length; i++) {
            let current_index = this.fitted_peaks_object.columns[0].length;
            let row = [];
            row.push(current_index + i);
            row.push(result.p1[i]);
            row.push(result.p2[i]);
            row.push(this.x_ppm_start + result.p1[i] * this.x_ppm_step);
            row.push(this.y_ppm_start + result.p2[i] * this.y_ppm_step);
            let fwhhx = 1.0692 * result.gammax[i] + Math.sqrt(0.8664 * result.gammax[i] ** 2 + 5.5452 * result.sigmax[i] ** 2);
            let fwhhy = 1.0692 * result.gammay[i] + Math.sqrt(0.8664 * result.gammay[i] ** 2 + 5.5452 * result.sigmay[i] ** 2);
            row.push(fwhhx);
            row.push(fwhhy);
            row.push(result.p_intensity[i]);
            let original_peak_index = result.peak_index[i];
            row.push(this.peak_assignments[original_peak_index] ? this.peak_assignments[original_peak_index] : "peak");
            row.push(result.sigmax[i]);
            row.push(result.sigmay[i]);
            row.push(result.gammax[i]);
            row.push(result.gammay[i]);
            row.push(result.group[i]);
            row.push(result.nround[i]);

            this.fitted_peaks_object.add_row(row);
        }
    };
};
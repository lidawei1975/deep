
/**
 * Linear regression function to calculate the slope of a line, without intercept
 * @param {*} x 
 * @param {*} y 
 * @param {*} w - weights for each point
 * @returns an object with slope and a predict function
 */

function linearRegressionNoIntercept(x,w,y) {
  
    const n = x.length;
  
    // Calculate the sum of x*y and the sum of x*x.
    let sumXY = 0;
    let sumXSquared = 0;
  
    for (let i = 0; i < n; i++) {
      sumXY += x[i] * y[i] * w[i];
      sumXSquared += x[i] * x[i] * w[i];
    }
  
    // Calculate the slope (beta).
    const slope = sumXY / sumXSquared;
  
    // predicted y values based on the slope
    let y_pre = x.map(xi => slope * xi);
  
    // Return the slope and the predict function.
    return {
      slope: slope,
      y_pre: y_pre,
    };
  }


/**
 * Define the peaks object, to store peak parameters of a spectrum
 * and peak processing methods.
 */
class cpeaks {

    constructor() {
        this.comments = []; // comments, string array
        this.column_headers = []; // column headers, string array
        this.column_formats = []; // column formats, string array
        this.columns = []; // columns, array of arrays (arrays have same length, but different types)
    };

    /**
     * Clear all data in the peaks object
     */
    clear_all_data() {
        this.comments = [];
        this.column_headers = [];
        this.column_formats = [];
        this.columns = [];
        this.gradients = null;
        this.scale_constant = 1.0;
    }

    /**
     * Apply X_PPM reference shift to all X_PPM values
     */
    update_x_ppm_ref(x_ppm_ref) {
        let x_ppm_index = this.column_headers.indexOf('X_PPM');
        if (x_ppm_index === -1) {
            return false;
        }
        this.columns[x_ppm_index] = this.columns[x_ppm_index].map(x => x + x_ppm_ref);
        return true;
    }

    /**
     * Apply Y_PPM reference shift to all Y_PPM values
     */
    update_y_ppm_ref(y_ppm_ref) {
        let y_ppm_index = this.column_headers.indexOf('Y_PPM');
        if (y_ppm_index === -1) {
            return false;
        }
        this.columns[y_ppm_index] = this.columns[y_ppm_index].map(y => y + y_ppm_ref);
        return true;
    }

    /**
     * A class method to process a peaks.tab file (nmrPipe format)
     * @param {string} peaks_tab - the peaks.tab file content as one big string, separated by newlines
     */
    process_peaks_tab(peaks_tab) {

        /**
         * Clear all data first
         */
        this.clear_all_data();


        let lines = peaks_tab.split('\n');

        /**
         * Remove empty lines or lines with only white spaces, tabs, etc.
         */
        lines = lines.filter(line => line.trim() !== '');

        /**
         * Put lines starts with DATA into this.comments
         */
        this.comments = lines.filter(line => line.startsWith('DATA'));

        // Extract headers from the VARS line
        const varsLine = lines.find(line => line.startsWith('VARS'));
        this.column_headers = varsLine.split(/\s+/).slice(1);

        // Extract formats from the FORMAT line
        const formatLine = lines.find(line => line.startsWith('FORMAT'));
        this.column_formats = formatLine.split(/\s+/).slice(1);

        /**
         * Get all lines that are after the VARS and FORMAT lines
         */
        lines = lines.slice(lines.indexOf(varsLine) + 1);

        // Extract data rows (skipping lines starting with VARS and FORMAT and DATA or is empty)
        let dataRows = lines.filter(line => !line.startsWith('VARS') && !line.startsWith('FORMAT') && !line.startsWith('DATA') && line.trim() !== '');


        /**
         * Remove leading and trailing white spaces in each line of dataRows then
         * remove dataRows that don't have same number of columns as the number of column headers
         */
        dataRows = dataRows.filter(row => row.trim().split(/\s+/).length === this.column_headers.length);

        /**
         * Split data rows into columns and save each column as an array in this.columns
         * IF column format contains " %3s" or " %4s, etc., treat it as a string
         * ELSE if column format contains " %4d", "%1d", %d, etc., treat it as an integer
         * ELSE treat it as a float
         */
        this.columns = this.column_headers.map((header, index) => {
            return dataRows.map(row => {
                /**
                 * Clear leading and trailing white spaces
                 */
                row = row.trim();
                const value = row.split(/\s+/)[index];
                if (this.column_formats[index].includes('s')) {
                    return value;
                } else if (this.column_formats[index].includes('d')) {
                    return parseInt(value);
                } else {
                    return parseFloat(value);
                }
            });
        });

        /**
         * Replace values corresponding to column_header with row index + 1
         */
        let index_column = this.column_headers.indexOf('INDEX');
        if (index_column !== -1) {
            this.columns[index_column] = this.columns[index_column].map((value, index) => index + 1);
        }
        else {
             /**
             * Add a column_headers at the beginning, "INDEX", column_formats is "%5d"
             * and column values is 1,2,3,4
             */
            this.column_headers.unshift("INDEX");
            this.column_formats.unshift("%5d");
            let index_array=[];
            for(let i=0;i<this.columns[0].length;i++)
            {
                index_array.push(i);
            }
            this.columns.unshift(index_array);
        }
    };

    /**
     * Class method to process a peaks.list file from Sparky
     */
    process_peaks_list(peaks_list) {
        /**
         * Clear all data first
         */
        this.clear_all_data();

        const lines = peaks_list.split('\n');

        let b_header = false;

        /**
         * Check all lines for header line and data lines
         */
        lines.forEach((line) => {
            /**
             * Trim and split the line by white spaces
             */
            let parts = line.trim().split(/\s+/);
            /**
             * Check if the line is a header line (contain w2 and w1)
             */
            if (b_header == false && parts.includes('w2') && parts.includes('w1')) {
                this.column_headers = parts;
                /**
                 * Assign column formats based on the header,
                 * w2 and w1 are %10.4f, Assignment is %s and height is %e
                 * other columns are %s
                 */
                this.column_formats = parts.map((header) => {
                    if (header === 'w2' || header === 'w1') {
                        return '%10.4f';
                    } else if (header === 'Assignment') {
                        return '%s';
                    } else if (header === 'height') {
                        return '%e';
                    } else {
                        return '%s';
                    }
                });
                b_header = true;
                this.columns =[];
                for(let i=0;i<this.column_headers.length;i++)
                {
                    this.columns.push([]);
                }
            }
            else if (b_header === true) {
                /**
                 * only when the header line is found, process the data lines
                 * and only process as data line when number of parts is the same as number of headers
                 */
                if (parts.length === this.column_headers.length) {
                    for (let i = 0; i < this.column_headers.length; i++) {
                        if (this.column_formats[i].includes('s')) {
                            this.columns[i].push(parts[i]);
                        }
                        else if (this.column_formats[i].includes('f')) {
                            this.columns[i].push(parseFloat(parts[i]));
                        }
                        else if (this.column_formats[i].includes('e')) {
                            this.columns[i].push(parseFloat(parts[i]));
                        }
                    }
                }
            }
        });

        /**
         * Sort this.column_headers according to the order of w2 < w1 < height < Assignment
         * Keep track of the original index of each column header
         * Note not necessary all column headers are present
         */
        let original_indexes = this.column_headers.map((header, index) =>{ return { header, index };});
        original_indexes.sort((a, b) => {
            let order = ['w2', 'w1', 'height', 'Assignment'];
            return order.indexOf(a.header) - order.indexOf(b.header);
        });

        /**
         * Apply the sorting to this.column_headers, this.column_formats, and this.columns
         */
        this.column_headers = original_indexes.map((item) => this.column_headers[item.index]);
        this.column_formats = original_indexes.map((item) => this.column_formats[item.index]);
        this.columns = original_indexes.map((item) => this.columns[item.index]);

        /**
         * Change header "Assignment" to "ASS", "w1" to "Y_PPM", "w2" to "X_PPM", and Height to HEIGHT
         */
        
        this.column_headers = this.column_headers.map(item => {
            if (item == "Assignment") {
                return "ASS";
            }
            else if (item == "w1") {
                return "Y_PPM";
            }
            else if (item == "w2") {
                return "X_PPM";
            }
            else if (item == "Height") {
                return "HEIGHT";
            }
            else {
                return item;
            }
        });

        /**
         * Add a column_headers at the beginning, "INDEX", column_formats is "%5d"
         * and column values is 1,2,3,4
         */
        this.column_headers.unshift("INDEX");
        this.column_formats.unshift("%5d");
        let index_array=[];
        for(let i=0;i<this.columns[0].length;i++)
        {
            index_array.push(i);
        }
        this.columns.unshift(index_array);
        return;
    }

    /**
     * A class method to change all values in a column by add,sub,mul,div a value.
     * @param {string} column_header_name - the column header name
     * @param {string} operation - the operation to be performed: add, sub, mul, div
     * @param {number} value - the value to be added, subtracted, multiplied, or divided
     * @return {boolean} - true if the column is successfully changed, false if the column is not found, not a number, or the operation is invalid
     */
    change_column(column_header_name, operation, value) {
        let index = this.column_headers.indexOf(column_header_name);
        if (index === -1) {
            return false;
        }
        if (isNaN(value)) {
            return false;
        }

        /**
         * Check column_formats[index] to determine the type of the column, must be float or integer
         * unless operation is set (which is allowed for any type of column)
         */
        if( operation !== "set" && this.column_formats[index].includes('s')) {
            return false;
        }

        if (operation === 'add') {
            this.columns[index] = this.columns[index].map(x => x + value);
        } else if (operation === 'sub') {
            this.columns[index] = this.columns[index].map(x => x - value);
        } else if (operation === 'mul') {
            this.columns[index] = this.columns[index].map(x => x * value);
        } else if (operation === 'div') {
            this.columns[index] = this.columns[index].map(x => x / value);
        } 
        else if (operation === 'set') {
            this.columns[index] = this.columns[index].map(x => value);
        }
        else {
            return false;
        }
        return true;
    }

    /**
     * Set a value in a column by column header name and at a specific index (row)
     * This work for both numbers and strings
     */
    set_column_row_value(column_header_name, index, value) {
        let column_index = this.column_headers.indexOf(column_header_name);
        if (column_index === -1) {
            return false;
        }
        this.columns[column_index][index] = value;
        return true;
    }

    /**
     * Get a column as a array, using colomn header name
     */
    get_column_by_header(column_header_name) {
        let index = this.column_headers.indexOf(column_header_name);
        if (index === -1) {
            return [];
        }
        return this.columns[index];
    }

    /**
     * Copy all data from another peaks object
     */
    copy_data(peaks) {
        this.comments = peaks.comments;
        this.column_headers = peaks.column_headers;
        this.column_formats = peaks.column_formats;
        this.columns = peaks.columns;
    }

    /**
     * Get a array of json objects, each object is a row of the peaks object with only selected columns
     * @param {string[]} column_header_names - the column header names to be selected
     */
    get_selected_columns(column_header_names) {
        let indexes = column_header_names.map(header => this.column_headers.indexOf(header));
        let result = [];
        for (let i = 0; i < this.columns[0].length; i++) {
            let row = {};
            for (let j = 0; j < indexes.length; j++) {
                if (indexes[j] === -1) {
                    continue;
                }
                row[column_header_names[j]] = this.columns[indexes[j]][i];
            }
            result.push(row);
        }
        return result;
    }

    
    /**
     * Get a array of array, each array is a row of the peaks object with only selected columns
     * @param {string[]} column_header_names - the column header names to be selected
     */
    get_selected_columns_as_array(column_header_names) {
        let indexes = column_header_names.map(header => this.column_headers.indexOf(header));
        let result = [];
        for (let i = 0; i < this.columns[0].length; i++) {
            let row = [];
            let counter = 0;
            for (let j = 0; j < indexes.length; j++) {
                if (indexes[j] === -1) {
                    continue;
                }
                row[counter] = this.columns[indexes[j]][i];
                counter++;
            }
            result.push(row);
        }
        return result;
    }

    /**
     * Get an array of arrays of selected columns
     * result[column_index][row_index]
     * @param {string[]} column_header_names - the column header names to be selected
     */
    get_selected_column_values(column_header_names) {
        let indexes = column_header_names.map(header => this.column_headers.indexOf(header));
        let result = [];
        for (let i = 0; i < indexes.length; i++) {
            result.push(this.columns[indexes[i]]);
        }
        return result;
    }



    /**
     * Filter a column by a range of values
     * then apply the filter to all columns
     */
    filter_by_column_range(column_header_name, min_value, max_value) {
        let index = this.column_headers.indexOf(column_header_name);
        if (index === -1) {
            return false;
        }
        if (isNaN(min_value) || isNaN(max_value)) {
            return false;
        }
        const indexes = this.columns[index].map((value, index) => (value >= min_value && value <= max_value) ? index : -1)
            .filter((index) => index !== -1);

        /**
         * Apply the filter to all columns
         */
        this.columns = this.columns.map((column) => indexes.map((index) => column[index]));
    }

    /**
     * Filter by several columns by a range of values (must fulfill all conditions)
     * @param {*} column_header_names 
     * @param {*} min_values 
     * @param {*} max_values 
     * @param {bool} b_keep: true to keep the rows that fulfill the conditions, false to remove them
     * @return {bool} - true if the filter is successfully applied, false if the columns are not found, not a number, or the operation is invalid
     */
    filter_by_columns_range(column_header_names, min_values, max_values, b_keep = true) {
        let indexes = column_header_names.map(header => this.column_headers.indexOf(header));
        if (indexes.includes(-1)) {
            return false;
        }
        if (min_values.length !== indexes.length || max_values.length !== indexes.length) {
            return false;
        }
        if (min_values.some(isNaN) || max_values.some(isNaN)) {
            return false;
        }

        for (let i = this.columns[0].length - 1; i >= 0; i--) {
            let b_fulfill = true;
            for (let j = 0; j < indexes.length; j++) {
                if (this.columns[indexes[j]][i] < min_values[j] || this.columns[indexes[j]][i] > max_values[j]) {
                    b_fulfill = false;
                    break;
                }
            }

            if (b_fulfill === false && b_keep === true) {
                for (let j = 0; j < this.columns.length; j++) {
                    this.columns[j].splice(i, 1);
                }
            }
            else if (b_fulfill === true && b_keep === false) {
                for (let j = 0; j < this.columns.length; j++) {
                    this.columns[j].splice(i, 1);
                }
            }
        }

        /**
         * Need to update the INDEX column
         */
        let index_column = this.column_headers.indexOf('INDEX');
        if (index_column !== -1) {
            for (let i = 0; i < this.columns[index_column].length; i++) {
                this.columns[index_column][i] = i + 1;
            }
        }

        return true;
    }

    /**
     * Remove a row by column with header named "INDEX"
     * @param {number} index - the index of the row to be removed
     */
    remove_row(index) {
        if(index < 0 || index >= this.columns[0].length) {
            return false;
        }
        for(let i = 0; i < this.columns.length; i++) {
            this.columns[i].splice(index, 1);
        }

        /**
         * Also need to update the INDEX column
         */
        let index_column = this.column_headers.indexOf('INDEX');
        if (index_column !== -1) {
            for(let i = index; i < this.columns[index_column].length; i++) {
                this.columns[index_column][i] = i + 1;
            }
        }
        
    }

    /**
     * Update X_PPM and Y_PPM of a row, from index (value of the column with header "INDEX")
     */
    update_row(index, x_ppm, y_ppm) {
        let index_index = this.column_headers.indexOf('INDEX');
        if (index_index === -1) {
            return false;
        }
        let row_index = this.columns[index_index].indexOf(index);
        if (row_index === -1) {
            return false;
        }
        let x_ppm_index = this.column_headers.indexOf('X_PPM');
        let y_ppm_index = this.column_headers.indexOf('Y_PPM');
        if (x_ppm_index === -1 || y_ppm_index === -1) {
            return false;
        }
        this.columns[x_ppm_index][row_index] = x_ppm;
        this.columns[y_ppm_index][row_index] = y_ppm;
        return true;
    }

        /**
         * Update X_PPM and Y_PPM of a row, from index (value of the column with header "INDEX")
         */
        update_row_1d(index, x_ppm, height) {
            let index_index = this.column_headers.indexOf('INDEX');
            if (index_index === -1) {
                return false;
            }
            let row_index = this.columns[index_index].indexOf(index);
            if (row_index === -1) {
                return false;
            }
            let x_ppm_index = this.column_headers.indexOf('X_PPM');
            let height_index = this.column_headers.indexOf('HEIGHT');
            if (x_ppm_index === -1 || height_index === -1) {
                return false;
            }
            this.columns[x_ppm_index][row_index] = x_ppm;
            this.columns[height_index][row_index] = height;
            return true;
        }

    /**
     * Add a row to the peaks object from a json object
     * {X_PPM: x_ppm,Y_PPM: y_ppm, HEIGHT: data_height};
     * INDEX will be 10000, 10001, 10002, etc.
     * Set X_PPM and Y_PPM and HEIGHT columns to the values in the json object
     * For others:
     * For number columns, set to median value of the column
     * For string columns, set to the first value of the column
     */
    add_row(new_row) {
        
        for (let i = 0; i < this.column_headers.length; i++) {
            if (this.column_headers[i] === 'INDEX') {
                this.columns[i].push(this.columns[i].length + 1);
            }
            else if (this.column_headers[i] === 'X_PPM') {
                this.columns[i].push(new_row.X_PPM);
            }
            else if (this.column_headers[i] === 'Y_PPM') {
                this.columns[i].push(new_row.Y_PPM);
            }
            else if (this.column_headers[i] === 'HEIGHT') {
                this.columns[i].push(new_row.HEIGHT);
            }
            else if (this.column_formats[i].includes('s')) {
                this.columns[i].push(this.columns[i][0]);
            }
            else {
                /**
                 * Keep original values in the column, then calculate median using a sorted copy of the column
                 */
                let temp=this.columns[i].slice();
                temp.sort((a, b) => a - b);
                let median_temp = temp[temp.length >> 1];
                this.columns[i].push(median_temp);
            }
        }
        return true;
    }


    /**
     * An helper function to generate string from value, according to saved format string
     */
    format_value(value, format) {

        if (value === null || value === undefined) {
            if(format.includes('s')){
                value = 'n.a.';
            }
            else {
                value = 0;
            }
        }

        if (format.includes('s')) {
            return value.toString();
        }
        else if (format.includes('d')) {
            let num = parseInt(format.substring(1,format.length - 1));
            return value.toFixed(0).padStart(num, ' ');
        }
        else if (format.includes('f')) {
            /**
             * Get number of decimal places from column_formats[j].
             * First get %5.3f to 5.3
             * Use toFixed to format the number to that many decimal places
            */
            let num = format.substring(1, format.length - 1);
            let decimal_places = 3;
            let width = 6;
            /**
             * num could be 7.3, or 7, or empty
             */
            if (num.includes('.')) {
                decimal_places = parseInt(num.split('.')[1]);
                width = parseInt(num.split('.')[0]);
            }
            else if (num.length === 0) {
                /**
                 * If format is %f, set default to 6.3f decimal places
                 */
                decimal_places = 3;
                width = 6;
            }
            else {
                /**
                 * If format is %7f, set decimal to 7-3, but must >=1
                 */
                width = parseInt(num);
                decimal_places = width - 3;
                if (decimal_places < 1) {
                    decimal_places = 1;
                }
            }
            return value.toFixed(decimal_places).padStart(width, ' ');
        }
        else if (format.includes('e')) {
            /**
             * Get number of decimal places from column_formats[j]
             * Use toExponential to format the number to that many decimal places
             */
            let decimal_places = 3;  //default if not specified
            let number = format.substring(1, format.length - 1);
            if (number.includes('.')) {
                decimal_places = number.split('.')[1];
            }
            let str = value.toExponential(decimal_places);
            /**
             * Per C++ sprintf, if there is only single digit after the e+ or e-, add a zero
             */
            if (str.includes('e+')) {
                let num = str.split('e+')[1];
                if (num.length === 1) {
                    str = str.replace('e+', 'e+0');
                }
            }
            if (str.includes('e-')) {
                let num = str.split('e-')[1];
                if (num.length === 1) {
                    str = str.replace('e-', 'e-0');
                }
            }
            return str;
        }
        else {
            return value.toString();
        }
    };

    /**
     * Add a row to the peaks object. Length of row must be the same as the number of columns
     * @param {array} row 
     */
    add_row(row)
    {
        if (row.length !== this.column_headers.length) {
            return false;
        }
        /**
         * Check if row has all values, if not, set to 0 or 'n.a.'
         */
        for (let i = 0; i < row.length; i++) {
            if (row[i] === null || row[i] === undefined) {
                if (this.column_formats[i].includes('s')) {
                    row[i] = 'n.a.';
                }
                else {
                    row[i] = 0;
                }
            }
        }
        this.columns.forEach((column, index) => {
            column.push(row[index]);
        });
        return true;
    }

    /**
     * Class method to save the peaks object as a peaks.tab file (nmrPipe format)
     */
    save_peaks_tab() {
        let peaks_tab = '';
        peaks_tab += this.comments.join('\n') + '\n';
        peaks_tab += 'VARS ' + this.column_headers.join(' ') + '\n';
        peaks_tab += 'FORMAT ' + this.column_formats.join(' ') + '\n';
        for (let i = 0; i < this.columns[0].length; i++) {
            let row = '';
            for (let j = 0; j < this.columns.length; j++) {

                /**
                 * If this.columns[j][i] is null
                 * or is a number and is NaN, set it to 0
                 */
                if (this.columns[j][i] === null || (typeof this.columns[j][i] === 'number' && isNaN(this.columns[j][i])))
                {
                    this.columns[j][i] = 0;
                }

                /**
                 * Simulate a c++ sprintf function to format the value according to column_formats[j]
                 */
                if (this.column_formats[j].includes('s')) {
                    row += this.columns[j][i].toString().padEnd(parseInt(this.column_formats[j]), ' ');
                }
                else if (this.column_formats[j].includes('d')) {
                    /**
                     * Get the number from a string like %4d
                     */
                    let num = parseInt(this.column_formats[j].substring(1, this.column_formats[j].length - 1));
                    row += this.columns[j][i].toFixed(0).padStart(num, ' ');
                }
                else if (this.column_formats[j].includes('f')) {
                    /**
                     * Get number of decimal places from column_formats[j].
                     * First get %5.3f to 5.3
                     * Use toFixed to format the number to that many decimal places
                     */
                    let num = this.column_formats[j].substring(1, this.column_formats[j].length - 1);
                    let decimal_places = 3;
                    let width = 6;
                    /**
                     * num could be 7.3, or 7, or empty
                     */
                    if (num.includes('.')) {
                        decimal_places = parseInt(num.split('.')[1]);
                        width = parseInt(num.split('.')[0]);
                    }
                    else if(num.length === 0) {
                        /**
                         * If format is %f, set default to 6.3f decimal places
                         */
                        decimal_places = 3;
                        width = 6;
                    }
                    else{
                        /**
                         * If format is %7f, set decimal to 7-3, but must >=1
                         */
                        width = parseInt(num);
                        decimal_places = width - 3;
                        if (decimal_places < 1) {
                            decimal_places = 1;
                        }
                    }
                    row += this.columns[j][i].toFixed(decimal_places).padStart(width, ' ');
                }
                else if (this.column_formats[j].includes('e')) {
                    /**
                     * Get number of decimal places from column_formats[j]
                     * Use toExponential to format the number to that many decimal places
                     */
                    let decimal_places = 6;  //default if not specified
                    let number = this.column_formats[j].substring(1, this.column_formats[j].length - 1);
                    if (number.includes('.')) {
                        decimal_places = number.split('.')[1];
                    }
                    let str = this.columns[j][i].toExponential(decimal_places);
                    /**
                     * Per C++ sprintf, if there is only single digit after the e+ or e-, add a zero
                     */
                    if (str.includes('e+')) {
                        let num = str.split('e+')[1];
                        if (num.length === 1) {
                            str = str.replace('e+', 'e+0');
                        }
                    }
                    if (str.includes('e-')) {
                        let num = str.split('e-')[1];
                        if (num.length === 1) {
                            str = str.replace('e-', 'e-0');
                        }
                    }

                    row += str;
                }
                else {
                    /**
                     * If column format is not recognized, just pad the value with spaces
                     */
                    row += this.columns[j][i].toString() + ' ';
                }
                /**
                 * Add a space between columns, except for the last column
                 */
                if (j < this.columns.length - 1) {
                    row += ' ';
                }
            }
            peaks_tab += row + '\n';
        }
        return peaks_tab;
    }

    /**
     * Function to run DOSY fitting on all peaks (pseudo 3D peak list only, with Z_A1, Z_A2, Z_A3 ... columns)
     * This function will add a column "DOSY" to the peaks object, with the fitted DOSY value for each peak
     */
    run_dosy_fitting(gradients,weights,scale_constant = 1.0) {

        /**
         * Save gradients and scale constant as properties of the peaks object
         * for future reference
         */
        this.gradients = gradients;
        this.scale_constant = scale_constant;

        /**
         * Get total # of Z_A1, Z_A2, Z_A3 ... columns, which must == gradients.length
         */
        let z_columns = this.column_headers.filter(header => header.startsWith('Z_A') && !header.endsWith('_STD'));
        if (z_columns.length !== gradients.length) {
            return {
                message: 'Number of Z_A columns does not match the number of gradients',
                result: false
            };
        }
        /**
         * If previous DOSY column exists, remove it
         */
        let dosy_index = this.column_headers.indexOf('DOSY');
        if (dosy_index !== -1) {
            this.column_headers.splice(dosy_index, 1);
            this.column_formats.splice(dosy_index, 1);
            this.columns.splice(dosy_index, 1);
        }

        /**
         * Get a 2D array of Z_A values, each row is a peak, each column is a Z_A value. 
         * The array should be row-major, for optimal performance in the fitting function
         * Step 1, get column indexes of Z_A columns
         */
        let z_column_indexes = z_columns.map(header => this.column_headers.indexOf(header));
        /**
         * Step 2, get the Z_A values for each peak
         */
        for(var i = 0; i < this.columns[0].length; i++) {
            let z_values = z_column_indexes.map(index => this.columns[index][i]);
            /**
             * Step 3, run the fitting function to get the DOSY value
             */
            let dosy_value = this.dosy_fitting_core(z_values, weights,gradients);
            /**
             * Step 4, add the DOSY value to the peaks object
             */
            if (i === 0) {
                this.column_headers.push('DOSY');
                this.column_formats.push('%6.4e');
                this.columns.push([]);
            }
            this.columns[this.column_headers.indexOf('DOSY')].push(-dosy_value.slope*scale_constant);
        }
        return {
            message: 'DOSY fitting completed successfully',
            result: true,
        };
    }

    /**
     * Dosy fitting function to calculate the DOSY value from the Z_A values and gradients
     * I=I0*exp(-D*G^2), so D = -ln(I/I0)/G^2, so we can get D from a linear fit of -ln(I) vs G^2
     */
    dosy_fitting_core(z_values,weights, gradients) {
        /**
         * Step 1, get the natural log of the Z_A values.
         * z[0] is always 1.0, and z becomes smaller as the gradient increases
         */
        let ln_z = z_values.map(z => Math.log(z));
        /**
         * Step 2, get the square of the gradients
         */
        let g_squared = gradients.map(g => g * g);
        /**
         * Step 3, perform a linear fit of ln_z vs g_squared to get the slope while forcing the intercept to be 0
         */
        let result = linearRegressionNoIntercept(g_squared, weights,ln_z);
        /**
         * Step 4, calculate the DOSY value from the slope
         */
        return result; //object with two properties: slope and y_pre
    };


    /**
     * Run error estimate from an array of cpeaks objects (all have same column headers and same number of rows)
     * on selected columns only
     * @param {object[]} peaks_array - array of cpeaks objects
     * @param {string[]} column_header_names - array of column header names to be selected
     */
    error_estimate(peaks_array, column_header_names) {
        /**
         * Check if all peaks_array have the same column headers and same number of rows
         */
        if (peaks_array.length === 0) {
            return {
                message: 'No peaks array provided',
                result: false
            };
        }
        let num_rows = peaks_array[0].columns[0].length;
        for (let i = 1; i < peaks_array.length; i++) {
            if (peaks_array[i].columns[0].length !== num_rows) {
                return {
                    message: 'Number of rows in peaks array do not match',
                    result: false
                };
            }
        }
        /**
         * Get the selected columns from each peaks object
         */
        let selected_columns = [];
        for (let i = 0; i < peaks_array.length; i++) {
            selected_columns.push(peaks_array[i].get_selected_column_values(column_header_names));
        }

        /**
         * selected_columns is an array of array of array 
         * selected_columns[error_run_index][column_index][peak_index]
         * We need to get the std of each column for each peak
         * That is, std_result[column_index][peak_index]
         */
        let std_result = [];
        for (let i = 0; i < selected_columns[0].length; i++)
        {
            let std_result_column = [];
            for (let j = 0; j < selected_columns[0][i].length; j++)
            {
                let sum = 0;
                for (let k = 0; k < selected_columns.length; k++)
                {
                    sum += selected_columns[k][i][j];
                }
                let mean = sum / selected_columns.length;
                let sum_sq = 0;
                for (let k = 0; k < selected_columns.length; k++)
                {
                    sum_sq += Math.pow(selected_columns[k][i][j] - mean, 2);
                }
                let std = Math.sqrt(sum_sq / (selected_columns.length - 1));
                std_result_column.push(std);
            }
            std_result.push(std_result_column);
        }
     

        /**
         * Add a new column to this object with the std values for column_header_names 
         */
        for(let i=0;i<column_header_names.length;i++)
        {
            let new_header_name = column_header_names[i] + '_STD';
            let new_format = peaks_array[0].column_formats[peaks_array[0].column_headers.indexOf(column_header_names[i])]; //same format as the original column
            this.column_headers.push(new_header_name);
            this.column_formats.push(new_format);
            this.columns.push(std_result[i]);
        }

        return {
            message: 'Error estimate completed successfully',
            result: true,
        };
    };

    /**
     * append_columns to this peaks object
     * @param {object} new_peak - the peaks object to be appended
     */
    append_columns(new_peak) {
        this.column_formats = this.column_formats.concat(new_peak.column_formats);
        this.column_headers = this.column_headers.concat(new_peak.column_headers);
        this.columns = this.columns.concat(new_peak.columns);
        return {
            message: 'Columns appended successfully',
            result: true,
        };
    };

    /**
     * Remove all columns that are error (end with _STD)
     */
    remove_error_columns(){
        let error_columns = this.column_headers.filter(header => header.endsWith('_STD'));
        for (let i = 0; i < error_columns.length; i++) {
            let index = this.column_headers.indexOf(error_columns[i]);
            this.column_headers.splice(index, 1);
            this.column_formats.splice(index, 1);
            this.columns.splice(index, 1);
        }
        return {
            message: 'Error columns removed successfully',
            result: true,
        };
    }
};

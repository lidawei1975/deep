/**
 * JS workflow wrapper for 2D peak fitting with WebAssembly gaussian_fit.
 * This follows the same high-level flow as spectrum_fit.cpp:
 * load spectrum + peaks -> partition into regions -> fit each region -> gather results.
 */

function vectorToArray(v) {
    const out = [];
    const n = v.size();
    for (let i = 0; i < n; i++) {
        out.push(v.get(i));
    }
    return out;
}

function toVectorDouble(Module, arr) {
    const v = new Module.VectorDouble();
    for (const x of arr) {
        v.push_back(Number(x));
    }
    return v;
}

function toVectorInt(Module, arr) {
    const v = new Module.VectorInt();
    for (const x of arr) {
        v.push_back(x | 0);
    }
    return v;
}

function median(values) {
    if (!values || values.length === 0) {
        return 0.0;
    }
    const t = values.slice().sort((a, b) => a - b);
    const m = Math.floor(t.length / 2);
    if (t.length % 2 === 0) {
        return 0.5 * (t[m - 1] + t[m]);
    }
    return t[m];
}

function argsort(values) {
    return values
        .map((value, index) => ({ value, index }))
        .sort((a, b) => a.value - b.value)
        .map((x) => x.index);
}

class GaussianFitWasm {
    constructor(wasmModule) {
        this.module = wasmModule;
        this.instance = new wasmModule.gaussian_fit();
        this._nspectra = 0;
    }

    delete() {
        if (this.instance) {
            this.instance.delete();
            this.instance = null;
        }
    }

    init(config) {
        const {
            xstart,
            ystart,
            xdim,
            ydim,
            nspectra,
            surface,
            x,
            y,
            amp,
            sigmax,
            sigmay,
            gammax,
            gammay,
            originalNdx,
            cannotMove,
            medianWidthX,
            medianWidthY
        } = config;

        const npeak = x.length;
        if (npeak === 0) {
            return false;
        }
        if (y.length !== npeak || sigmax.length !== npeak || sigmay.length !== npeak || gammax.length !== npeak || gammay.length !== npeak) {
            throw new Error("Peak vector sizes are inconsistent");
        }
        if (originalNdx.length !== npeak || cannotMove.length !== npeak) {
            throw new Error("Index vector sizes are inconsistent");
        }
        if (amp.length !== npeak * nspectra) {
            throw new Error("amp length must be npeak * nspectra");
        }
        if (surface.length !== xdim * ydim * nspectra) {
            throw new Error("surface length must be xdim * ydim * nspectra");
        }

        const vsurface = toVectorDouble(this.module, surface);
        const vx = toVectorDouble(this.module, x);
        const vy = toVectorDouble(this.module, y);
        const va = toVectorDouble(this.module, amp);
        const vsx = toVectorDouble(this.module, sigmax);
        const vsy = toVectorDouble(this.module, sigmay);
        const vgx = toVectorDouble(this.module, gammax);
        const vgy = toVectorDouble(this.module, gammay);
        const vorig = toVectorInt(this.module, originalNdx);
        const vmove = toVectorInt(this.module, cannotMove);

        let ok = false;
        try {
            ok = this.instance.init(
                xstart,
                ystart,
                xdim,
                ydim,
                nspectra,
                vsurface,
                vx,
                vy,
                va,
                vsx,
                vsy,
                vgx,
                vgy,
                vorig,
                vmove,
                medianWidthX,
                medianWidthY
            );
        } finally {
            vsurface.delete();
            vx.delete();
            vy.delete();
            va.delete();
            vsx.delete();
            vsy.delete();
            vgx.delete();
            vgy.delete();
            vorig.delete();
            vmove.delete();
        }

        this._nspectra = nspectra;
        return !!ok;
    }

    setEverything(peakType, maxRounds, clusterIndex) {
        this.instance.set_everything_wasm(peakType, maxRounds, clusterIndex);
    }

    setPeakParas(wx, wy, noise, minHeight, tooNearCutoff, xppmStep, yppmStep, removalCutoff) {
        this.instance.set_peak_paras(wx, wy, noise, minHeight, tooNearCutoff, xppmStep, yppmStep, removalCutoff);
    }

    run() {
        return !!this.instance.run(1);
    }

    changeSign() {
        return !!this.instance.change_sign();
    }

    getResult() {
        return {
            xstart: this.instance.xstart,
            ystart: this.instance.ystart,
            x: vectorToArray(this.instance.x),
            y: vectorToArray(this.instance.y),
            amp: vectorToArray(this.instance.amp),
            sigmax: vectorToArray(this.instance.sigmax),
            sigmay: vectorToArray(this.instance.sigmay),
            gammax: vectorToArray(this.instance.gammax),
            gammay: vectorToArray(this.instance.gammay),
            err: vectorToArray(this.instance.err),
            originalNdx: vectorToArray(this.instance.original_ndx),
            nround: this.instance.get_nround(),
            nspectra: this._nspectra
        };
    }
}

class SpectrumFitter {
    constructor(wasmModule) {
        this.module = wasmModule;
    }

    _extractSpectraArrays(spectrumObj) {
        if (Array.isArray(spectrumObj.spectra) && spectrumObj.spectra.length > 0) {
            return spectrumObj.spectra;
        }
        if (spectrumObj.raw_data && spectrumObj.raw_data.length > 0) {
            return [spectrumObj.raw_data];
        }
        throw new Error("No spectrum data found. Expected spectrum.raw_data or spectrum.spectra[]");
    }

    _buildPeakInput(peaksObj, spectrumObj, spects) {
        const xppm = peaksObj.get_column_by_header("X_PPM");
        const yppm = peaksObj.get_column_by_header("Y_PPM");
        if (xppm.length === 0 || yppm.length === 0 || xppm.length !== yppm.length) {
            throw new Error("Peaks must provide X_PPM and Y_PPM columns with equal size");
        }

        const ass = peaksObj.get_column_by_header("ASS");
        const height = peaksObj.get_column_by_header("HEIGHT");
        const sxIn = peaksObj.get_column_by_header("SIGMAX");
        const syIn = peaksObj.get_column_by_header("SIGMAY");
        const gxIn = peaksObj.get_column_by_header("GAMMAX");
        const gyIn = peaksObj.get_column_by_header("GAMMAY");

        const nDirect = spectrumObj.n_direct;
        const nIndirect = spectrumObj.n_indirect;
        const begin1 = spectrumObj.x_ppm_start;
        const begin2 = spectrumObj.y_ppm_start;
        const step1 = spectrumObj.x_ppm_step;
        const step2 = spectrumObj.y_ppm_step;

        const p1 = [];
        const p2 = [];
        const p1ppm = [];
        const p2ppm = [];
        const pIntensity = [];
        const sigmax = [];
        const sigmay = [];
        const gammax = [];
        const gammay = [];
        const peakIndex = [];
        const comments = [];

        for (let i = 0; i < xppm.length; i++) {
            const xPoint = (xppm[i] - begin1) / step1;
            const yPoint = (yppm[i] - begin2) / step2;
            if (xPoint < 1 || xPoint > nDirect - 2 || yPoint < 1 || yPoint > nIndirect - 2) {
                continue;
            }

            const n1 = Math.min(nDirect - 1, Math.max(0, Math.round(xPoint) - 1));
            const n2 = Math.min(nIndirect - 1, Math.max(0, Math.round(yPoint) - 1));
            const dataHeight = Number(spects[0][n2 * nDirect + n1]);

            p1.push(xPoint);
            p2.push(yPoint);
            p1ppm.push(xppm[i]);
            p2ppm.push(yppm[i]);
            pIntensity.push(height.length > i ? Number(height[i]) : dataHeight);
            sigmax.push(sxIn.length > i ? Number(sxIn[i]) : 3.0);
            sigmay.push(syIn.length > i ? Number(syIn[i]) : 3.0);
            gammax.push(gxIn.length > i ? Number(gxIn[i]) : 1e-20);
            gammay.push(gyIn.length > i ? Number(gyIn[i]) : 1e-20);
            peakIndex.push(i);
            comments.push(ass.length > i ? String(ass[i]) : `peaks${i + 1}`);
        }

        if (p1.length === 0) {
            throw new Error("No valid in-bound peaks remain after coordinate conversion");
        }

        const pIntensityAllSpectra = [];
        for (let i = 0; i < p1.length; i++) {
            const n1 = Math.min(nDirect - 1, Math.max(0, Math.round(p1[i]) - 1));
            const n2 = Math.min(nIndirect - 1, Math.max(0, Math.round(p2[i]) - 1));
            const row = [pIntensity[i]];
            for (let k = 1; k < spects.length; k++) {
                row.push(Number(spects[k][n2 * nDirect + n1]));
            }
            pIntensityAllSpectra.push(row);
        }

        const sxFwhh = [];
        const syFwhh = [];
        for (let i = 0; i < p1.length; i++) {
            sxFwhh.push(0.5346 * gammax[i] * 2.0 + Math.sqrt(0.2166 * 4.0 * gammax[i] * gammax[i] + sigmax[i] * sigmax[i] * 8.0 * 0.6931));
            syFwhh.push(0.5346 * gammay[i] * 2.0 + Math.sqrt(0.2166 * 4.0 * gammay[i] * gammay[i] + sigmay[i] * sigmay[i] * 8.0 * 0.6931));
        }

        let medianWidthX = Math.max(3.0, median(sxFwhh));
        let medianWidthY = Math.max(3.0, median(syFwhh));

        return {
            p1,
            p2,
            p1ppm,
            p2ppm,
            pIntensity,
            pIntensityAllSpectra,
            sigmax,
            sigmay,
            gammax,
            gammay,
            peakIndex,
            comments,
            medianWidthX,
            medianWidthY
        };
    }

    _partitionCore(params, signFlag, peakMap2, peakMap3) {
        const {
            nDirect,
            nIndirect,
            spect,
            noiseLevel,
            userScale2,
            peakMap,
            p1,
            p2,
            sigmax,
            sigmay,
            gammax,
            gammay,
            pIntensityAllSpectra,
            peakIndex,
            spects,
            maxround,
            peakShape,
            wx,
            wy,
            tooNearCutoff,
            step1,
            step2,
            removalCutoff,
            medianWidthX,
            medianWidthY
        } = params;

        const mapFlag = signFlag === 0 ? peakMap2 : peakMap3;
        const lowest = noiseLevel * userScale2;

        const segBegin = Array.from({ length: nIndirect }, () => []);
        const segStop = Array.from({ length: nIndirect }, () => []);
        const used = Array.from({ length: nIndirect }, () => []);

        for (let j = 0; j < nIndirect; j++) {
            if (signFlag === 0) {
                if (spect[j * nDirect + 0] >= lowest && mapFlag[j] === 1) {
                    segBegin[j].push(0);
                }
                for (let i = 1; i < nDirect; i++) {
                    const prevOn = spect[j * nDirect + i - 1] >= lowest && mapFlag[(i - 1) * nIndirect + j] === 1;
                    const currOn = spect[j * nDirect + i] >= lowest && mapFlag[i * nIndirect + j] === 1;
                    if (!prevOn && currOn) {
                        segBegin[j].push(i);
                    }
                    if (prevOn && !currOn) {
                        segStop[j].push(i);
                    }
                }
            } else {
                if (spect[j * nDirect + 0] <= -lowest && mapFlag[j] === 1) {
                    segBegin[j].push(0);
                }
                for (let i = 1; i < nDirect; i++) {
                    const prevOn = spect[j * nDirect + i - 1] <= -lowest && mapFlag[(i - 1) * nIndirect + j] === 1;
                    const currOn = spect[j * nDirect + i] <= -lowest && mapFlag[i * nIndirect + j] === 1;
                    if (!prevOn && currOn) {
                        segBegin[j].push(i);
                    }
                    if (prevOn && !currOn) {
                        segStop[j].push(i);
                    }
                }
            }

            if (segStop[j].length < segBegin[j].length) {
                segStop[j].push(nDirect);
            }
            used[j] = new Array(segStop[j].length).fill(0);
        }

        const clusters = [];
        for (let j = 0; j < nIndirect; j++) {
            for (let i = 0; i < used[j].length; i++) {
                if (used[j][i] === 1) {
                    continue;
                }
                used[j][i] = 1;
                const work = [[j, i]];
                let pos = 0;
                while (pos < work.length) {
                    const c = work[pos++];
                    for (let jj = Math.max(0, c[0] - 1); jj < Math.min(nIndirect, c[0] + 2); jj++) {
                        if (jj === c[0]) {
                            continue;
                        }
                        for (let ii = 0; ii < used[jj].length; ii++) {
                            if (used[jj][ii] === 1) {
                                continue;
                            }
                            if (segStop[jj][ii] >= segBegin[c[0]][c[1]] && segBegin[jj][ii] <= segStop[c[0]][c[1]]) {
                                work.push([jj, ii]);
                                used[jj][ii] = 1;
                            }
                        }
                    }
                }
                clusters.push(work);
            }
        }

        const regionFits = [];
        for (let c = 0; c < clusters.length; c++) {
            let min1 = 1000000;
            let min2 = 1000000;
            let max1 = -1000000;
            let max2 = -1000000;

            for (const [j, i] of clusters[c]) {
                const begin = segBegin[j][i];
                const stop = segStop[j][i];
                if (begin <= min1) min1 = begin;
                if (stop >= max1) max1 = stop;
                if (j <= min2) min2 = j;
                if (j >= max2) max2 = j;
            }
            max1 += 1;
            max2 += 1;

            if (max1 - min1 < 3 || max2 - min2 < 3) {
                continue;
            }

            const regionX = max1 - min1;
            const regionY = max2 - min2;
            const xydim = regionX * regionY;
            const spectParts = new Array(spects.length * xydim).fill(0.0);

            const xx = [];
            const yy = [];
            const sx = [];
            const sy = [];
            const gx = [];
            const gy = [];
            const ori = [];
            const move = [];
            const aas = [];

            for (const [j, i] of clusters[c]) {
                const begin = segBegin[j][i];
                const stop = segStop[j][i];
                for (let kk = begin; kk < stop; kk++) {
                    for (let k = 0; k < spects.length; k++) {
                        spectParts[k * xydim + (kk - min1) * regionY + (j - min2)] = Number(spects[k][kk + j * nDirect]);
                    }

                    const pndx = peakMap[kk * nIndirect + j];
                    if (pndx >= 0) {
                        xx.push(kk - min1);
                        yy.push(j - min2);
                        sx.push(sigmax[pndx]);
                        sy.push(sigmay[pndx]);
                        gx.push(gammax[pndx]);
                        gy.push(gammay[pndx]);
                        ori.push(peakIndex[pndx]);
                        move.push(0);
                        aas.push(...pIntensityAllSpectra[pndx]);
                    }
                }
            }

            if (xx.length === 0) {
                continue;
            }

            regionFits.push({
                clusterLocalIndex: c,
                xstart: min1,
                ystart: min2,
                xdim: regionX,
                ydim: regionY,
                nspectra: spects.length,
                surface: spectParts,
                x: xx,
                y: yy,
                amp: aas,
                sigmax: sx,
                sigmay: sy,
                gammax: gx,
                gammay: gy,
                originalNdx: ori,
                cannotMove: move,
                medianWidthX,
                medianWidthY,
                peakShape,
                maxround,
                peakSign: signFlag === 0 ? 1 : -1,
                peakParas: {
                    wx: wx * 1.5,
                    wy: wy * 1.5,
                    noise: noiseLevel,
                    minHeight: noiseLevel * userScale2,
                    tooNearCutoff,
                    xppmStep: step1,
                    yppmStep: step2,
                    removalCutoff
                }
            });
        }

        return regionFits;
    }

    partitionRegions(input, options) {
        const nDirect = options.nDirect;
        const nIndirect = options.nIndirect;

        const peakMap = new Array(nDirect * nIndirect).fill(-1);
        const peakMap2 = new Array(nDirect * nIndirect).fill(0);
        const peakMap3 = new Array(nDirect * nIndirect).fill(0);

        for (let i = 0; i < input.p1.length; i++) {
            const xx = Math.round(input.p1[i]);
            const yy = Math.round(input.p2[i]);
            if (xx >= 0 && xx < nDirect && yy >= 0 && yy < nIndirect) {
                peakMap[xx * nIndirect + yy] = i;
            }
        }

        const dRange = 1.5;
        for (let i = 0; i < input.p1.length; i++) {
            let xfrom = Math.round(input.p1[i] - options.wx * dRange);
            let xto = Math.round(input.p1[i] + options.wx * dRange);
            let yfrom = Math.round(input.p2[i] - options.wy * dRange) + 1;
            let yto = Math.round(input.p2[i] + options.wy * dRange) + 1;

            xfrom = Math.max(0, xfrom);
            xto = Math.min(nDirect, xto);
            yfrom = Math.max(0, yfrom);
            yto = Math.min(nIndirect, yto);

            for (let m = xfrom; m < xto; m++) {
                for (let n = yfrom; n < yto; n++) {
                    if (input.pIntensity[i] > 0) {
                        peakMap2[m * nIndirect + n] = 1;
                    } else {
                        peakMap3[m * nIndirect + n] = 1;
                    }
                }
            }
        }

        const params = {
            nDirect,
            nIndirect,
            spect: options.spect,
            noiseLevel: options.noiseLevel,
            userScale2: options.userScale2,
            peakMap,
            p1: input.p1,
            p2: input.p2,
            sigmax: input.sigmax,
            sigmay: input.sigmay,
            gammax: input.gammax,
            gammay: input.gammay,
            pIntensityAllSpectra: input.pIntensityAllSpectra,
            peakIndex: input.peakIndex,
            spects: options.spects,
            maxround: options.maxround,
            peakShape: options.peakShape,
            wx: options.wx,
            wy: options.wy,
            tooNearCutoff: options.tooNearCutoff,
            step1: options.step1,
            step2: options.step2,
            removalCutoff: options.removalCutoff,
            medianWidthX: input.medianWidthX,
            medianWidthY: input.medianWidthY
        };

        const positive = this._partitionCore(params, 0, peakMap2, peakMap3);
        const negative = this._partitionCore(params, 1, peakMap2, peakMap3);
        return positive.concat(negative);
    }

    runFitWorkflow(spectrumObj, peaksObj, options = {}) {
        const spects = this._extractSpectraArrays(spectrumObj);
        const nDirect = Number(spectrumObj.n_direct);
        const nIndirect = Number(spectrumObj.n_indirect);
        const noiseLevel = Number(spectrumObj.noise_level || 0.0);
        const step1 = Number(spectrumObj.x_ppm_step);
        const step2 = Number(spectrumObj.y_ppm_step);

        const input = this._buildPeakInput(peaksObj, spectrumObj, spects);

        const wxPoints = options.wxPpm ? Number(options.wxPpm) / Math.abs(step1) : input.medianWidthX * 1.6;
        const wyPoints = options.wyPpm ? Number(options.wyPpm) / Math.abs(step2) : input.medianWidthY * 1.6;

        const fitOptions = {
            nDirect,
            nIndirect,
            spect: spects[0],
            spects,
            noiseLevel,
            step1,
            step2,
            wx: wxPoints,
            wy: wyPoints,
            userScale2: options.userScale2 !== undefined ? Number(options.userScale2) : 3.0,
            tooNearCutoff: options.tooNearCutoff !== undefined ? Number(options.tooNearCutoff) : 0.2e-10,
            removalCutoff: options.removalCutoff !== undefined ? Number(options.removalCutoff) : 0.1,
            maxround: options.maxround !== undefined ? Number(options.maxround) : 20,
            peakShape: options.peakShape !== undefined ? Number(options.peakShape) : 1
        };

        const regions = this.partitionRegions(input, fitOptions);

        const single = [];
        const multi = [];
        for (const region of regions) {
            if (region.x.length === 1) {
                single.push(region);
            } else {
                multi.push(region);
            }
        }

        const allResults = [];
        const runRegion = (region, clusterIndex) => {
            const fit = new GaussianFitWasm(this.module);
            try {
                fit.setEverything(region.peakShape, region.maxround, clusterIndex);
                const okInit = fit.init(region);
                if (!okInit) {
                    return null;
                }
                fit.setPeakParas(
                    region.peakParas.wx,
                    region.peakParas.wy,
                    region.peakParas.noise,
                    region.peakParas.minHeight,
                    region.peakParas.tooNearCutoff,
                    region.peakParas.xppmStep,
                    region.peakParas.yppmStep,
                    region.peakParas.removalCutoff
                );
                const okRun = fit.run();
                if (!okRun) {
                    return null;
                }
                fit.changeSign();
                return fit.getResult();
            } finally {
                fit.delete();
            }
        };

        let clusterCounter = 0;
        for (const region of single) {
            const r = runRegion(region, clusterCounter++);
            if (r) allResults.push(r);
        }
        for (const region of multi) {
            const r = runRegion(region, clusterCounter++);
            if (r) allResults.push(r);
        }

        const gathered = {
            p1: [],
            p2: [],
            p1ppm: [],
            p2ppm: [],
            pIntensity: [],
            pIntensityAllSpectra: [],
            sigmax: [],
            sigmay: [],
            gammax: [],
            gammay: [],
            peakIndex: [],
            err: [],
            nround: [],
            group: [],
            comments: []
        };

        const begin1 = Number(spectrumObj.x_ppm_start);
        const begin2 = Number(spectrumObj.y_ppm_start);

        for (let g = 0; g < allResults.length; g++) {
            const fit = allResults[g];
            for (let i = 0; i < fit.x.length; i++) {
                const absX = fit.x[i] + fit.xstart;
                const absY = fit.y[i] + fit.ystart;
                gathered.p1.push(absX);
                gathered.p2.push(absY);
                gathered.p1ppm.push(begin1 + step1 * absX);
                gathered.p2ppm.push(begin2 + step2 * absY);
                gathered.pIntensity.push(fit.amp[i * fit.nspectra]);
                gathered.sigmax.push(Math.abs(fit.sigmax[i]));
                gathered.sigmay.push(Math.abs(fit.sigmay[i]));
                gathered.gammax.push(Math.abs(fit.gammax[i]));
                gathered.gammay.push(Math.abs(fit.gammay[i]));
                gathered.peakIndex.push(fit.originalNdx[i]);
                gathered.err.push(fit.err[i]);
                gathered.nround.push(fit.nround);
                gathered.group.push(g);
                gathered.comments.push(input.comments[fit.originalNdx[i]] || `peaks${fit.originalNdx[i] + 1}`);

                const row = [];
                const start = i * fit.nspectra;
                for (let k = 0; k < fit.nspectra; k++) {
                    row.push(fit.amp[start + k]);
                }
                gathered.pIntensityAllSpectra.push(row);
            }
        }

        let order = [];
        if (spects.length > 1) {
            order = argsort(gathered.peakIndex);
        } else {
            order = argsort(gathered.pIntensity).reverse();
        }

        const applyOrder = (arr) => order.map((i) => arr[i]);
        for (const key of Object.keys(gathered)) {
            gathered[key] = applyOrder(gathered[key]);
        }

        const fitted = new cpeaks();
        fitted.clear_all_data();
        fitted.comments = [];
        fitted.column_headers = [
            "INDEX",
            "X_AXIS",
            "Y_AXIS",
            "X_PPM",
            "Y_PPM",
            "HEIGHT",
            "DHEIGHT",
            "ASS",
            "CLUSTID",
            "SIGMAX",
            "SIGMAY",
            "GAMMAX",
            "GAMMAY",
            "NROUND"
        ];
        fitted.column_formats = [
            "%5d",
            "%9.3f",
            "%9.3f",
            "%10.6f",
            "%10.6f",
            "%+e",
            "%+e",
            "%s",
            "%4d",
            "%f",
            "%f",
            "%f",
            "%f",
            "%4d"
        ];

        fitted.columns = [
            gathered.peakIndex.map((_, i) => i + 1),
            gathered.p1.map((x) => x + 1.0),
            gathered.p2.map((y) => y + 1.0),
            gathered.p1ppm,
            gathered.p2ppm,
            gathered.pIntensity,
            gathered.err,
            gathered.comments,
            gathered.group,
            gathered.sigmax,
            gathered.sigmay,
            gathered.gammax,
            gathered.gammay,
            gathered.nround
        ];

        spectrumObj.fitted_peaks_object = fitted;

        return {
            regions,
            fitCount: allResults.length,
            peaksFitted: gathered.p1.length,
            gathered,
            fittedPeaksObject: fitted
        };
    }
}

if (typeof module !== "undefined" && module.exports) {
    module.exports = { GaussianFitWasm, SpectrumFitter };
}

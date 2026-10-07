import { MathHelpers } from './MathHelpers.js'
import { MatrixHelpers } from './MatrixHelpers.js'
import { WebGPUHelpers } from './WebGPUHelpers.js'

const WORKGROUP_SIZE = WebGPUHelpers.WORKGROUP_SIZE_2D;

// Query index counter for timestamps
let queryIndex = 0;

const matmulComputeShaderObjects = {};
matmulComputeShaderObjects[MatrixHelpers.kernel.naive] = {
    label: "Matrix Multiplication Naive Compute Shader",
    code: `
        struct sizesStruct {
            sizeMX: u32,
            sizeXY: u32,
            sizeNY: u32,
            extra: u32
        };
        @group(0) @binding(0) var<uniform> sizes: sizesStruct;

        @group(0) @binding(1) var<storage> matrixM: array<f32>;
        @group(0) @binding(2) var<storage> matrixN: array<f32>;
        @group(0) @binding(3) var<storage, read_write> matrixP: array<f32>;

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) globalIdx: vec3u) {
            let px = globalIdx.x;
            let py = globalIdx.y;

            let sizeMX = sizes.sizeMX;
            let sizeXY = sizes.sizeXY;
            let sizeNY = sizes.sizeNY;

            if (px >= sizeMX || py >= sizeNY) {
                return;
            }

            var result = 0.0;
            for (var k = u32(0); k < sizeXY; k++) {
                let m = matrixM[k * sizeMX + px];
                let n = matrixN[py * sizeXY + k];
                result += m * n;
            }
            matrixP[py * sizeMX + px] = result;
        }
    `
};

matmulComputeShaderObjects[MatrixHelpers.kernel.tiled] = {
    label: "Matrix Multiplication Tiled Compute Shader",
    code: `
        struct sizesStruct {
            sizeMX: u32,
            sizeXY: u32,
            sizeNY: u32,
            extra: u32
        };

        var<workgroup> sM: array<f32, ${WORKGROUP_SIZE} * ${WORKGROUP_SIZE}>;
        var<workgroup> sN: array<f32, ${WORKGROUP_SIZE} * ${WORKGROUP_SIZE}>;

        @group(0) @binding(0) var<uniform> sizes: sizesStruct;

        @group(0) @binding(1) var<storage> matrixM: array<f32>;
        @group(0) @binding(2) var<storage> matrixN: array<f32>;
        @group(0) @binding(3) var<storage, read_write> matrixP: array<f32>;

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(
                @builtin(global_invocation_id) globalIdx: vec3u, // blockIdx * blockdim + threadIdx
                @builtin(workgroup_id) blockIdx: vec3u,
                @builtin(local_invocation_id) threadIdx: vec3u) {

            let sizeMX = sizes.sizeMX;
            let sizeXY = sizes.sizeXY;
            let sizeNY = sizes.sizeNY;

            // Cannot do this with workgroupBarrier() - "error: 'workgroupBarrier' must only be called from uniform control flow"
            //if (globalIdx.x >= sizeMX || globalIdx.y >= sizeNY) { return; }

            var result = 0.0;
            var tileFactor = u32(ceil(f32(sizeXY) / ${WORKGROUP_SIZE}));
            for (var t = u32(0); t < tileFactor; t++) {
                var tileOffset = t * ${WORKGROUP_SIZE};
                if (globalIdx.x < sizeMX && (tileOffset + threadIdx.y) < sizeXY) {
                    sM[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = matrixM[globalIdx.x + sizeMX * (tileOffset + threadIdx.y)];
                }  else {
                    sM[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = 0.0;
                }

                if ((tileOffset + threadIdx.x) < sizeXY && globalIdx.y < sizeNY) {
                    sN[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = matrixN[(tileOffset + threadIdx.x) + (globalIdx.y * sizeXY)];
                } else {
                    sN[threadIdx.y * ${WORKGROUP_SIZE} + threadIdx.x] = 0.0;
                }

                workgroupBarrier(); // syncthreads()

                var elementCount = min(${WORKGROUP_SIZE}, sizeXY - tileOffset);
                for (var k = u32(0); k < elementCount; k++) {
                    result += sN[threadIdx.y * ${WORKGROUP_SIZE} + k] * sM[k * ${WORKGROUP_SIZE} + threadIdx.x];
                }

                workgroupBarrier(); // syncthreads()
            }

            if ((globalIdx.x < sizeMX) && (globalIdx.y < sizeNY)) {
                matrixP[globalIdx.y * sizeMX + globalIdx.x] = result;
            }
        }
    `
};

async function webGPUMatrixMultiplication(device, matmulObject, kernel, verbose) {
    const matrixM = matmulObject.matrixM;
    const matrixN = matmulObject.matrixN;
    const matrixP = matmulObject.matrixP;

    const sizeMX = matmulObject.sizeMX;
    const sizeXY = matmulObject.sizeXY;
    const sizeNY = matmulObject.sizeNY;


    /**
     * Set up M Matrix on CPU and GPU
     */
    // Create buffer on GPU for M
    const matrixMGPUBuffer = device.createBuffer({
        label: "M Matrix GPU Buffer",
        mappedAtCreation: true,
        size: matrixM.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    // Copy matrixM to GPU
    const matrixMArrayBuffer = matrixMGPUBuffer.getMappedRange();
    new Float32Array(matrixMArrayBuffer).set(matrixM);
    matrixMGPUBuffer.unmap();

    /**
     * Set up N Matrix on CPU and GPU
     */
    // Create buffer on GPU for N
    // Alternate method of copying data to GPU
    const matrixNGPUBuffer = device.createBuffer({
        label: "N Matrix GPU Buffer",
        size: matrixN.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(matrixNGPUBuffer, 0, matrixN);

    /**
     * Set up result P Matrix on GPU (no copy from host)
     */
    // Create buffer on GPU for P
    const matrixPGPUBuffer = device.createBuffer({
        label: "P Matrix GPU Buffer",
        size: matrixP.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    /**
     * Create uniform for sizes
     */
    const uniformBufferSize = 4 * Uint32Array.BYTES_PER_ELEMENT; // lets store size as float too
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });
    const uniformValues = new Uint32Array([sizeMX, sizeXY, sizeNY, 0]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

    /**
     * Create the compute shader for MatrixMultiplication
     */
    const matmulComputeShaderModule = device.createShaderModule(matmulComputeShaderObjects[kernel]);
    /**
     * Create a Bind Group Layout.
     * The Bing Group Layout connects the different buffers into the locations in shaders.
     * It also sets up the type and read-write permissions
    */
    const matmulCSBindGroupLayout = device.createBindGroupLayout({
        label: "Matrix Multiplication Compute Shader Bind Group Layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.COMPUTE, // Matrix Sizes - Read-only
            buffer: {}
        }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE, // M Matrix Input - Read-only
            buffer: {
                type: "read-only-storage"
            }
        }, {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE, // N Matrix Input - Read only
            buffer: {
                type: "read-only-storage"
            }
        }, {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE, // P Matrix Result - Write
            buffer: {
                type: "storage"
            }
        }]
    });

    /**
     * Create a Bind Group Layout.
     * The Bind Group connects the actual buffers to the locations.
     * This is essentially like the function parameters for calling shaders.
    */
    const matmulCSBindGroup = device.createBindGroup({
            label: "Matrix Multiplication Compute Shader Bind Group",
            layout: matmulCSBindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            }, {
                binding: 1,
                resource: { buffer: matrixMGPUBuffer }
            }, {
                binding: 2,
                resource: { buffer: matrixNGPUBuffer }
            }, {
                binding: 3,
                resource: { buffer: matrixPGPUBuffer }
            }],
        });

    /**
     * Create a pipeline layout.
     * A pipeline layout is a list of bind group layouts that one or more pipelines use. The order of the bind group layouts in the array needs to correspond with the @group attributes in the shaders. (This means that bindGroupLayout is associated with @group(0).)
    */
    const matmulCSPipelineLayout = device.createPipelineLayout({
        label: "Matrix Multiplication Compute Shader Pipeline Layout",
        bindGroupLayouts: [matmulCSBindGroupLayout],
    });

    /**
     * Create a draw and simulation pipelines.
     * The pipeline connects the shaders and the layouts, which in turn connects the buffers and bind groups.
    */
    const matmulCSPipeline = device.createComputePipeline({
        label: "Matrix Multiplication Compute Shader Pipeline",
        layout: matmulCSPipelineLayout,
        compute: {
            module: matmulComputeShaderModule,
            entryPoint: "computeMain",
        }
    });

    /**
     * In order to do pretty much anything else in WebGPU, you need to provide some commands to the GPU instructing it what to do.
     * To do this, have the device create a GPUCommandEncoder, which provides an interface for recording GPU commands.
     * The commands you want to send to the GPU are related to compute (or rendering), so the next step is to use the encoder to begin a Compute Pass.
     */
    const encoder = device.createCommandEncoder();

    /**
     * Performance timers
     */
    queryIndex = 0;
    const { querySet, timestampWrites, performanceResolveBuffer, performanceResultBuffer } = (() => {
        const querySet = device.createQuerySet({
            type: 'timestamp',
            count: 2,
        });
        const timestampWrites = {
            querySet: querySet,
            beginningOfPassWriteIndex: queryIndex,
            endOfPassWriteIndex: queryIndex + 1
        };
        const performanceResolveBuffer = device.createBuffer({
            size: querySet.count * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        const performanceResultBuffer = device.createBuffer({
            size: performanceResolveBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        return { querySet, timestampWrites, performanceResolveBuffer, performanceResultBuffer };
    })();

    /**
     * Compute passes are the actual invocation of compute operations. Each one starts off with a beginComputePass() call,
     * which defines the pipelines, buffers, layouts that are the inputs and outputs.
     * It's important to know that simply making these calls does not cause the GPU to actually do anything. They're just recording commands for the GPU to do later.
     */
    const computePass = encoder.beginComputePass({ timestampWrites });
    computePass.setPipeline(matmulCSPipeline);
    computePass.setBindGroup(0, matmulCSBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(sizeMX / WORKGROUP_SIZE), Math.ceil(sizeNY / WORKGROUP_SIZE));
    computePass.end();

    // Incremement the Query Index counter
    queryIndex += 2;


    // Connect the performance timers.
    encoder.resolveQuerySet(querySet, 0, querySet.count, performanceResolveBuffer, 0);
    if (performanceResultBuffer.mapState === 'unmapped') {
        encoder.copyBufferToBuffer(performanceResolveBuffer, 0, performanceResultBuffer, 0, performanceResultBuffer.size);
    }

    /**
     * In order to create a GPUCommandBuffer, call finish() on the command encoder. The command buffer is an opaque handle to the recorded commands.
     * Submit the command buffer to the GPU using the queue of the GPUDevice.
     * The queue performs all GPU commands, ensuring that their execution is well ordered and properly synchronized.
     * The queue's submit() method takes in an array of command buffers, though in this case you only have one.
     * Once you submit a command buffer, it cannot be used again, so there's no need to hold on to it.
     * If you want to submit more commands, you need to build another command buffer. That's why it's fairly common to see those two steps collapsed into one.
     */
    device.queue.submit([encoder.finish()]);

    const gpu = matmulObject.gpu[kernel];
    if (performanceResultBuffer.mapState === 'unmapped') {
        performanceResultBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const times = new BigInt64Array(performanceResultBuffer.getMappedRange());
            gpu.elapsedTime = Number(times[1] - times[0]) / MathHelpers.TEN_POWER_NINE; // seconds
            performanceResultBuffer.unmap();
        });
    }

    /**
     * Copy the result P Matrix to the GPU to check the computation.
     */
    // Get a GPU buffer for reading in an unmapped state.
    const matrixPStagingBuffer = device.createBuffer({
        size: matrixP.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Create a new encoder as the previous one is no longer usable as a result of calling submit and finish.
    const resultCopyEncoder = device.createCommandEncoder();

    // Encode commands for copying buffer to buffer.
    resultCopyEncoder.copyBufferToBuffer(
        matrixPGPUBuffer /* source buffer */,
        0 /* source offset */,
        matrixPStagingBuffer /* destination buffer */,
        0 /* destination offset */,
        matrixP.byteLength /* size */
    );
    device.queue.submit([resultCopyEncoder.finish()]);

    // Read buffer
    await matrixPStagingBuffer.mapAsync(GPUMapMode.READ);
    const matrixPGPUResult = new Float32Array(matrixPStagingBuffer.getMappedRange());

    /**
     * Compare GPU and CPU results
     */
    let error = false;
    let epsilon = (sizeMX > 1000 || sizeNY > 1000) ? 1e-2 : 1e-3;
    for (let y = 0; y < sizeNY; y++) {
        for (let x = 0; x < sizeMX; x++) {
            const index = y * sizeMX + x;
            if (!MathHelpers.equalsEpsilon(matrixPGPUResult[index], matrixP[index], epsilon)) {
                console.log(`Mismatch Error: GPU = ${matrixPGPUResult[index]} and CPU = ${matrixP[index]} at index ${index} with Epsilon = ${epsilon}.`);
                error = true;
                break;
            }
        }
        if (error)
            break;
    }

    matrixPStagingBuffer.unmap();

    if (!error && verbose) {
        console.log('Results match');
    }

    return !error;
}

async function run(results, device, matmulObject, verbose) {
    const kernelNaive = MatrixHelpers.kernel.naive;
    const kernelTiled = MatrixHelpers.kernel.tiled;

    MatrixHelpers.initMatrixMultiplication(matmulObject);

    if (verbose) {
        console.log('===========================================================;');
        console.log(`[${matmulObject.sizeMX}, ${matmulObject.sizeXY}] x [${matmulObject.sizeXY}, ${matmulObject.sizeNY}]`);
    }

    results.size.mx.push(matmulObject.sizeMX);
    results.size.xy.push(matmulObject.sizeXY);
    results.size.ny.push(matmulObject.sizeNY);

    MatrixHelpers.cpuMatrixMultiplication(matmulObject);
    results.cpuTime.push(matmulObject.cpu.elapsedTime);

    const gpuNaiveSuccess = await webGPUMatrixMultiplication(device, matmulObject, kernelNaive, verbose);
    results.status[kernelNaive].push(gpuNaiveSuccess);
    results.gpuTime[kernelNaive].push(matmulObject.gpu[kernelNaive].elapsedTime);
    if (gpuNaiveSuccess) {
        const timeSpeedUp = matmulObject.cpu.elapsedTime / matmulObject.gpu[kernelNaive].elapsedTime;
        if (verbose) {
            console.log(`Naive Speed Up: ${timeSpeedUp.toFixed(3)}x`);
        }
    }

    const gpuTiledSuccess = await webGPUMatrixMultiplication(device, matmulObject, kernelTiled, verbose);
    results.status[kernelTiled].push(gpuTiledSuccess);
    results.gpuTime[kernelTiled].push(matmulObject.gpu[kernelTiled].elapsedTime);
    if (gpuTiledSuccess) {
        const timeSpeedUp = matmulObject.cpu.elapsedTime / matmulObject.gpu[kernelTiled].elapsedTime;
        if (verbose) {
            console.log(`Tiled Speed Up: ${timeSpeedUp.toFixed(3)}x`);
        }
    }

    if (verbose) {
        console.log('===========================================================');
    }
}

async function matrixMultiplication(mode) {
    const verbose = false;
    const progressElement = document.getElementById('progress');

    const { adapterInfo, device } = await WebGPUHelpers.initGPUDevice(true);

    document.getElementById('gpu-info').innerText = `GPU: ${adapterInfo.description}`;

    const results = {
        mode: mode,
        size: {
            mx: [],
            xy: [],
            ny: []
        },
        status: {
            naive: [],
            tiled: []
        },
        cpuTime: [],
        gpuTime: {
            naive: [],
            tiled: []
        }
    };

    const matmulObject = MatrixHelpers.getEmptyMatmulObject();
    const maxSizeX = (1 << 11) + 1;
    const maxSizeY = (1 << 11) + 1;
    const maxSizeZ = (1 << 11) + 1;

    if (mode === MatrixHelpers.mode.BASIC) {
        for (let size = 16; size < maxSizeX; size *= Math.SQRT2) { // Test NPOT Size too
            matmulObject.sizeMX = matmulObject.sizeXY = matmulObject.sizeNY = Math.trunc(size);
            progressElement.innerText = `Running WebGPU Matrix Multiplication for size [${matmulObject.sizeMX}, ${matmulObject.sizeXY}]`;
            await run(results, device, matmulObject, verbose);
        }
    } else if (mode === MatrixHelpers.mode.ADVANCE) {
        for (let sizeY = 16; sizeY < maxSizeY; sizeY *= Math.SQRT2) { // Test NPOT Size too
            for (let sizeX = 16; sizeX < maxSizeX; sizeX *= Math.SQRT2) { // Test NPOT Size too
                matmulObject.sizeMX = matmulObject.sizeNY = Math.trunc(sizeX);
                matmulObject.sizeXY = Math.trunc(sizeY);
                progressElement.innerText = `Running WebGPU Matrix Multiplication for size [${matmulObject.sizeMX}, ${matmulObject.sizeXY}] x [${matmulObject.sizeXY}, ${matmulObject.sizeNY}]`;
                await run(results, device, matmulObject, verbose);
            }
        }
    } else if (mode === MatrixHelpers.mode.FULL) {
        for (let sizeZ = 16; sizeZ < maxSizeZ; sizeZ *= Math.SQRT2) { // Test NPOT Size too
            for (let sizeY = 16; sizeY < maxSizeY; sizeY *= Math.SQRT2) { // Test NPOT Size too
                for (let sizeX = 16; sizeX < maxSizeX; sizeX *= Math.SQRT2) { // Test NPOT Size too
                    matmulObject.sizeMX = Math.trunc(sizeX);
                    matmulObject.sizeXY = Math.trunc(sizeY);
                    matmulObject.sizeNY = Math.trunc(sizeZ);
                    progressElement.innerText = `Running WebGPU Matrix Multiplication for size [${matmulObject.sizeMX}, ${matmulObject.sizeXY}] x [${matmulObject.sizeXY}, ${matmulObject.sizeNY}]`;
                    await run(results, device, matmulObject, verbose);
                }
            }
        }
    } else {
        console.error('Mode not defined or invalid');
    }

    if (verbose) {
        console.log('***Matrix Multiplication Complete***');
    }

    progressElement.remove();

    if (mode === MatrixHelpers.mode.BASIC) {
        resultsToTableBasic(results);
    } else if (mode === MatrixHelpers.mode.ADVANCE || mode === MatrixHelpers.mode.FULL) {
        resultsToTableAdvance(results);
    }
}

function resultsToTableBasic(results) {
    const kernelNaive = MatrixHelpers.kernel.naive;
    const kernelTiled = MatrixHelpers.kernel.tiled;

    const naiveCPUSpeedUps = results.cpuTime.map((value, index) => value / results.gpuTime[kernelNaive][index]);
    const tiledCPUSpeedUps = results.cpuTime.map((value, index) => value / results.gpuTime[kernelTiled][index]);
    const tiledGPUSpeedUps = results.gpuTime[kernelTiled].map((value, index) => results.gpuTime[kernelNaive][index] / results.gpuTime[kernelTiled][index]);

    const tableData = [
        results.size.xy.map((value) => `${value} x ${value}`),
        results.cpuTime.map((value) => value.toFixed(6)),

        results.status[kernelNaive],
        results.gpuTime[kernelNaive].map((value) => value.toFixed(6)),

        results.status[kernelTiled],
        results.gpuTime[kernelTiled].map((value) => value.toFixed(6)),

        naiveCPUSpeedUps.map((value, index) => (results.status[kernelNaive][index] === true) ? (value.toFixed(3) + 'x') : ''),
        tiledCPUSpeedUps.map((value, index) => (results.status[kernelTiled][index] === true) ? (value.toFixed(3) + 'x') : ''),
        tiledGPUSpeedUps.map((value, index) => (results.status[kernelTiled][index] === true && results.status[kernelNaive][index] === true) ? (value.toFixed(3) + 'x') : '')
    ];

    const columnHeaders = [
        ['Matrix', ''],
        ['CPU Time (sec)', ''],
        ['WebGPU Naive Matmul', 'Success'],
        ['WebGPU Naive Matmul', 'Time (sec)'],
        ['WebGPU Tiled Matmul', 'Success'],
        ['WebGPU Tiled Matmul', 'Time (sec)'],
        ['Speed Up', 'Naive vs CPU'],
        ['Speed Up', 'Tiled vs CPU'],
        ['Speed Up', 'Tiled vs Naive'],
    ];

    function dataListener(x0, y0, x1, y1) {
        return {
            num_rows: results.size.xy.length,
            num_columns: tableData.length,
            data: tableData.slice(x0, x1).map((col) => col.slice(y0, y1)),
            column_headers: columnHeaders
        };
    }

    regularTable.setDataListener(dataListener);

    regularTable.addStyleListener(() => {
        for (const td of regularTable.querySelectorAll("td")) {
            const meta = regularTable.getMeta(td);
            if (meta.column_header[1] === 'Success') {
                td.style.color = meta.value ? 'green' : 'red';
            }
        }
    });

    regularTable.draw();
}

function resultsToTableAdvance(results) {
    const kernelNaive = MatrixHelpers.kernel.naive;
    const kernelTiled = MatrixHelpers.kernel.tiled;

    const naiveCPUSpeedUps = results.cpuTime.map((value, index) => value / results.gpuTime[kernelNaive][index]);
    const tiledCPUSpeedUps = results.cpuTime.map((value, index) => value / results.gpuTime[kernelTiled][index]);
    const tiledGPUSpeedUps = results.gpuTime[kernelTiled].map((value, index) => results.gpuTime[kernelNaive][index] / results.gpuTime[kernelTiled][index]);

    const tableData = [
        results.size.mx.map((value, index) => `[${value}, ${results.size.xy[index]}]`),
        results.size.ny.map((value, index) => `[${results.size.xy[index]}, ${value}]`),
        results.size.mx.map((value, index) => `[${value}, ${results.size.ny[index]}]`),

        results.cpuTime.map((value) => value.toFixed(6)),

        results.status[kernelNaive],
        results.gpuTime[kernelNaive].map((value) => value.toFixed(6)),

        results.status[kernelTiled],
        results.gpuTime[kernelTiled].map((value) => value.toFixed(6)),

        naiveCPUSpeedUps.map((value, index) => (results.status[kernelNaive][index] === true) ? (value.toFixed(3) + 'x') : ''),
        tiledCPUSpeedUps.map((value, index) => (results.status[kernelTiled][index] === true) ? (value.toFixed(3) + 'x') : ''),
        tiledGPUSpeedUps.map((value, index) => (results.status[kernelTiled][index] === true && results.status[kernelNaive][index] === true) ? (value.toFixed(3) + 'x') : '')
    ];

    const columnHeaders = [
        ['Matrix', 'M'], ['Matrix', 'N'], ['Matrix', 'P'],
        ['CPU Time (sec)', ''],
        ['WebGPU Naive Matmul', 'Success'],
        ['WebGPU Naive Matmul', 'Time (sec)'],
        ['WebGPU Tiled Matmul', 'Success'],
        ['WebGPU Tiled Matmul', 'Time (sec)'],
        ['Speed Up', 'Naive vs CPU'],
        ['Speed Up', 'Tiled vs CPU'],
        ['Speed Up', 'Tiled vs Naive'],
    ];

    function dataListener(x0, y0, x1, y1) {
        return {
            num_rows: results.size.xy.length,
            num_columns: tableData.length,
            data: tableData.slice(x0, x1).map((col) => col.slice(y0, y1)),
            column_headers: columnHeaders
        };
    }

    regularTable.setDataListener(dataListener);

    regularTable.addStyleListener(() => {
        for (const td of regularTable.querySelectorAll("td")) {
            const meta = regularTable.getMeta(td);
            if (meta.column_header[1] === 'Success') {
                td.style.color = meta.value ? 'green' : 'red';
            }
        }
    });

    regularTable.draw();
}

const launchButton = document.getElementById("launch");
launchButton.addEventListener("click", async () => {
    const modeElement = document.getElementById("mode");
    mode.disabled = true;
    launchButton.remove();
    await matrixMultiplication(MatrixHelpers.mode[modeElement.value]);
});
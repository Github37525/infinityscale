// ==========================================
// 0. TensorFlow.js 自定义计算层注册 (解决 ESRGAN 报错)
// ==========================================
function registerCustomTfLayers() {
    if (typeof tf === 'undefined') {
        console.warn("[InfinityScale] TensorFlow.js 未加载，AI 超分层注册跳过。");
        return;
    }
    try {
        class MultiplyBeta extends tf.layers.Layer {
            constructor(config) {
                super(config || {});
                this.beta = config && config.beta !== undefined ? config.beta : 0.2;
            }

            computeOutputShape(inputShape) {
                return inputShape;
            }

            call(inputs) {
                return tf.tidy(() => {
                    const input = Array.isArray(inputs) ? inputs[0] : inputs;
                    return input.mul(this.beta);
                });
            }

            static get className() {
                return 'MultiplyBeta';
            }

            getConfig() {
                const config = super.getConfig();
                Object.assign(config, { beta: this.beta });
                return config;
            }
        }
        tf.serialization.registerClass(MultiplyBeta);

        class PixelShuffle extends tf.layers.Layer {
            constructor(config) {
                super(config || {});
                this.scale = config && config.scale !== undefined ? config.scale : 2;
            }

            computeOutputShape(inputShape) {
                const [batch, height, width, channels] = inputShape;
                return [
                    batch,
                    height * this.scale,
                    width * this.scale,
                    channels / (this.scale * this.scale)
                ];
            }

            call(inputs) {
                return tf.tidy(() => {
                    const input = Array.isArray(inputs) ? inputs[0] : inputs;
                    return tf.depthToSpace(input, this.scale, 'NHWC');
                });
            }

            static get className() {
                return 'PixelShuffle';
            }

            getConfig() {
                const config = super.getConfig();
                Object.assign(config, { scale: this.scale });
                return config;
            }
        }
        tf.serialization.registerClass(PixelShuffle);
        console.log("[InfinityScale] TensorFlow.js 自定义层注册成功。");
    } catch (e) {
        console.error("[InfinityScale] 注册 TensorFlow 自定义层失败:", e);
    }
}

// 状态管理
const state = {
    originalImage: null,      // Image 对象
    originalFileName: '',      // 文件名
    originalWidth: 0,
    originalHeight: 0,
    activeAlgo: 'svg',        // 'svg' | 'ai' | 'lanczos'
    aiScale: 2,               // AI 放大倍数 (2 | 4)
    upscaler: null,           // UpscalerJS 实例
    upscalerScale: null,
    resizer: null,
    vtracerWorker: null,
    vtracerWorkerUrl: null,
    vtracerRequestId: 0,
    vtracerPending: new Map(),
    isProcessing: false,
    processingJobId: 0,
    
    // 处理后的数据
    processedType: null,      // 'svg' | 'canvas'
    processedSVGString: '',   // 矢量图源码
    processedCanvas: null,    // 放大后的位图 Canvas
    
    // 对比滑块位置 (0 - 100)
    sliderPosition: 50,
    originalFileSize: 0,
    originalFileType: '',
    recommendation: null,
    crop: { x: 0.3, y: 0.35, width: 0.4, height: 0.3 },
    localPreviewToken: 0
};

const MAX_INPUT_FILE_BYTES = 50 * 1024 * 1024;
const MAX_INPUT_PIXELS = 80000000;
const SAFE_CANVAS_MAX_SIDE = 8192;
const SAFE_CANVAS_MAX_PIXELS = 36000000;

// SVG 预设参数映射
const SVG_PRESETS = {
    typography: {
        ltres: 1.0,
        qtres: 1.0,
        pathomit: 4,
        colorsampling: 0,
        numberofcolors: 2,
        rightangleenhance: true,
        desc: false
    },
    logo: {
        ltres: 1.0,
        qtres: 1.2,
        pathomit: 8,
        colorsampling: 1,
        numberofcolors: 8,
        rightangleenhance: true,
        desc: false
    },
    lineart: {
        ltres: 1.0,
        qtres: 1.2,
        pathomit: 8,
        colorsampling: 0,
        numberofcolors: 2,
        rightangleenhance: false,
        desc: false
    },
    cartoon: {
        ltres: 1.2,
        qtres: 1.5,
        pathomit: 8,
        colorsampling: 2,
        numberofcolors: 12,
        rightangleenhance: false,
        desc: false
    },
    posterized: {
        ltres: 1.5,
        qtres: 1.8,
        pathomit: 12,
        colorsampling: 2,
        numberofcolors: 24,
        rightangleenhance: false,
        desc: false
    },
    photo: {
        ltres: 1.2,
        qtres: 1.5,
        pathomit: 8,
        colorsampling: 2,
        numberofcolors: 32,
        rightangleenhance: false,
        desc: false
    }
};

// 初始化事件绑定
document.addEventListener('DOMContentLoaded', () => {
    registerCustomTfLayers();
    initUI();
    initUpload();
    initCompareSlider();
    initLocalPreview();
    initDpiCalculator();
});

// ==========================================
// 1. UI 交互初始化
// ==========================================
function initUI() {
    // 算法切换按钮
    const algoButtons = document.querySelectorAll('.algo-btn');
    algoButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const algo = btn.dataset.algo;
            switchAlgorithm(algo, { manual: true });
        });
    });

    // 绑定所有的 slider 值更新显示
    const sliders = [
        { id: 'svg-colors', valId: 'val-svg-colors', suffix: '' },
        { id: 'svg-noise', valId: 'val-svg-noise', suffix: ' 像素' },
        { id: 'svg-detail', valId: 'val-svg-detail', suffix: '' },
        { id: 'svg-threshold', valId: 'val-svg-threshold', suffix: '' },
        { id: 'lanczos-scale', valId: 'val-lanczos-scale', suffix: 'X' },
        { id: 'lanczos-smooth', valId: 'val-lanczos-smooth', suffix: '' },
        { id: 'lanczos-color-sig', valId: 'val-lanczos-color-sig', suffix: '' },
        { id: 'lanczos-sharpen', valId: 'val-lanczos-sharpen', suffix: '%' },
        { id: 'clarity-denoise', valId: 'val-clarity-denoise', suffix: '' },
        { id: 'clarity-deblur', valId: 'val-clarity-deblur', suffix: '' },
        { id: 'clarity-contrast', valId: 'val-clarity-contrast', suffix: '' }
    ];

    sliders.forEach(slider => {
        const el = document.getElementById(slider.id);
        const valEl = document.getElementById(slider.valId);
        if (el && valEl) {
            el.addEventListener('input', (e) => {
                valEl.textContent = e.target.value + slider.suffix;
                invalidateProcessedResult('参数已更改，请重新处理');
            });
        }
    });

    // 二值化开关事件联动
    const svgBinarize = document.getElementById('svg-binarize');
    const groupThreshold = document.getElementById('group-svg-threshold');
    if (svgBinarize && groupThreshold) {
        svgBinarize.addEventListener('change', (e) => {
            groupThreshold.style.opacity = e.target.checked ? '1' : '0.4';
            groupThreshold.style.pointerEvents = e.target.checked ? 'auto' : 'none';
            invalidateProcessedResult('参数已更改，请重新处理');
        });
    }

    // SVG 预设切换事件
    const svgPresetSelect = document.getElementById('svg-preset');
    svgPresetSelect.addEventListener('change', (e) => {
        applySvgPreset(e.target.value);
    });

    // AI 放大倍数按钮组
    const aiScaleButtons = document.querySelectorAll('#ai-scale-group .select-btn');
    aiScaleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            aiScaleButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.aiScale = parseInt(btn.dataset.scale);
            const processButton = document.getElementById('btn-process');
            if (processButton) processButton.innerHTML = `<span class="material-symbols-rounded">bolt</span>开始 AI ${state.aiScale}× 超分`;
            invalidateProcessedResult('放大倍数已更改，请重新处理');
        });
    });

    // 开始处理按钮
    const btnProcess = document.getElementById('btn-process');
    btnProcess.addEventListener('click', startProcessing);

    // 清除按钮
    const btnReset = document.getElementById('btn-reset');
    btnReset.addEventListener('click', resetApp);

    // 适应屏幕按钮
    const btnZoomFit = document.getElementById('btn-zoom-fit');
    btnZoomFit.addEventListener('click', () => {
        showToast('已自动缩放至适应预览窗口');
    });

    // 导出下拉菜单
    const btnExport = document.getElementById('btn-export-dropdown');
    const exportMenu = document.getElementById('export-menu');
    btnExport.addEventListener('click', (e) => {
        e.stopPropagation();
        exportMenu.style.display = exportMenu.style.display === 'block' ? 'none' : 'block';
    });

    document.addEventListener('click', () => {
        exportMenu.style.display = 'none';
    });

    // 导出具体选项
    document.getElementById('opt-export-png').addEventListener('click', () => exportImage('png'));
    document.getElementById('opt-export-svg').addEventListener('click', () => exportImage('svg'));
    document.getElementById('opt-export-pdf').addEventListener('click', () => exportImage('pdf'));

    // 清晰度增强面板事件联动
    const clarityEnable = document.getElementById('clarity-enable');
    const clarityControls = document.getElementById('clarity-controls');
    if (clarityEnable && clarityControls) {
        clarityEnable.addEventListener('change', (e) => {
            clarityControls.style.opacity = e.target.checked ? '1' : '0.4';
            clarityControls.style.pointerEvents = e.target.checked ? 'auto' : 'none';
            invalidateProcessedResult('后处理设置已更改，请重新处理');
        });
    }

    const outputScale = document.getElementById('output-scale-select');
    outputScale.addEventListener('change', () => {
        if (!state.originalImage || outputScale.value === 'auto') return;
        const scale = parseFloat(outputScale.value);
        if (state.activeAlgo === 'ai' && (scale === 2 || scale === 4)) {
            state.aiScale = scale;
            document.querySelectorAll('#ai-scale-group .select-btn').forEach(button => {
                button.classList.toggle('active', parseInt(button.dataset.scale) === scale);
            });
            const processButton = document.getElementById('btn-process');
            processButton.innerHTML = `<span class="material-symbols-rounded">bolt</span>开始 AI ${scale}× 超分`;
            invalidateProcessedResult('输出倍率已更改，请重新处理');
        } else {
            switchAlgorithm('lanczos', { manual: true });
            const slider = document.getElementById('lanczos-scale');
            slider.value = String(scale);
            document.getElementById('val-lanczos-scale').textContent = `${scale.toFixed(1)}X`;
            invalidateProcessedResult('输出倍率已更改，请重新处理');
        }
        updateOutputControls();
    });

    const outputDpi = document.getElementById('output-dpi-select');
    outputDpi.addEventListener('change', () => {
        document.getElementById('calc-dpi').value = outputDpi.value;
        calculateRequiredPixels();
        updateStats(Boolean(state.processedType));
    });

    setExportEnabled(false);
    updateOutputControls();
}

function setExportEnabled(enabled) {
    const exportButton = document.getElementById('btn-export-dropdown');
    if (!exportButton) return;
    exportButton.disabled = !enabled;
    exportButton.setAttribute('aria-disabled', String(!enabled));
    if (!enabled) {
        const exportMenu = document.getElementById('export-menu');
        if (exportMenu) exportMenu.style.display = 'none';
    }
    const svgOption = document.getElementById('opt-export-svg');
    const svgChip = document.getElementById('format-svg-chip');
    const svgAvailable = enabled && state.processedType === 'svg';
    if (svgOption) svgOption.disabled = !svgAvailable;
    if (svgChip) svgChip.classList.toggle('available', svgAvailable);
    updateOutputEstimate(enabled);
}

function setProcessingControlsDisabled(disabled) {
    document.querySelectorAll('.algo-btn, #params-container input, #params-container select, #params-container button, #btn-reset, #btn-print-calc')
        .forEach(control => {
            control.disabled = disabled;
        });

    const processButton = document.getElementById('btn-process');
    if (processButton) {
        processButton.disabled = disabled;
        processButton.setAttribute('aria-busy', String(disabled));
    }
}

function showOriginalPreviewOnly(message = '等待处理') {
    const container = document.getElementById('compare-container');
    const afterImg = document.getElementById('compare-img-after');
    const handle = document.getElementById('slider-handle');
    const badge = document.getElementById('badge-after-text');

    if (container) container.style.display = state.originalImage ? 'block' : 'none';
    if (afterImg) {
        afterImg.innerHTML = '';
        afterImg.style.display = 'none';
    }
    if (handle) handle.style.display = 'none';
    if (badge) {
        badge.textContent = message;
        badge.style.display = state.originalImage ? 'block' : 'none';
    }
}

function invalidateProcessedResult(message = '设置已更改，请重新处理') {
    state.processedType = null;
    state.processedSVGString = '';
    state.processedCanvas = null;
    setExportEnabled(false);
    if (state.originalImage) showOriginalPreviewOnly(message);
    updateStats(false);
    updateLocalPreview();
}

// 切换算法模式面板
function switchAlgorithm(algo, options = {}) {
    if (state.isProcessing && !options.force) return;
    const changed = state.activeAlgo !== algo;
    state.activeAlgo = algo;
    
    // 切换按钮高亮
    document.querySelectorAll('.algo-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`algo-${algo}`).classList.add('active');

    // 切换参数面板显示
    document.getElementById('panel-svg').style.display = algo === 'svg' ? 'flex' : 'none';
    document.getElementById('panel-ai').style.display = algo === 'ai' ? 'flex' : 'none';
    document.getElementById('panel-lanczos').style.display = algo === 'lanczos' ? 'flex' : 'none';

    const processButton = document.getElementById('btn-process');
    if (processButton) {
        const label = algo === 'svg' ? '开始矢量追踪' : algo === 'ai' ? `开始 AI ${state.aiScale}× 超分` : '开始忠实重采样';
        processButton.innerHTML = `<span class="material-symbols-rounded">bolt</span>${label}`;
    }

    if (changed) invalidateProcessedResult('算法已更改，请重新处理');

    if (options.manual && state.originalImage) showManualOverride(algo);
    updateOutputControls();

    showToast(`已切换至: ${
        algo === 'svg' ? 'SVG 矢量追踪' : 
        algo === 'ai' ? 'AI 图像超分' : '高质量重采样与边缘优化'
    }`);
}

// 应用 SVG 预设
function applySvgPreset(presetName) {
    const preset = SVG_PRESETS[presetName];
    if (!preset) return;

    const colorsSlider = document.getElementById('svg-colors');
    const noiseSlider = document.getElementById('svg-noise');
    const detailSlider = document.getElementById('svg-detail');
    const colorsGroup = document.getElementById('group-svg-colors');

    colorsSlider.value = preset.numberofcolors;
    document.getElementById('val-svg-colors').textContent = preset.numberofcolors;

    noiseSlider.value = preset.pathomit;
    document.getElementById('val-svg-noise').textContent = preset.pathomit + ' 像素';

    detailSlider.value = preset.ltres;
    document.getElementById('val-svg-detail').textContent = preset.ltres;

    // 自动判断并设置二值化优化开关
    const svgBinarize = document.getElementById('svg-binarize');
    const groupThreshold = document.getElementById('group-svg-threshold');
    if (svgBinarize && groupThreshold) {
        const shouldBinarize = (presetName === 'typography' || presetName === 'lineart');
        svgBinarize.checked = shouldBinarize;
        groupThreshold.style.opacity = shouldBinarize ? '1' : '0.4';
        groupThreshold.style.pointerEvents = shouldBinarize ? 'auto' : 'none';
    }

    // 对于黑白线稿和双色文字，隐藏颜色调节或做标记
    if (presetName === 'lineart' || presetName === 'typography') {
        colorsGroup.style.opacity = '0.5';
    } else {
        colorsGroup.style.opacity = '1';
    }

    invalidateProcessedResult('矢量预设已更改，请重新处理');
}

// ==========================================
// 2. 上传与读取文件
// ==========================================
function initUpload() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const selectBtn = dropzone.querySelector('.dropzone-btn');

    // 点击上传
    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // 拖拽上传
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // 粘贴剪贴板截图
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                handleFile(blob, '剪贴板图像.png');
            }
        }
    });
}

function handleFile(file, customName) {
    if (!file.type.startsWith('image/')) {
        showToast('请上传有效的图像文件！', true);
        return;
    }

    if (file.size > MAX_INPUT_FILE_BYTES) {
        showToast('图片文件超过 50 MB，请先压缩或改用批处理桌面版。', true);
        return;
    }

    state.originalFileName = customName || file.name;
    state.originalFileSize = file.size || 0;
    state.originalFileType = file.type || 'image';
    const reader = new FileReader();

    showLoading('正在加载文件...');

    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            if (img.width * img.height > MAX_INPUT_PIXELS) {
                hideLoading();
                showToast('图片像素超过 8000 万，当前浏览器模式无法安全处理。', true);
                return;
            }

            state.originalImage = img;
            state.originalWidth = img.width;
            state.originalHeight = img.height;
            state.processedType = null;
            state.processedSVGString = '';
            state.processedCanvas = null;

            initializeImageWorkspace(img, file);

            // 显示在 Contrast 滑块的左侧
            const beforeContainer = document.getElementById('compare-img-before');
            beforeContainer.innerHTML = '';
            const cloneImg = img.cloneNode();
            beforeContainer.appendChild(cloneImg);

            // 更新状态底栏
            updateStats();

            // 隐藏上传区域，准备运行
            document.getElementById('dropzone').style.display = 'none';
            document.getElementById('btn-reset').style.display = 'flex';
            
            // 启用底部状态栏
            const bottomBar = document.getElementById('bottom-bar');
            bottomBar.style.opacity = '1';
            bottomBar.style.pointerEvents = 'auto';
            setExportEnabled(false);

            hideLoading();
            const recommendation = analyzeImageForRecommendation(img);
            applyRecommendation(recommendation);
            showToast(`图像加载成功，已推荐“${recommendation.title}”。`);

            showOriginalPreviewOnly('等待处理');
            updateLocalPreview();
        };
        img.onerror = () => {
            hideLoading();
            showToast('图片解码失败，请确认文件没有损坏并使用 PNG、JPG 或 WEBP。', true);
        };
        img.src = e.target.result;
    };
    reader.onerror = () => {
        hideLoading();
        showToast('读取图片失败，请重新选择文件。', true);
    };
    reader.readAsDataURL(file);
}

function initializeImageWorkspace(img, file) {
    const heroImage = document.getElementById('hero-image');
    const heroEmpty = document.getElementById('hero-empty');
    const cropBox = document.getElementById('crop-box');
    const heroMeta = document.getElementById('hero-meta');
    const fileCard = document.getElementById('file-card');

    heroImage.src = img.src;
    heroImage.hidden = false;
    heroEmpty.hidden = true;
    cropBox.hidden = false;
    heroMeta.hidden = false;
    fileCard.hidden = false;

    const sampleWidth = Math.min(state.originalWidth, Math.max(80, Math.min(560, Math.round(state.originalWidth * 0.4))));
    const sampleHeight = Math.min(state.originalHeight, Math.max(40, Math.round(sampleWidth / 2)));
    state.crop.width = sampleWidth / state.originalWidth;
    state.crop.height = sampleHeight / state.originalHeight;
    state.crop.x = Math.max(0, (1 - state.crop.width) / 2);
    state.crop.y = Math.max(0, (1 - state.crop.height) / 2);

    document.getElementById('file-name-display').textContent = state.originalFileName;
    document.getElementById('file-meta-display').textContent = `${state.originalWidth} × ${state.originalHeight} px · ${formatFileSize(file.size || 0)}`;
    document.getElementById('workspace-file-name').textContent = state.originalFileName;
    document.getElementById('hero-dimensions').textContent = `${state.originalWidth} × ${state.originalHeight} px`;

    requestAnimationFrame(positionCropBox);
}

function formatFileSize(bytes) {
    if (!bytes) return '剪贴板图像';
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function analyzeImageForRecommendation(img) {
    const size = 96;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;
    const colors = new Set();
    const gray = new Float32Array(size * size);
    let saturationSum = 0;
    let transparent = 0;

    for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        saturationSum += max === 0 ? 0 : (max - min) / max;
        transparent += data[i + 3] < 245 ? 1 : 0;
        gray[p] = r * 0.299 + g * 0.587 + b * 0.114;
        colors.add(`${r >> 4}-${g >> 4}-${b >> 4}`);
    }

    let edgeSum = 0;
    let edgeCount = 0;
    for (let y = 1; y < size; y += 1) {
        for (let x = 1; x < size; x += 1) {
            const p = y * size + x;
            edgeSum += Math.abs(gray[p] - gray[p - 1]) + Math.abs(gray[p] - gray[p - size]);
            edgeCount += 2;
        }
    }

    const edgeStrength = edgeSum / edgeCount / 255;
    const saturation = saturationSum / (size * size);
    const transparency = transparent / (size * size);
    const lowColorArtwork = colors.size < 180 && edgeStrength > 0.055;
    const smallSource = Math.max(img.width, img.height) < 1200;

    if (lowColorArtwork || (transparency > 0.025 && colors.size < 300)) {
        return {
            algo: 'svg',
            title: 'SVG 矢量追踪',
            kicker: '图形 / 线条特征',
            reason: `检测到较集中的颜色和清晰边缘（约 ${colors.size} 个量化色组），适合先尝试矢量路径。`,
            preset: colors.size < 70 ? 'logo' : 'cartoon'
        };
    }

    if (smallSource && colors.size > 300 && saturation > 0.12) {
        return {
            algo: 'ai',
            title: 'AI 图像超分 2×',
            kicker: '低分辨率照片特征',
            reason: '图像颜色和纹理较复杂、原始边长较小。AI 模型更可能改善观感，但会重建细节。'
        };
    }

    return {
        algo: 'lanczos',
        title: 'Pica 忠实重采样',
        kicker: '连续色调 / 细节充足',
        reason: '原图已有足够像素且颜色层次较多，优先采用忠实重采样，避免模型改写纹理。'
    };
}

function applyRecommendation(recommendation) {
    state.recommendation = recommendation;
    document.getElementById('recommendation-kicker').textContent = recommendation.kicker;
    document.getElementById('recommendation-title').textContent = recommendation.title;
    document.getElementById('recommendation-reason').textContent = recommendation.reason;
    switchAlgorithm(recommendation.algo, { force: true, recommended: true });
    if (recommendation.algo === 'svg' && recommendation.preset) {
        const presetSelect = document.getElementById('svg-preset');
        presetSelect.value = recommendation.preset;
        applySvgPreset(recommendation.preset);
    }
}

function showManualOverride(algo) {
    const names = {
        svg: 'SVG 矢量追踪',
        ai: `AI 图像超分 ${state.aiScale}×`,
        lanczos: 'Pica 忠实重采样'
    };
    document.getElementById('recommendation-kicker').textContent = '已手动覆盖推荐';
    document.getElementById('recommendation-title').textContent = `当前使用：${names[algo]}`;
    const original = state.recommendation?.title;
    document.getElementById('recommendation-reason').textContent = original
        ? `内容分析原本建议“${original}”。当前会按你的选择处理，切换参数后旧结果会立即失效。`
        : '当前会按你的选择处理；切换参数后旧结果会立即失效。';
}

function initLocalPreview() {
    const cropBox = document.getElementById('crop-box');
    let drag = null;

    cropBox.addEventListener('pointerdown', (event) => {
        if (!state.originalImage) return;
        drag = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, x: state.crop.x, y: state.crop.y };
        cropBox.setPointerCapture(event.pointerId);
        event.preventDefault();
    });

    cropBox.addEventListener('pointermove', (event) => {
        if (!drag || drag.pointerId !== event.pointerId) return;
        const rect = getDisplayedImageRect();
        if (!rect.width || !rect.height) return;
        state.crop.x = Math.max(0, Math.min(1 - state.crop.width, drag.x + (event.clientX - drag.clientX) / rect.width));
        state.crop.y = Math.max(0, Math.min(1 - state.crop.height, drag.y + (event.clientY - drag.clientY) / rect.height));
        positionCropBox();
        updateLocalPreview();
    });

    const endDrag = (event) => {
        if (drag && drag.pointerId === event.pointerId) drag = null;
    };
    cropBox.addEventListener('pointerup', endDrag);
    cropBox.addEventListener('pointercancel', endDrag);
    window.addEventListener('resize', positionCropBox);
}

function getDisplayedImageRect() {
    const stage = document.getElementById('hero-stage');
    const rect = stage.getBoundingClientRect();
    if (!state.originalWidth || !state.originalHeight || !rect.width || !rect.height) return { left: 0, top: 0, width: 0, height: 0 };
    const scale = Math.min(rect.width / state.originalWidth, rect.height / state.originalHeight);
    const width = state.originalWidth * scale;
    const height = state.originalHeight * scale;
    return { left: (rect.width - width) / 2, top: (rect.height - height) / 2, width, height };
}

function positionCropBox() {
    if (!state.originalImage) return;
    const box = document.getElementById('crop-box');
    const rect = getDisplayedImageRect();
    box.style.left = `${rect.left + state.crop.x * rect.width}px`;
    box.style.top = `${rect.top + state.crop.y * rect.height}px`;
    box.style.width = `${Math.max(24, state.crop.width * rect.width)}px`;
    box.style.height = `${Math.max(20, state.crop.height * rect.height)}px`;
}

function drawPreviewRegion(source, canvas, sx, sy, sw, sh) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
}

function updateLocalPreview() {
    const beforeCanvas = document.getElementById('local-before-canvas');
    const afterCanvas = document.getElementById('local-after-canvas');
    const beforePlaceholder = document.getElementById('local-before-placeholder');
    const afterPlaceholder = document.getElementById('local-after-placeholder');
    const beforeLabel = document.getElementById('local-before-label');
    const afterLabel = document.getElementById('local-after-label');
    const afterCtx = afterCanvas.getContext('2d');
    const token = ++state.localPreviewToken;

    if (!state.originalImage) {
        beforeCanvas.getContext('2d').clearRect(0, 0, beforeCanvas.width, beforeCanvas.height);
        afterCtx.clearRect(0, 0, afterCanvas.width, afterCanvas.height);
        beforePlaceholder.hidden = false;
        afterPlaceholder.hidden = false;
        beforeLabel.textContent = '等待上传';
        afterLabel.textContent = '等待处理';
        return;
    }

    const sx = Math.round(state.crop.x * state.originalWidth);
    const sy = Math.round(state.crop.y * state.originalHeight);
    const sw = Math.max(1, Math.round(state.crop.width * state.originalWidth));
    const sh = Math.max(1, Math.round(state.crop.height * state.originalHeight));
    drawPreviewRegion(state.originalImage, beforeCanvas, sx, sy, sw, sh);
    beforePlaceholder.hidden = true;
    beforeLabel.textContent = `${sw} × ${sh} px 取样`;

    afterCtx.clearRect(0, 0, afterCanvas.width, afterCanvas.height);
    if (!state.processedType) {
        afterPlaceholder.hidden = false;
        afterLabel.textContent = '设置变更后需重新处理';
        return;
    }

    if (state.processedType === 'canvas' && state.processedCanvas) {
        const scaleX = state.processedCanvas.width / state.originalWidth;
        const scaleY = state.processedCanvas.height / state.originalHeight;
        drawPreviewRegion(state.processedCanvas, afterCanvas, sx * scaleX, sy * scaleY, sw * scaleX, sh * scaleY);
        afterPlaceholder.hidden = true;
        afterLabel.textContent = `${Math.round(sw * scaleX)} × ${Math.round(sh * scaleY)} px 结果`;
        return;
    }

    if (state.processedType === 'svg' && state.processedSVGString) {
        const svgImage = new Image();
        svgImage.onload = () => {
            if (token !== state.localPreviewToken) return;
            const scaleX = svgImage.naturalWidth / state.originalWidth;
            const scaleY = svgImage.naturalHeight / state.originalHeight;
            drawPreviewRegion(svgImage, afterCanvas, sx * scaleX, sy * scaleY, sw * scaleX, sh * scaleY);
            afterPlaceholder.hidden = true;
            afterLabel.textContent = '矢量路径局部渲染';
        };
        svgImage.onerror = () => {
            if (token !== state.localPreviewToken) return;
            afterPlaceholder.hidden = false;
            afterPlaceholder.textContent = '局部渲染失败，可导出后检查';
        };
        svgImage.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(state.processedSVGString)}`;
    }
}

function updateOutputEstimate(enabled) {
    const estimate = document.getElementById('output-estimate');
    if (!estimate) return;
    if (!enabled) {
        estimate.textContent = state.originalImage ? '当前设置尚未处理；旧导出结果已停用。' : '处理完成后可导出。实际文件大小取决于图像内容。';
        return;
    }
    if (state.processedType === 'svg') {
        const kb = Math.max(1, Math.round(new Blob([state.processedSVGString]).size / 1024));
        estimate.textContent = `SVG 路径数据约 ${kb} KB；PNG 与 PDF 大小会因渲染尺寸而变化。`;
        return;
    }
    const megapixels = (state.processedCanvas.width * state.processedCanvas.height / 1000000).toFixed(1);
    estimate.textContent = `结果约 ${megapixels} 百万像素；导出时才会编码，不会预先占用额外文件内存。`;
}

function updateOutputControls() {
    const scaleSelect = document.getElementById('output-scale-select');
    const widthValue = document.getElementById('output-width-value');
    const heightValue = document.getElementById('output-height-value');
    const dpiSelect = document.getElementById('output-dpi-select');
    if (!scaleSelect || !widthValue || !heightValue || !dpiSelect) return;

    dpiSelect.value = document.getElementById('calc-dpi')?.value || '300';
    if (!state.originalImage) {
        scaleSelect.value = 'auto';
        scaleSelect.disabled = true;
        widthValue.textContent = '—';
        heightValue.textContent = '—';
        return;
    }

    if (state.activeAlgo === 'svg') {
        scaleSelect.value = 'auto';
        scaleSelect.disabled = true;
        widthValue.textContent = '矢量';
        heightValue.textContent = '不限';
        return;
    }

    scaleSelect.disabled = false;
    const scale = state.activeAlgo === 'ai'
        ? state.aiScale
        : parseFloat(document.getElementById('lanczos-scale').value || 1);
    scaleSelect.value = String(scale);
    const width = state.processedType === 'canvas' && state.processedCanvas
        ? state.processedCanvas.width
        : Math.round(state.originalWidth * scale);
    const height = state.processedType === 'canvas' && state.processedCanvas
        ? state.processedCanvas.height
        : Math.round(state.originalHeight * scale);
    widthValue.textContent = String(width);
    heightValue.textContent = String(height);
}

// ==========================================
// 3. 对比滑块交互逻辑
// ==========================================
function initCompareSlider() {
    const container = document.getElementById('compare-container');
    const handle = document.getElementById('slider-handle');
    const afterImg = document.getElementById('compare-img-after');
    
    let isDragging = false;

    function move(clientX) {
        const rect = container.getBoundingClientRect();
        const x = clientX - rect.left;
        let percentage = (x / rect.width) * 100;
        
        // 限制边界
        percentage = Math.max(0, Math.min(100, percentage));
        
        handle.style.left = `${percentage}%`;
        afterImg.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
        state.sliderPosition = percentage;
    }

    handle.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
        isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        move(e.clientX);
    });

    // Touch events for mobile
    handle.addEventListener('touchstart', (e) => {
        isDragging = true;
        e.preventDefault();
    }, {passive: false});

    window.addEventListener('touchend', () => {
        isDragging = false;
    });

    window.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        if (e.touches.length > 0) {
            move(e.touches[0].clientX);
        }
    });
}

function resetCompareView() {
    const container = document.getElementById('compare-container');
    const handle = document.getElementById('slider-handle');
    const afterImg = document.getElementById('compare-img-after');
    const badge = document.getElementById('badge-after-text');
    
    container.style.display = 'block';
    afterImg.style.display = 'flex';
    handle.style.display = 'block';
    if (badge) badge.style.display = 'block';
    handle.style.left = '50%';
    afterImg.style.clipPath = 'polygon(50% 0, 100% 0, 100% 100%, 50% 100%)';
    state.sliderPosition = 50;
}

// 智能清晰度增强管线 (WebGL/TFJS GPU 加速版)
async function enhanceClarityGPUAsync(imgData, denoiseVal, deblurVal, contrastVal) {
    let inputTensor = tf.tidy(() => tf.browser.fromPixels(imgData).toFloat().div(255));
    
    // 1. 智能去噪（双边滤波）
    if (denoiseVal > 0) {
        const prevTensor = inputTensor;
        const h = prevTensor.shape[0];
        const w = prevTensor.shape[1];
        const radius = Math.max(1, Math.min(4, Math.round(denoiseVal / 25) + 1));
        const sigmaColor = denoiseVal * 0.6 + 10;
        
        const sigSpaSq = 2 * radius * radius;
        const sigColSq = 2 * (sigmaColor / 255) * (sigmaColor / 255);
        
        let sumImg = tf.tidy(() => tf.zerosLike(prevTensor));
        let sumWeight = tf.tidy(() => tf.zeros([h, w, 1]));
        
        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                const distSq = dx * dx + dy * dy;
                if (distSq > radius * radius) continue;
                
                const spaceWeight = Math.exp(-distSq / sigSpaSq);
                
                const result = tf.tidy(() => {
                    let padLeft = dx < 0 ? -dx : 0;
                    let padRight = dx > 0 ? dx : 0;
                    let padTop = dy < 0 ? -dy : 0;
                    let padBottom = dy > 0 ? dy : 0;
                    
                    let padded = tf.pad(prevTensor, [[padTop, padBottom], [padLeft, padRight], [0, 0]], 'reflect');
                    
                    let sliceY = dy < 0 ? 0 : dy;
                    let sliceX = dx < 0 ? 0 : dx;
                    let shifted = tf.slice(padded, [sliceY, sliceX, 0], [h, w, 3]);
                    
                    const colorDiff = prevTensor.sub(shifted);
                    const colorDiffSq = colorDiff.square().sum(-1, true);
                    const colorWeight = colorDiffSq.div(-sigColSq).exp();
                    
                    const weight = colorWeight.mul(spaceWeight);
                    const termImg = shifted.mul(weight);
                    return { termImg, weight };
                });
                
                const oldSumImg = sumImg;
                const oldSumWeight = sumWeight;
                
                sumImg = sumImg.add(result.termImg);
                sumWeight = sumWeight.add(result.weight);
                
                oldSumImg.dispose();
                oldSumWeight.dispose();
                result.termImg.dispose();
                result.weight.dispose();
                
                await tf.nextFrame();
            }
        }
        
        inputTensor = sumImg.div(sumWeight.add(1e-5)).clipByValue(0, 1);
        sumImg.dispose();
        sumWeight.dispose();
        prevTensor.dispose();
    }
    
    // 2. 对比度与暗部修复 (Gamma自适应提亮 + S曲线对比度拉伸)
    if (contrastVal > 0) {
        const prevTensor = inputTensor;
        inputTensor = tf.tidy(() => {
            // 暗部自适应 Gamma 提亮
            const gamma = 1.0 - (contrastVal / 100) * 0.35;
            const gammaImg = prevTensor.pow(gamma);
            
            // 对比度 S 曲线拉伸
            const gain = 1.0 + (contrastVal / 100) * 0.4;
            const contrastImg = gammaImg.sub(0.5).mul(gain).add(0.5);
            
            return contrastImg.clipByValue(0, 1);
        });
        prevTensor.dispose();
    }
    
    // 3. 去模糊与边缘锐化（USM 锐化）
    if (deblurVal > 0) {
        const prevTensor = inputTensor;
        inputTensor = tf.tidy(() => {
            const sharpenAmount = (deblurVal / 100) * 1.6;
            const kernel2d = tf.tensor2d([
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]).expandDims(2).expandDims(3);
            const kernel = tf.tile(kernel2d, [1, 1, 3, 1]);
            const expanded = prevTensor.expandDims(0);
            const blurred = tf.depthwiseConv2d(expanded, kernel, 1, 'same').squeeze(0);
            
            return prevTensor.add(prevTensor.sub(blurred).mul(sharpenAmount)).clipByValue(0, 1);
        });
        prevTensor.dispose();
    }
    
    // 转化为 uint8 类型
    const finalTensor = tf.tidy(() => inputTensor.mul(255).toInt());
    inputTensor.dispose();
    
    const outCanvas = document.createElement('canvas');
    outCanvas.width = imgData.width;
    outCanvas.height = imgData.height;
    
    await tf.browser.toPixels(finalTensor, outCanvas);
    finalTensor.dispose();
    
    return outCanvas.getContext('2d').getImageData(0, 0, outCanvas.width, outCanvas.height);
}

// CPU Fallback 清晰度增强管线 (使用查表法 LUT 加速对比度映射)
async function enhanceClarityCPUAsync(imgData, denoiseVal, deblurVal, contrastVal) {
    let resultData = imgData;
    
    // 1. Denoise (双边滤波)
    if (denoiseVal > 0) {
        const radius = Math.max(1, Math.min(3, Math.round(denoiseVal / 33)));
        const sigmaColor = denoiseVal * 0.5 + 10;
        resultData = await applyBilateralFilterAsync(resultData, radius, sigmaColor);
    }
    
    // 2. Deblur (锐化)
    if (deblurVal > 0) {
        const amount = (deblurVal / 100) * 0.8;
        resultData = await applyUnsharpMaskAsync(resultData, amount);
    }
    
    // 3. Contrast (自适应 S 曲线与 Gamma 调整)
    if (contrastVal > 0) {
        const width = resultData.width;
        const height = resultData.height;
        const data = resultData.data;
        const output = new Uint8ClampedArray(data.length);
        
        const gamma = 1.0 - (contrastVal / 100) * 0.35;
        const gain = 1.0 + (contrastVal / 100) * 0.4;
        
        // 查表优化
        const lut = new Uint8Array(256);
        for (let i = 0; i < 256; i++) {
            const norm = i / 255;
            const g = Math.pow(norm, gamma);
            const c = (g - 0.5) * gain + 0.5;
            lut[i] = Math.max(0, Math.min(255, Math.round(c * 255)));
        }
        
        for (let i = 0; i < data.length; i += 4) {
            output[i] = lut[data[i]];     // R
            output[i+1] = lut[data[i+1]]; // G
            output[i+2] = lut[data[i+2]]; // B
            output[i+3] = data[i+3];       // A
        }
        resultData = new ImageData(output, width, height);
    }
    
    return resultData;
}

// ==========================================
// 4. 核心图像处理路由
// ==========================================
async function startProcessing() {
    if (!state.originalImage) {
        showToast('请先上传一张图片！', true);
        return;
    }

    if (state.isProcessing) {
        showToast('当前任务仍在处理中，请等待完成。', true);
        return;
    }

    state.isProcessing = true;
    const jobId = ++state.processingJobId;
    setProcessingControlsDisabled(true);
    setExportEnabled(false);
    showLoading('正在准备处理...');

    const backupOrigImage = state.originalImage;
    let completed = false;
    const isClarityEnabled = document.getElementById('clarity-enable').checked;
    
    try {
        if (isClarityEnabled) {
            showLoading('正在应用清晰度后处理...');
            const denoiseVal = parseInt(document.getElementById('clarity-denoise').value);
            const deblurVal = parseInt(document.getElementById('clarity-deblur').value);
            const contrastVal = parseInt(document.getElementById('clarity-contrast').value);
            
            // 创建临时 Canvas 得到原图的 ImageData
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = state.originalWidth;
            tempCanvas.height = state.originalHeight;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(backupOrigImage, 0, 0);
            
            let imgData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            
            let useGPU = false;
            if (typeof tf !== 'undefined') {
                try {
                    const backend = tf.getBackend();
                    const isSoft = isSoftwareWebGL();
                    useGPU = (backend !== 'cpu' && !isSoft);
                } catch (e) {
                    console.warn("TFJS backend check failed in startProcessing:", e);
                }
            }
            
            if (useGPU && (denoiseVal > 0 || deblurVal > 0 || contrastVal > 0)) {
                imgData = await enhanceClarityGPUAsync(imgData, denoiseVal, deblurVal, contrastVal);
            } else {
                imgData = await enhanceClarityCPUAsync(imgData, denoiseVal, deblurVal, contrastVal);
            }
            
            tempCtx.putImageData(imgData, 0, 0);
            
            // 直接将 tempCanvas 代理为 originalImage，规避 toDataURL 开销
            state.originalImage = tempCanvas;
        }

        if (state.activeAlgo === 'svg') {
            await processSVG();
        } else if (state.activeAlgo === 'ai') {
            await processAI();
        } else if (state.activeAlgo === 'lanczos') {
            await processLanczos();
        }
        
        if (jobId !== state.processingJobId) return;

        // 更新底栏参数
        updateStats(true);
        resetCompareView();
        setExportEnabled(true);
        completed = true;
        showToast('处理完成！请放大检查细节后再导出。');
    } catch (err) {
        console.error(err);
        if (jobId === state.processingJobId) {
            invalidateProcessedResult('处理失败，请调整设置后重试');
            showToast(`处理失败: ${err.message}`, true);
        }
    } finally {
        state.originalImage = backupOrigImage; // 总是恢复原图引用
        if (jobId === state.processingJobId) {
            state.isProcessing = false;
            setProcessingControlsDisabled(false);
            hideLoading();
            if (completed) updateLocalPreview();
        }
    }
}

// ==========================================
// 4.1 SVG 矢量化模块 (VTracer WebAssembly)
// ==========================================
function getVTracerWorker() {
    if (state.vtracerWorker) return state.vtracerWorker;

    const wrapperUrl = 'https://cdn.jsdelivr.net/npm/@visioncortex/vtracer@1.0.0-alpha.2/pkg/vtracer_wasm.js';
    const wasmUrl = 'https://cdn.jsdelivr.net/npm/@visioncortex/vtracer@1.0.0-alpha.2/pkg/vtracer_wasm_bg.wasm';
    const workerCode = `
        const engineReady = (async () => {
            const response = await fetch(${JSON.stringify(wasmUrl)});
            if (!response.ok) throw new Error('VTracer WASM 下载失败: ' + response.status);
            const wasmBytes = new Uint8Array(await response.arrayBuffer());
            self.exports = {};
            self.__dirname = '';
            self.require = (name) => {
                if (name === 'fs') return { readFileSync: () => wasmBytes };
                throw new Error('Unsupported VTracer dependency: ' + name);
            };
            importScripts(${JSON.stringify(wrapperUrl)});
            return self.exports;
        })();

        self.onmessage = async (event) => {
            const { id, data, width, height, options } = event.data;
            try {
                const engine = await engineReady;
                const svgString = engine.vectorize_rgba(data, width, height, options);
                self.postMessage({ id, success: true, svgString });
            } catch (error) {
                self.postMessage({ id, success: false, error: error.message || String(error) });
            }
        };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    state.vtracerWorkerUrl = URL.createObjectURL(blob);
    state.vtracerWorker = new Worker(state.vtracerWorkerUrl);
    state.vtracerWorker.onmessage = ({ data }) => {
        const pending = state.vtracerPending.get(data.id);
        if (!pending) return;
        state.vtracerPending.delete(data.id);
        if (data.success) pending.resolve(data.svgString);
        else pending.reject(new Error(data.error || 'VTracer 矢量追踪失败'));
    };
    state.vtracerWorker.onerror = (error) => {
        state.vtracerPending.forEach(({ reject }) => reject(error));
        state.vtracerPending.clear();
        state.vtracerWorker?.terminate();
        state.vtracerWorker = null;
        if (state.vtracerWorkerUrl) URL.revokeObjectURL(state.vtracerWorkerUrl);
        state.vtracerWorkerUrl = null;
    };
    return state.vtracerWorker;
}

function runVTracer(imgData, options) {
    const worker = getVTracerWorker();
    const id = ++state.vtracerRequestId;
    return new Promise((resolve, reject) => {
        state.vtracerPending.set(id, { resolve, reject });
        worker.postMessage({
            id,
            width: imgData.width,
            height: imgData.height,
            data: imgData.data,
            options
        }, [imgData.data.buffer]);
    });
}

async function processSVG() {
    showLoading('正在通过 VTracer WASM 提取矢量路径...');

    const presetName = document.getElementById('svg-preset').value;
    const customColors = parseInt(document.getElementById('svg-colors').value);
    const customNoise = parseInt(document.getElementById('svg-noise').value);
    const customDetail = parseFloat(document.getElementById('svg-detail').value);
    const binarizeEnabled = document.getElementById('svg-binarize').checked;
    const thresholdVal = parseInt(document.getElementById('svg-threshold').value);
    const binaryMode = binarizeEnabled || presetName === 'typography' || presetName === 'lineart';

    const maxInputDimension = 2400;
    const maxTracePixels = 6000000;
    let traceScale = Math.min(1, maxInputDimension / Math.max(state.originalWidth, state.originalHeight));
    traceScale = Math.min(traceScale, Math.sqrt(maxTracePixels / (state.originalWidth * state.originalHeight)));
    const traceWidth = Math.max(1, Math.round(state.originalWidth * traceScale));
    const traceHeight = Math.max(1, Math.round(state.originalHeight * traceScale));

    const canvas = document.createElement('canvas');
    canvas.width = traceWidth;
    canvas.height = traceHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(state.originalImage, 0, 0, traceWidth, traceHeight);
    const imgData = ctx.getImageData(0, 0, traceWidth, traceHeight);

    const options = {
        clustering: binaryMode ? 'bw' : 'color-cluster',
        hierarchical: 'cutout',
        mode: 'spline',
        filterSpeckle: Math.max(customNoise, customColors > 16 ? 8 : 4),
        colorPrecision: presetName === 'photo' ? 7 : 6,
        layerDifference: presetName === 'photo' ? 8 : presetName === 'posterized' ? 18 : 12,
        cornerThreshold: 60,
        lengthThreshold: Math.max(3.5, Math.min(10, customDetail * 4)),
        spliceThreshold: 45,
        simplify: Math.max(0.5, Math.min(2.5, customDetail)),
        pathPrecision: 3,
        maxColors: customColors,
        optimize: 2,
        binaryThreshold: thresholdVal,
        adaptive: false
    };

    const svgString = await runVTracer(imgData, options);
    if (!svgString || !svgString.includes('<svg')) {
        throw new Error('VTracer 未返回有效 SVG');
    }

    state.processedType = 'svg';
    state.processedSVGString = svgString;
    const afterContainer = document.getElementById('compare-img-after');
    afterContainer.innerHTML = svgString;
    const svgEl = afterContainer.querySelector('svg');
    if (svgEl) {
        svgEl.style.width = '100%';
        svgEl.style.height = '100%';
        svgEl.style.maxWidth = '100%';
        svgEl.style.maxHeight = '100%';
    }
    document.getElementById('badge-after-text').textContent = 'VTracer 矢量路径';
}

// 辅助方法：检测 WebGL 是否为 CPU 软件模拟渲染器 (如 SwiftShader / Mesa / llvmpipe)
// 在软件模拟环境下，WebGL 运行极其慢且容易把主线程同步卡死，必须拦截退避
function isSoftwareWebGL() {
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (!gl) return true;
        
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (!debugInfo) return false;
        
        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '';
        console.log("WebGL Unmasked Renderer detected:", renderer);
        
        const lowRenderer = renderer.toLowerCase();
        if (lowRenderer.includes('swiftshader') || 
            lowRenderer.includes('software') || 
            lowRenderer.includes('llvmpipe') || 
            lowRenderer.includes('mesa') || 
            lowRenderer.includes('microsoft basic') || 
            lowRenderer.includes('citrix') ||
            lowRenderer.includes('virtualbox')) {
            return true;
        }
        return false;
    } catch (e) {
        return true;
    }
}

// ==========================================
// 4.2 AI 智能超分模块 (TensorFlow.js + UpscalerJS)
// ==========================================
async function loadUpscaler(mirror = 'jsdelivr') {
    const modelDefinition = state.aiScale === 4 ? window.ESRGANThick4x : window.ESRGANThick2x;
    if (!modelDefinition) {
        throw new Error(`ESRGAN Thick ${state.aiScale}X 模型配置未载入`);
    }

    let modelConfig;
    if (typeof modelDefinition === 'function') {
        modelConfig = modelDefinition(tf);
    } else {
        modelConfig = { ...modelDefinition };
    }
    
    const baseCDN = mirror === 'jsdelivr' 
        ? 'https://cdn.jsdelivr.net/npm/@upscalerjs/esrgan-thick@1.0.0/'
        : 'https://unpkg.com/@upscalerjs/esrgan-thick@1.0.0/';
        
    if (modelConfig.path) {
        modelConfig.path = new URL(modelConfig.path, baseCDN).toString();
    } else {
        modelConfig.path = `${baseCDN}models/x${state.aiScale}/model.json`;
    }
    console.log(`Rewritten modelConfig.path (${mirror}):`, modelConfig.path);

    const upscaler = new Upscaler({
        model: modelConfig
    });

    try {
        // 预热同时验证模型文件确实可以加载。
        await warmupModel(upscaler);
        state.upscaler = upscaler;
        state.upscalerScale = state.aiScale;
    } catch (err) {
        if (typeof upscaler.dispose === 'function') {
            await upscaler.dispose();
        }
        throw err;
    }
}

// 静默预热模型，避免首次推理时编译着色器导致的浏览器同步卡死
async function warmupModel(upscaler = state.upscaler) {
    if (!upscaler) return;
    console.log("Starting WebGL Shaders warmup (8x8 dummy)...");
    const dummyCanvas = document.createElement('canvas');
    dummyCanvas.width = 8;
    dummyCanvas.height = 8;
    const warmupOutput = await upscaler.upscale(dummyCanvas, {
        output: 'tensor',
        patchSize: 8,
        padding: 0,
        awaitNextFrame: true
    });
    if (warmupOutput && typeof warmupOutput.dispose === 'function') {
        warmupOutput.dispose();
    }
    console.log("WebGL Shaders warmup complete.");
}

async function processAI() {
    if (typeof tf === 'undefined') {
        showToast('本地 AI 引擎未载入成功，无法使用 AI 超分！已自动选用快速插值模式。', true);
        switchAlgorithm('lanczos', { force: true });
        await processLanczos();
        return;
    }

    // 综合加速校验：检测 CPU 后端或 CPU 软件模拟渲染 (SwiftShader / llvmpipe 等)
    // 软件渲染下 AI 推理通常过慢，自动切换到重采样模式。
    const backend = tf.getBackend();
    const isSoft = isSoftwareWebGL();
    console.log("TFJS running backend:", backend, "Is Software WebGL:", isSoft);

    if (backend === 'cpu' || isSoft) {
        showToast('您的浏览器未启用 GPU 硬件加速 (或处于软件模拟模式)，已自动为您选用快速插值以防崩溃！', false);
        // 修改当前选中的算法模式为插值
        switchAlgorithm('lanczos', { force: true });
        // 调用 processLanczos 完成放大
        await processLanczos();
        return;
    }

    // 防御性安全限制：AI 模式在输出大像素时计算极其吃力
    // 浏览器端模型的输出像素上限，降低显存溢出的风险。
    const targetScale = state.aiScale; // 2 或 4
    const targetWidth = state.originalWidth * targetScale;
    const targetHeight = state.originalHeight * targetScale;
    const totalTargetPixels = targetWidth * targetHeight;
    const maxAIPixels = 12000000; // 1200万像素安全分水岭 (约 3500x3500px)
    
    if (totalTargetPixels > maxAIPixels) {
        showToast('目标像素过高，为防止显存溢出卡顿，已为您选用快速插值模式！', false);
        // 修改当前选中的算法模式为插值
        switchAlgorithm('lanczos', { force: true });
        // 更新界面上的插值放大 Slider 为匹配的倍数
        const scaleSlider = document.getElementById('lanczos-scale');
        if (scaleSlider) {
            scaleSlider.value = targetScale.toFixed(1);
            const valLabel = document.getElementById('val-lanczos-scale');
            if (valLabel) valLabel.textContent = targetScale.toFixed(1) + 'X';
        }
        // 调用 processLanczos 完成放大
        await processLanczos();
        return;
    }

    showLoading('正在加载 AI 超分模型...');

    if (state.upscaler && state.upscalerScale !== state.aiScale) {
        const oldUpscaler = state.upscaler;
        state.upscaler = null;
        state.upscalerScale = null;
        if (typeof oldUpscaler.dispose === 'function') await oldUpscaler.dispose();
    }

    // 按倍率懒加载对应的 ESRGAN Thick 模型。
    if (!state.upscaler) {
        let lastLoadError = null;
        for (const mirror of ['jsdelivr', 'unpkg']) {
            try {
                await loadUpscaler(mirror);
                state.aiCdnMirror = mirror;
                lastLoadError = null;
                break;
            } catch (err) {
                lastLoadError = err;
                state.upscaler = null;
                console.warn(`AI model load failed from ${mirror}:`, err);
            }
        }
        if (lastLoadError || !state.upscaler) {
            throw new Error(`AI 模型加载失败，请检查网络或离线模型缓存: ${lastLoadError?.message || '未知错误'}`);
        }
    }

    let patch = parseInt(document.getElementById('ai-patch').value);
    // 智能分块防崩自适应：大图下调小分块，以减轻单次 GPU 负载时间，配合 awaitNextFrame 保证界面顺畅刷新
    if (Math.max(state.originalWidth, state.originalHeight) > 1200) {
        console.log(`Original image size exceeds 1200px. Clamping patch size to 128 for safety.`);
        patch = Math.min(patch, 128);
    }
    
    // 更新 Loading 信息
    document.getElementById('progress-container').style.display = 'block';
    updateProgressBar(0);
    
    showLoading('AI 超分模型已就绪，正在分块推理...');

    let upscaledTensor = null;
    let finalTensor = null;
    let finalInt = null;
    try {
        upscaledTensor = await state.upscaler.upscale(state.originalImage, {
            output: 'tensor',
            patchSize: patch,
            padding: 4,
            awaitNextFrame: true,
            progress: (rate) => {
                const percent = Math.round(rate * 100);
                updateProgressBar(percent);
                document.getElementById('loading-text').innerText = `AI 推理中: ${percent}%（分块处理）`;
                console.log("AI progress (pass 1):", percent, "%");
            }
        });

        // 2X 与 4X 分别使用对应倍率训练的模型，不再串联两次 2X 推理。
        finalTensor = upscaledTensor;

        // 终期 Tensor 转换为 Canvas，先强制转换为 0-255 的 int32，防 float32 越界报错
        const outCanvas = document.createElement('canvas');
        outCanvas.width = finalTensor.shape[1];
        outCanvas.height = finalTensor.shape[0];
        
        finalInt = tf.tidy(() => finalTensor.clipByValue(0, 255).toInt());
        await tf.browser.toPixels(finalInt, outCanvas);

        state.processedType = 'canvas';
        state.processedCanvas = outCanvas;

        // 渲染到 Contrast 右侧 (直传 Canvas 节点，免去 Base64 再次解码与 DOM 渲染卡顿)
        const afterContainer = document.getElementById('compare-img-after');
        afterContainer.innerHTML = '';
        outCanvas.style.width = '100%';
        outCanvas.style.height = '100%';
        outCanvas.style.maxWidth = '100%';
        outCanvas.style.maxHeight = '100%';
        afterContainer.appendChild(outCanvas);

        document.getElementById('badge-after-text').textContent = `AI 图像超分 (${state.aiScale}X)`;
    } finally {
        if (finalInt) finalInt.dispose();
        if (finalTensor) finalTensor.dispose();
        if (upscaledTensor && upscaledTensor !== finalTensor) upscaledTensor.dispose();
        document.getElementById('progress-container').style.display = 'none';
    }
}

// ==========================================
// 4.3 浏览器高质量重采样 + 双边滤波
// ==========================================
async function processLanczos() {
    let scale = parseFloat(document.getElementById('lanczos-scale').value);
    const smoothRadius = parseInt(document.getElementById('lanczos-smooth').value);
    const colorSigma = parseInt(document.getElementById('lanczos-color-sig').value);
    const sharpenAmount = parseInt(document.getElementById('lanczos-sharpen').value);

    let targetW = Math.round(state.originalWidth * scale);
    let targetH = Math.round(state.originalHeight * scale);
    let totalTargetPixels = targetW * targetH;

    // 安全像素限制 (防止超大图卷积挂起或超出浏览器 8192px 的 Canvas 单边硬件极限)
    const maxPixels = 36000000; // 3600 万像素安全线
    const maxSide = 8192;       // 8192 像素单边极限值

    if (totalTargetPixels > maxPixels || targetW > maxSide || targetH > maxSide) {
        let ratio = 1.0;
        if (totalTargetPixels > maxPixels) {
            ratio = Math.min(ratio, Math.sqrt(maxPixels / totalTargetPixels));
        }
        if (targetW * ratio > maxSide) {
            ratio = Math.min(ratio, maxSide / targetW);
        }
        if (targetH * ratio > maxSide) {
            ratio = Math.min(ratio, maxSide / targetH);
        }

        scale = scale * ratio;
        targetW = Math.round(state.originalWidth * scale);
        targetH = Math.round(state.originalHeight * scale);
        totalTargetPixels = targetW * targetH;

        showToast(`目标像素过高或超出硬件极限，已自适应缩放到安全规格 (${scale.toFixed(1)}X)`, false);

        // 联动同步修改 UI 滑块和标签数值
        const scaleSlider = document.getElementById('lanczos-scale');
        if (scaleSlider) {
            scaleSlider.value = scale.toFixed(1);
            const valLabel = document.getElementById('val-lanczos-scale');
            if (valLabel) valLabel.textContent = scale.toFixed(1) + 'X';
        }
    }

    showLoading('正在进行高质量重采样...');

    if (typeof window.pica !== 'function') {
        throw new Error('高质量重采样引擎未载入，请检查网络或离线缓存');
    }

    const upscaleCanvas = document.createElement('canvas');
    upscaleCanvas.width = targetW;
    upscaleCanvas.height = targetH;
    const ctx = upscaleCanvas.getContext('2d');
    state.resizer = state.resizer || window.pica({ tile: 1024, concurrency: 2 });
    await state.resizer.resize(state.originalImage, upscaleCanvas, {
        filter: 'mks2013',
        unsharpAmount: 0
    });

    // 检测 GPU 加速状态以选择 WebGL 还是 CPU Fallback 滤波
    let useGPU = false;
    if (typeof tf !== 'undefined') {
        try {
            const backend = tf.getBackend();
            const isSoft = isSoftwareWebGL();
            useGPU = backend !== 'cpu' && !isSoft;
        } catch (error) {
            console.warn('TFJS backend check failed in resize pipeline:', error);
        }
    }
    console.log("Filter pipeline execution backend. GPU acceleration available:", useGPU);

    if (smoothRadius > 0 || sharpenAmount > 0) {
        let imgData = ctx.getImageData(0, 0, targetW, targetH);

        if (useGPU) {
            try {
                imgData = await applyFiltersGPUAsync(imgData, smoothRadius, colorSigma, sharpenAmount / 100);
            } catch (gpuErr) {
                console.warn("GPU filter pipeline failed, fallback to CPU pipeline:", gpuErr);
                if (smoothRadius > 0) {
                    imgData = await applyBilateralFilterAsync(imgData, smoothRadius, colorSigma);
                }
                if (sharpenAmount > 0) {
                    imgData = await applyUnsharpMaskAsync(imgData, sharpenAmount / 100);
                }
            }
        } else {
            if (smoothRadius > 0) {
                imgData = await applyBilateralFilterAsync(imgData, smoothRadius, colorSigma);
            }
            if (sharpenAmount > 0) {
                imgData = await applyUnsharpMaskAsync(imgData, sharpenAmount / 100);
            }
        }

        ctx.putImageData(imgData, 0, 0);
    }

    state.processedType = 'canvas';
    state.processedCanvas = upscaleCanvas;

    // 渲染到 Contrast 右侧 (直传 Canvas 节点，完全省去同步 toDataURL 的巨大时钟开销)
    const afterContainer = document.getElementById('compare-img-after');
    afterContainer.innerHTML = '';
    upscaleCanvas.style.width = '100%';
    upscaleCanvas.style.height = '100%';
    upscaleCanvas.style.maxWidth = '100%';
    upscaleCanvas.style.maxHeight = '100%';
    afterContainer.appendChild(upscaleCanvas);

    document.getElementById('badge-after-text').textContent = `高质量重采样 (${scale.toFixed(1)}X)`;
}

// 一站式 GPU 图像滤波器 (Bilateral Filter + Unsharp Mask)，避免多次从 GPU 读写 ImageData 的大开销
async function applyFiltersGPUAsync(imgData, smoothRadius, colorSigma, sharpenAmount) {
    let inputTensor = tf.tidy(() => tf.browser.fromPixels(imgData).toFloat().div(255));
    
    if (smoothRadius > 0) {
        const prevTensor = inputTensor;
        const h = prevTensor.shape[0];
        const w = prevTensor.shape[1];
        const sigSpaSq = 2 * smoothRadius * smoothRadius;
        const sigColSq = 2 * (colorSigma / 255) * (colorSigma / 255);
        
        let sumImg = tf.tidy(() => tf.zerosLike(prevTensor));
        let sumWeight = tf.tidy(() => tf.zeros([h, w, 1]));
        
        for (let dy = -smoothRadius; dy <= smoothRadius; dy++) {
            for (let dx = -smoothRadius; dx <= smoothRadius; dx++) {
                const distSq = dx * dx + dy * dy;
                if (distSq > smoothRadius * smoothRadius) continue;
                
                const spaceWeight = Math.exp(-distSq / sigSpaSq);
                
                const result = tf.tidy(() => {
                    let padLeft = dx < 0 ? -dx : 0;
                    let padRight = dx > 0 ? dx : 0;
                    let padTop = dy < 0 ? -dy : 0;
                    let padBottom = dy > 0 ? dy : 0;
                    
                    let padded = tf.pad(prevTensor, [[padTop, padBottom], [padLeft, padRight], [0, 0]], 'reflect');
                    
                    let sliceY = dy < 0 ? 0 : dy;
                    let sliceX = dx < 0 ? 0 : dx;
                    let shifted = tf.slice(padded, [sliceY, sliceX, 0], [h, w, 3]);
                    
                    const colorDiff = prevTensor.sub(shifted);
                    const colorDiffSq = colorDiff.square().sum(-1, true);
                    const colorWeight = colorDiffSq.div(-sigColSq).exp();
                    
                    const weight = colorWeight.mul(spaceWeight);
                    const termImg = shifted.mul(weight);
                    return { termImg, weight };
                });
                
                const oldSumImg = sumImg;
                const oldSumWeight = sumWeight;
                
                sumImg = sumImg.add(result.termImg);
                sumWeight = sumWeight.add(result.weight);
                
                oldSumImg.dispose();
                oldSumWeight.dispose();
                result.termImg.dispose();
                result.weight.dispose();
                
                await tf.nextFrame();
            }
        }
        
        inputTensor = sumImg.div(sumWeight.add(1e-5)).clipByValue(0, 1);
        sumImg.dispose();
        sumWeight.dispose();
        prevTensor.dispose();
    }
    
    if (sharpenAmount > 0) {
        const prevTensor = inputTensor;
        inputTensor = tf.tidy(() => {
            const kernel2d = tf.tensor2d([
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]).expandDims(2).expandDims(3);
            const kernel = tf.tile(kernel2d, [1, 1, 3, 1]);
            const expanded = prevTensor.expandDims(0);
            const blurred = tf.depthwiseConv2d(expanded, kernel, 1, 'same').squeeze(0);
            
            return prevTensor.add(prevTensor.sub(blurred).mul(sharpenAmount)).clipByValue(0, 1);
        });
        prevTensor.dispose();
    }
    
    // 转为 uint8 tensor
    const finalTensor = tf.tidy(() => inputTensor.mul(255).toInt());
    inputTensor.dispose();
    
    // 创建一个临时 canvas 并将 finalTensor 画入
    const outCanvas = document.createElement('canvas');
    outCanvas.width = imgData.width;
    outCanvas.height = imgData.height;
    
    await tf.browser.toPixels(finalTensor, outCanvas);
    finalTensor.dispose();
    
    return outCanvas.getContext('2d').getImageData(0, 0, outCanvas.width, outCanvas.height);
}

// 异步图像双边滤波器 (Bilateral Filter)，带时间切片防界面卡死
async function applyBilateralFilterAsync(imgData, radius, sigmaColor) {
    const width = imgData.width;
    const height = imgData.height;
    const input = imgData.data;
    const output = new Uint8ClampedArray(input.length);
    
    const sigColSq = 2 * sigmaColor * sigmaColor;
    const sigSpaSq = 2 * radius * radius;
    
    // 先行计算空间距离的高斯权重表以提高速度
    const spaceWeights = [];
    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            const distSq = dx * dx + dy * dy;
            if (distSq <= radius * radius) {
                spaceWeights.push({
                    dx, dy,
                    w: Math.exp(-distSq / sigSpaSq)
                });
            }
        }
    }

    let lastYield = performance.now();

    // 逐像素处理
    for (let y = 0; y < height; y++) {
        // 每过 40ms 让出一次主线程并更新进度
        if (performance.now() - lastYield > 40) {
            const percent = Math.round((y / height) * 100);
            showLoading(`正在运行双边保边降噪滤波器: ${percent}%...`);
            await new Promise(r => setTimeout(r, 0));
            lastYield = performance.now();
        }
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            
            const r = input[idx];
            const g = input[idx+1];
            const b = input[idx+2];
            
            let sumR = 0, sumG = 0, sumB = 0;
            let totalW = 0;
            
            // 卷积核心
            for (let i = 0; i < spaceWeights.length; i++) {
                const { dx, dy, w: spaceW } = spaceWeights[i];
                
                const nx = Math.min(width - 1, Math.max(0, x + dx));
                const ny = Math.min(height - 1, Math.max(0, y + dy));
                const nIdx = (ny * width + nx) * 4;
                
                const nr = input[nIdx];
                const ng = input[nIdx+1];
                const nb = input[nIdx+2];
                
                // 计算色差
                const colorDistSq = (r-nr)*(r-nr) + (g-ng)*(g-ng) + (b-nb)*(b-nb);
                const colorW = Math.exp(-colorDistSq / sigColSq);
                
                const weight = spaceW * colorW;
                
                sumR += nr * weight;
                sumG += ng * weight;
                sumB += nb * weight;
                totalW += weight;
            }
            
            output[idx] = sumR / totalW;
            output[idx+1] = sumG / totalW;
            output[idx+2] = sumB / totalW;
            output[idx+3] = input[idx+3]; // 保持透明度不变
        }
    }
    
    return new ImageData(output, width, height);
}

/**
 * 非锐化滤镜 (Unsharp Mask)
 * 通过从原图叠加原图与模糊图的差值来实现边缘硬化
 */
// 异步非锐化滤镜 (Unsharp Mask)，带时间切片防界面卡死
async function applyUnsharpMaskAsync(imgData, amount) {
    const width = imgData.width;
    const height = imgData.height;
    const input = imgData.data;
    const output = new Uint8ClampedArray(input.length);
    
    // 简易的高斯模糊/箱式模糊作为低频信号提取
    const blurred = new Uint8ClampedArray(input.length);
    const radius = 1;
    
    let lastYield = performance.now();
    
    // 第一步：模糊滤波阶段 (0% - 50%)
    for (let y = 0; y < height; y++) {
        if (performance.now() - lastYield > 40) {
            const percent = Math.round((y / height) * 50);
            showLoading(`正在运行细节锐化滤波器 (提取低频): ${percent}%...`);
            await new Promise(r => setTimeout(r, 0));
            lastYield = performance.now();
        }
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            let sumR = 0, sumG = 0, sumB = 0, count = 0;
            
            for (let dy = -radius; dy <= radius; dy++) {
                const ny = Math.min(height - 1, Math.max(0, y + dy));
                for (let dx = -radius; dx <= radius; dx++) {
                    const nx = Math.min(width - 1, Math.max(0, x + dx));
                    const nIdx = (ny * width + nx) * 4;
                    sumR += input[nIdx];
                    sumG += input[nIdx+1];
                    sumB += input[nIdx+2];
                    count++;
                }
            }
            
            blurred[idx] = sumR / count;
            blurred[idx+1] = sumG / count;
            blurred[idx+2] = sumB / count;
        }
    }
    
    // 第二步：锐化混合阶段 (50% - 100%)
    for (let i = 0; i < input.length; i += 4) {
        if (i % 40000 === 0 && performance.now() - lastYield > 40) {
            const percent = 50 + Math.round((i / input.length) * 50);
            showLoading(`正在运行细节锐化滤波器 (增强细节): ${percent}%...`);
            await new Promise(r => setTimeout(r, 0));
            lastYield = performance.now();
        }
        output[i] = Math.min(255, Math.max(0, input[i] + (input[i] - blurred[i]) * amount));
        output[i+1] = Math.min(255, Math.max(0, input[i+1] + (input[i+1] - blurred[i+1]) * amount));
        output[i+2] = Math.min(255, Math.max(0, input[i+2] + (input[i+2] - blurred[i+2]) * amount));
        output[i+3] = input[i+3]; // 保持 Alpha
    }
    
    return new ImageData(output, width, height);
}

// ==========================================
// 5. 导出与保存模块
// ==========================================
async function exportImage(format) {
    if (!state.processedType) {
        showToast('请先点击“开始处理图像”！', true);
        return;
    }

    showLoading('正在打包导出文件，请稍候...');

    try {
        const baseName = state.originalFileName.substring(0, state.originalFileName.lastIndexOf('.')) || 'upscaled';
        
        if (format === 'svg') {
            if (state.processedType !== 'svg') {
                showToast('当前结果不是矢量路径，无法导出 SVG；请先使用“SVG 矢量追踪”。', true);
                hideLoading();
                return;
            }
            
            // 导出 SVG 文本文件
            downloadFile(
                'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(state.processedSVGString),
                `${baseName}_traced.svg`
            );
            showToast('SVG 矢量文件已导出，请检查路径和颜色是否符合预期。');
            hideLoading();
        } 
        
        else if (format === 'png') {
            let canvas = null;
            if (state.processedType === 'canvas') {
                canvas = state.processedCanvas;
            } else {
                // 如果当前是 SVG 模式，但用户需要导出高分 PNG，我们需要将 SVG 渲染到高分画布上
                canvas = await renderSvgToHighResCanvas(state.processedSVGString);
            }
            
            showLoading('正在编码 PNG 文件...');
            
            // 使用 canvas.toBlob 异步编码，避开 toDataURL 对主线程的死锁
            canvas.toBlob((blob) => {
                if (!blob) {
                    showToast('PNG 导出失败：无法创建图像二进制包', true);
                    hideLoading();
                    return;
                }
                const blobUrl = URL.createObjectURL(blob);
                downloadFile(blobUrl, `${baseName}_upscaled.png`);
                
                // 延迟释放，给浏览器预留触发下载的时间
                setTimeout(() => {
                    URL.revokeObjectURL(blobUrl);
                }, 1500);

                hideLoading();
                showToast('PNG 图片已导出。');
            }, 'image/png');
        } 
        
        else if (format === 'pdf') {
            // 将处理结果放入用户指定的物理页面，不拉伸、不裁切。
            let canvas = null;
            let width = state.originalWidth;
            let height = state.originalHeight;

            if (state.processedType === 'canvas') {
                canvas = state.processedCanvas;
                width = state.processedCanvas.width;
                height = state.processedCanvas.height;
            } else {
                // SVG 模式：使用高精 Canvas 渲染 (最大边 5000 像素)
                canvas = await renderSvgToHighResCanvas(state.processedSVGString);
                width = state.originalWidth;
                height = state.originalHeight;
            }

            showLoading('正在生成 PDF 文件...');
            
            // 使用 canvas.toBlob 异步编码 PNG，减少主线程上的同步内存峰值。
            canvas.toBlob(async (blob) => {
                if (!blob) {
                    showToast('PDF 导出失败：无法创建图像二进制包', true);
                    hideLoading();
                    return;
                }

                try {
                    // 将 Blob 读为 ArrayBuffer 并转为 Uint8Array
                    const arrayBuffer = await blob.arrayBuffer();
                    const uint8Array = new Uint8Array(arrayBuffer);

                    // 使用 jsPDF
                    const { jsPDF } = window.jspdf;
                    
                    const { widthMm: targetWidthMm, heightMm: targetHeightMm } = getRequestedPrintSpec();
                    const orientation = targetWidthMm >= targetHeightMm ? 'landscape' : 'portrait';
                    const pdf = new jsPDF({
                        orientation: orientation,
                        unit: 'mm',
                        format: [targetWidthMm, targetHeightMm]
                    });

                    // 获取 PDF 的物理宽高
                    const pdfWidth = pdf.internal.pageSize.getWidth();
                    const pdfHeight = pdf.internal.pageSize.getHeight();

                    const imageRatio = width / height;
                    const pageRatio = pdfWidth / pdfHeight;
                    let drawWidth = pdfWidth;
                    let drawHeight = pdfHeight;
                    if (imageRatio > pageRatio) {
                        drawHeight = pdfWidth / imageRatio;
                    } else {
                        drawWidth = pdfHeight * imageRatio;
                    }
                    const offsetX = (pdfWidth - drawWidth) / 2;
                    const offsetY = (pdfHeight - drawHeight) / 2;

                    pdf.addImage(uint8Array, 'PNG', offsetX, offsetY, drawWidth, drawHeight, undefined, 'FAST');
                    pdf.save(`${baseName}_printable.pdf`);
                    showToast('PDF 已按目标页面尺寸导出，图像保持原始比例。');
                } catch (pdfErr) {
                    console.error(pdfErr);
                    showToast(`PDF 导出失败: ${pdfErr.message}`, true);
                } finally {
                    hideLoading();
                }
            }, 'image/png');
        }
    } catch (err) {
        console.error(err);
        showToast(`导出失败: ${err.message}`, true);
        hideLoading();
    }
}

// 辅助方法：将 SVG 字符串渲染为高精度位图 Canvas (返回 Canvas 元素本身)
function renderSvgToHighResCanvas(svgString) {
    return new Promise((resolve, reject) => {
        const originalMaxSide = Math.max(state.originalWidth, state.originalHeight);
        if (!originalMaxSide) {
            reject(new Error('缺少原图尺寸，无法渲染 SVG'));
            return;
        }

        // 默认最多渲染到 5000px；大原图不再被强制放大 4 倍，避免超大 Canvas 内存峰值。
        const desiredMaxSide = Math.min(5000, Math.max(originalMaxSide, originalMaxSide * 4));
        let renderScale = desiredMaxSide / originalMaxSide;
        let targetW = Math.max(1, Math.round(state.originalWidth * renderScale));
        let targetH = Math.max(1, Math.round(state.originalHeight * renderScale));

        const sideRatio = Math.min(1, SAFE_CANVAS_MAX_SIDE / Math.max(targetW, targetH));
        const pixelRatio = Math.min(1, Math.sqrt(SAFE_CANVAS_MAX_PIXELS / (targetW * targetH)));
        const safetyRatio = Math.min(sideRatio, pixelRatio);
        targetW = Math.max(1, Math.floor(targetW * safetyRatio));
        targetH = Math.max(1, Math.floor(targetH * safetyRatio));

        const canvas = document.createElement('canvas');
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext('2d');

        // 构建带有宽高属性的 SVG Blob
        const parser = new DOMParser();
        const doc = parser.parseFromString(svgString, 'image/svg+xml');
        const svgEl = doc.querySelector('svg');
        if (!svgEl) {
            reject(new Error('解析生成的 SVG XML 结构失败，可能文件不完整'));
            return;
        }
        svgEl.setAttribute('width', targetW);
        svgEl.setAttribute('height', targetH);

        const svgBlob = new Blob([new XMLSerializer().serializeToString(svgEl)], { type: 'image/svg+xml;charset=utf-8' });
        const URL = window.URL || window.webkitURL || window;
        const blobURL = URL.createObjectURL(svgBlob);

        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0);
            URL.revokeObjectURL(blobURL);
            resolve(canvas);
        };
        img.onerror = (e) => {
            URL.revokeObjectURL(blobURL);
            reject(new Error('SVG 高分位图渲染失败'));
        };
        img.src = blobURL;
    });
}

function getRequestedPrintSpec() {
    const widthMm = Math.max(10, Math.min(5000, parseFloat(document.getElementById('calc-width').value) || 297));
    const heightMm = Math.max(10, Math.min(5000, parseFloat(document.getElementById('calc-height').value) || 210));
    const dpi = Math.max(1, parseInt(document.getElementById('calc-dpi').value) || 300);
    return { widthMm, heightMm, dpi };
}

function downloadFile(dataUrl, fileName) {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// ==========================================
// 6. 印刷尺寸 DPI 规格计算器
// ==========================================
function initDpiCalculator() {
    const modal = document.getElementById('modal-overlay');
    const btnOpen = document.getElementById('btn-print-calc');
    const btnClose = document.getElementById('modal-close');
    const widthInput = document.getElementById('calc-width');
    const heightInput = document.getElementById('calc-height');
    const dpiSelect = document.getElementById('calc-dpi');
    const btnApply = document.getElementById('btn-apply-calc-scale');

    btnOpen.addEventListener('click', () => {
        modal.classList.add('active');
        calculateRequiredPixels();
    });

    btnClose.addEventListener('click', () => {
        modal.classList.remove('active');
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
    });

    // 联动计算
    [widthInput, heightInput, dpiSelect].forEach(input => {
        input.addEventListener('input', calculateRequiredPixels);
        input.addEventListener('change', calculateRequiredPixels);
    });

    // 应用计算所得尺寸
    btnApply.addEventListener('click', () => {
        if (!state.originalImage) {
            showToast('请先上传图像，再应用印刷参数！', true);
            modal.classList.remove('active');
            return;
        }

        const { widthMm, heightMm, dpi } = getRequestedPrintSpec();
        const targetPixelsW = Math.round((widthMm / 25.4) * dpi);
        const targetPixelsH = Math.round((heightMm / 25.4) * dpi);
        const requiredScale = Math.max(
            targetPixelsW / state.originalWidth,
            targetPixelsH / state.originalHeight
        );
        const clampedScale = Math.max(1.5, Math.min(8.0, requiredScale));
        const sourceRatio = state.originalWidth / state.originalHeight;
        const targetRatio = targetPixelsW / targetPixelsH;
        const aspectMismatch = Math.abs(sourceRatio - targetRatio) / sourceRatio > 0.02;
        
        // 切换到插值算法模式并更新 Slider
        switchAlgorithm('lanczos');
        
        const scaleSlider = document.getElementById('lanczos-scale');
        scaleSlider.value = clampedScale.toFixed(1);
        document.getElementById('val-lanczos-scale').textContent = scaleSlider.value + 'X';
        invalidateProcessedResult('印刷规划已更改，请重新处理');

        modal.classList.remove('active');
        const notes = [];
        if (aspectMismatch) notes.push('页面比例与原图不同，PDF 将留白以避免拉伸或裁切');
        if (requiredScale > 8) notes.push(`所需约 ${requiredScale.toFixed(1)}X，当前浏览器流程最多应用 8.0X`);
        const suffix = notes.length ? `；${notes.join('；')}` : '';
        showToast(`已应用 ${scaleSlider.value}X 重采样参数${suffix}`);
    });
}

function calculateRequiredPixels() {
    const widthMm = parseFloat(document.getElementById('calc-width').value) || 297;
    const heightMm = parseFloat(document.getElementById('calc-height').value) || 210;
    const dpi = parseInt(document.getElementById('calc-dpi').value) || 300;

    // 毫米转英寸 (1 inch = 25.4 mm)
    // 像素 = 英寸 * DPI
    const pixelsW = Math.round((widthMm / 25.4) * dpi);
    const pixelsH = Math.round((heightMm / 25.4) * dpi);

    const resultPixels = document.getElementById('calc-result-pixels');
    const resultDesc = document.getElementById('calc-result-desc');

    resultPixels.textContent = `${pixelsW} x ${pixelsH} 像素`;

    const totalMegapixels = (pixelsW * pixelsH) / 1000000;
    let descText = `约 ${totalMegapixels.toFixed(1)} 百万像素`;

    if (widthMm === 297 && heightMm === 210) {
        descText += ' (A4 纸规格)';
    } else if (widthMm === 420 && heightMm === 297) {
        descText += ' (A3 纸规格)';
    } else if (widthMm === 210 && heightMm === 148) {
        descText += ' (A5 纸规格)';
    }

    resultDesc.textContent = descText;
}

// ==========================================
// 7. 辅助方法与通用组件
// ==========================================
function updateStats(hasProcessed = false) {
    const origDim = document.getElementById('stat-orig-dim');
    const procDim = document.getElementById('stat-proc-dim');
    const physical = document.getElementById('stat-proc-physical');
    const quality = document.getElementById('stat-print-quality');
    const dpiBlock = document.getElementById('stat-dpi-block');
    const arrowIndicator = document.getElementById('stat-arrow-indicator');
    updateOutputControls();

    if (!state.originalImage) {
        origDim.textContent = '0 x 0';
        procDim.textContent = '--';
        physical.textContent = '--';
        quality.textContent = '等待处理结果';
        quality.style.color = 'var(--text-muted)';
        if (arrowIndicator) arrowIndicator.style.display = 'none';
        dpiBlock.style.opacity = '0.5';
        return;
    }

    origDim.textContent = `${state.originalWidth} x ${state.originalHeight}`;

    if (!hasProcessed) {
        if (state.activeAlgo === 'svg') {
            procDim.textContent = '矢量路径';
            physical.textContent = '尺寸不限';
        } else {
            const scale = state.activeAlgo === 'ai'
                ? state.aiScale
                : parseFloat(document.getElementById('lanczos-scale').value || 1);
            procDim.textContent = `${Math.round(state.originalWidth * scale)} x ${Math.round(state.originalHeight * scale)} px`;
            physical.textContent = `${scale.toFixed(1)}× 预估`;
        }
        quality.textContent = '处理完成后计算有效 DPI';
        quality.style.color = 'var(--text-muted)';
        if (arrowIndicator) arrowIndicator.style.display = 'none';
        dpiBlock.style.opacity = '0.5';
        return;
    }

    dpiBlock.style.opacity = '1';
    if (arrowIndicator) arrowIndicator.style.display = 'inline-block';

    let currentW = state.originalWidth;
    let currentH = state.originalHeight;

    if (state.activeAlgo === 'svg') {
        procDim.textContent = '可缩放矢量路径';
        physical.textContent = '导出前检查路径与颜色';
        quality.textContent = '矢量边缘可缩放；还原质量取决于追踪结果';
        quality.style.color = 'var(--secondary)';
    } 
    
    else {
        if (!state.processedCanvas) return;
        currentW = state.processedCanvas.width;
        currentH = state.processedCanvas.height;

        procDim.textContent = `${currentW} x ${currentH} px`;
        
        // 印刷尺寸计算：300 DPI 下，毫米数 = 像素数 / 300 * 25.4
        const printW = Math.round((currentW / 300) * 25.4);
        const printH = Math.round((currentH / 300) * 25.4);
        physical.textContent = `${printW} x ${printH} mm`;

        const { widthMm, heightMm } = getRequestedPrintSpec();
        const effectiveDpi = Math.floor(Math.min(
            currentW / (widthMm / 25.4),
            currentH / (heightMm / 25.4)
        ));
        quality.textContent = `按 ${widthMm} × ${heightMm} mm 页面计算：有效约 ${effectiveDpi} DPI`;
        quality.style.color = effectiveDpi >= 300 ? '#34d399' : effectiveDpi >= 150 ? 'var(--secondary)' : 'var(--text-muted)';
    }
}

async function resetApp() {
    state.processingJobId += 1;
    state.isProcessing = false;
    state.originalImage = null;
    state.originalFileName = '';
    state.originalWidth = 0;
    state.originalHeight = 0;
    state.processedType = null;
    state.processedSVGString = '';
    state.processedCanvas = null;
    state.originalFileSize = 0;
    state.originalFileType = '';
    state.recommendation = null;

    const upscaler = state.upscaler;
    state.upscaler = null;
    state.upscalerScale = null;
    state.aiCdnMirror = null;
    if (upscaler && typeof upscaler.dispose === 'function') {
        try {
            await upscaler.dispose();
        } catch (e) {
            console.warn('Failed to dispose UpscalerJS instance on reset:', e);
        }
    }

    document.getElementById('dropzone').style.display = 'flex';
    document.getElementById('compare-container').style.display = 'none';
    document.getElementById('btn-reset').style.display = 'none';
    document.getElementById('file-input').value = '';
    document.getElementById('compare-img-before').innerHTML = '';
    document.getElementById('compare-img-after').innerHTML = '';
    document.getElementById('badge-after-text').textContent = '等待处理';
    document.getElementById('progress-container').style.display = 'none';
    document.getElementById('file-card').hidden = true;
    document.getElementById('hero-image').hidden = true;
    document.getElementById('hero-image').removeAttribute('src');
    document.getElementById('hero-empty').hidden = false;
    document.getElementById('crop-box').hidden = true;
    document.getElementById('hero-meta').hidden = true;
    document.getElementById('workspace-file-name').textContent = '等待上传';
    document.getElementById('recommendation-kicker').textContent = '等待图片';
    document.getElementById('recommendation-title').textContent = '上传后给出处理建议';
    document.getElementById('recommendation-reason').textContent = '推荐只影响默认选择，你随时可以改用其他方法。';
    hideLoading();
    setProcessingControlsDisabled(false);
    setExportEnabled(false);
    
    const bottomBar = document.getElementById('bottom-bar');
    bottomBar.style.opacity = '0.5';
    bottomBar.style.pointerEvents = 'none';

    updateStats(false);
    updateLocalPreview();
    showToast('重置成功，可以上传新的图像。');
}

// 封装 Progress Bar 更新
function updateProgressBar(percent) {
    const bar = document.getElementById('loading-progress-bar');
    if (bar) {
        bar.style.width = `${percent}%`;
    }
}

// 封装全局 Loading 遮罩
function showLoading(text) {
    const overlay = document.getElementById('loading-overlay');
    const textEl = document.getElementById('loading-text');
    if (overlay && textEl) {
        textEl.innerText = text;
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// 封装全局 Toast 通知
function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    const toastMsg = document.getElementById('toast-message');
    const toastIcon = toast.querySelector('.toast-icon');

    if (toast && toastMsg) {
        toastMsg.textContent = message;
        if (isError) {
            toast.style.borderLeftColor = 'var(--accent)';
            toastIcon.style.color = 'var(--accent)';
            // 改变图标为警告图标
            toastIcon.innerHTML = `<circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>`;
        } else {
            toast.style.borderLeftColor = 'var(--primary)';
            toastIcon.style.color = 'var(--primary)';
            // 恢复勾号图标
            toastIcon.innerHTML = `<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/>`;
        }
        
        toast.classList.add('active');
        
        // 自动隐藏
        clearTimeout(window.toastTimeout);
        window.toastTimeout = setTimeout(() => {
            toast.classList.remove('active');
        }, 3500);
    }
}

// ==========================================
// 8. PWA Service Worker 注册 (提供本地安装与离线使用)
// ==========================================
if ('serviceWorker' in navigator && window.location.protocol.startsWith('http')) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./sw.js')
            .then(reg => console.log('InfinityScale ServiceWorker registered successfully with scope:', reg.scope))
            .catch(err => console.warn('InfinityScale ServiceWorker registration failed:', err));
    });
}

// ==========================================
// 9. PWA 桌面客户端安装按钮交互控制
// ==========================================
let deferredPrompt = null;
const installBtn = document.getElementById('btn-install-pwa');

if (installBtn) {
    window.addEventListener('beforeinstallprompt', (e) => {
        // 阻止 Chrome 67 及更早版本自动展示安装提示
        e.preventDefault();
        // 存放事件以备点击触发
        deferredPrompt = e;
        // 显示安装按钮
        installBtn.style.display = 'flex';
        console.log("'beforeinstallprompt' event was fired, showing install button.");
    });

    installBtn.addEventListener('click', async () => {
        if (!deferredPrompt) return;
        // 弹出浏览器安装面板
        deferredPrompt.prompt();
        // 等待用户反馈
        const { outcome } = await deferredPrompt.userChoice;
        console.log(`User response to the install prompt: ${outcome}`);
        // 无论何种结果，都清空 Prompt 避免重复弹窗
        deferredPrompt = null;
        installBtn.style.display = 'none';
    });

    window.addEventListener('appinstalled', (evt) => {
        console.log('InfinityScale was installed successfully!');
        installBtn.style.display = 'none';
        showToast('桌面客户端安装成功！可直接从系统桌面打开使用。');
    });
}

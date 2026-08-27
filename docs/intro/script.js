const copy = {
  zh: {
    meta: {
      title: 'InfinityScale · 本地优先的图像工作台',
      description: 'InfinityScale：本地优先的浏览器图像放大、矢量追踪与印刷尺寸规划工作台。'
    },
    nav: { demo: '打开 Demo', github: 'GitHub' },
    hero: {
      eyebrow: 'LOCAL-FIRST IMAGE WORKBENCH',
      titleLead: '把像素放大，',
      titleAccent: '把选择说清楚。',
      summary: '在浏览器里比较重采样、AI 超分与矢量追踪，先看局部，再决定导出。',
      primaryCta: '打开图像工作台 ↗',
      secondaryCta: '查看源码',
      privacy: '原图不上传应用服务器 · 首次运行需加载 CDN 运行库',
      imageAlt: 'InfinityScale 图像处理工作台预览'
    },
    signals: { paths: '条处理路径', export: '本地导出', install: '可安装工作台' },
    methods: {
      eyebrow: 'CHOOSE WITH CONTEXT',
      titleLead: '不是一个按钮适合所有图像，',
      titleAccent: '而是一组透明取舍。',
      summary: 'InfinityScale 根据内容特征给出起点建议，同时始终允许你手动覆盖，并展示真正执行的方法。',
      trace: { label: 'VECTOR', title: 'VTracer SVG', body: '把 Logo、文字、签名和线稿转换成可检查的矢量路径。', use: '线条与透明边缘' },
      ai: { label: 'RECONSTRUCT', title: 'ESRGAN Thick 2× / 4×', body: '为照片重建细节；它是生成式超分，不承诺像素级恢复。', use: '照片与纹理' },
      resample: { label: 'RESAMPLE', title: 'Pica MKS2013', body: '用确定性的高质量重采样保留已经干净的插画、渐变与图形。', use: '忠实放大与印刷准备' }
    },
    workflow: {
      eyebrow: 'A CALM WORKFLOW', title: '上传、观察、导出。',
      upload: { title: '上传图片', body: '拖入文件、选择文件，或从剪贴板开始。内容分析只在本地发生。' },
      preview: { title: '看局部对比', body: '拖动取样框，比较原图和结果；需要时覆盖推荐的方法。' },
      export: { title: '规划并导出', body: '查看预计尺寸、目标 DPI 和有效 DPI，再导出 PNG、SVG 或印刷 PDF。' }
    },
    trust: {
      eyebrow: 'TECHNICAL HONESTY', title: '好工具也要说清楚边界。',
      body: 'AI 会重建细节，SVG 是矢量近似，传统重采样更接近像素保真。页面把这些差异放在用户真正做决定的位置。',
      item1: '连续处理不会把旧结果继续导出', item2: '输入与画布有明确的浏览器安全上限', item3: '首次使用不是完全离线，网络边界公开说明'
    },
    cta: { eyebrow: 'OPEN SOURCE, LOCAL BY DEFAULT', titleLead: '从一张图开始，', titleAccent: '把结果看明白。', body: '试用公开或合成图片，欢迎把浏览器、图像类型和结果反馈带回项目。', button: '进入 InfinityScale ↗' },
    footer: { note: 'MIT License · 图片在浏览器内处理' }
  },
  en: {
    meta: {
      title: 'InfinityScale · A local-first image workbench',
      description: 'InfinityScale is a local-first browser workbench for image upscaling, vector tracing, and print-size planning.'
    },
    nav: { demo: 'Open demo', github: 'GitHub' },
    hero: {
      eyebrow: 'LOCAL-FIRST IMAGE WORKBENCH',
      titleLead: 'Scale the pixels.',
      titleAccent: 'Explain the choice.',
      summary: 'Compare resampling, AI super-resolution, and vector tracing in the browser. Inspect a crop before you export.',
      primaryCta: 'Open the workbench ↗',
      secondaryCta: 'View source',
      privacy: 'Source pixels stay out of the app server · first run fetches CDN runtime assets',
      imageAlt: 'InfinityScale image-processing workbench preview'
    },
    signals: { paths: 'processing paths', export: 'local export', install: 'installable workbench' },
    methods: {
      eyebrow: 'CHOOSE WITH CONTEXT',
      titleLead: 'One button cannot fit every image,',
      titleAccent: 'so the trade-offs stay visible.',
      summary: 'InfinityScale recommends a starting point from image features, while keeping manual override and the active method explicit.',
      trace: { label: 'VECTOR', title: 'VTracer SVG', body: 'Turn logos, type, signatures, and line art into inspectable vector paths.', use: 'Lines and transparent edges' },
      ai: { label: 'RECONSTRUCT', title: 'ESRGAN Thick 2× / 4×', body: 'Reconstruct detail for photos; this is generative super-resolution, not pixel-faithful recovery.', use: 'Photos and texture' },
      resample: { label: 'RESAMPLE', title: 'Pica MKS2013', body: 'Use deterministic high-quality resampling for clean illustrations, gradients, and graphics.', use: 'Faithful scaling and print prep' }
    },
    workflow: {
      eyebrow: 'A CALM WORKFLOW', title: 'Upload, inspect, export.',
      upload: { title: 'Upload an image', body: 'Drop a file, choose one, or start from the clipboard. Content analysis stays local.' },
      preview: { title: 'Inspect a crop', body: 'Drag the sample region to compare source and result, then override the recommendation when needed.' },
      export: { title: 'Plan and export', body: 'Check predicted dimensions, target DPI, and effective DPI before exporting PNG, SVG, or print PDF.' }
    },
    trust: {
      eyebrow: 'TECHNICAL HONESTY', title: 'A useful tool should name its boundaries.',
      body: 'AI reconstructs detail, SVG is a vector approximation, and traditional resampling stays closer to the source pixels. Those differences appear where decisions happen.',
      item1: 'Repeated processing never keeps an old result exportable', item2: 'Input and canvas safety limits are explicit', item3: 'First use is not fully offline; network boundaries are documented'
    },
    cta: { eyebrow: 'OPEN SOURCE, LOCAL BY DEFAULT', titleLead: 'Start with one image.', titleAccent: 'See the result clearly.', body: 'Try a public-domain or synthetic image, then bring your browser, image class, and result back to the project.', button: 'Enter InfinityScale ↗' },
    footer: { note: 'MIT License · Images are processed in the browser' }
  }
};

function getValue(source, path) {
  return path.split('.').reduce((value, key) => value && value[key], source);
}

function renderLanguage(language) {
  const selected = copy[language] ? language : 'zh';
  const strings = copy[selected];
  document.documentElement.lang = selected === 'zh' ? 'zh-CN' : 'en';
  document.title = strings.meta.title;
  document.querySelector('meta[name="description"]').setAttribute('content', strings.meta.description);

  document.querySelectorAll('[data-i18n]').forEach((element) => {
    const value = getValue(strings, element.dataset.i18n);
    if (value !== undefined) element.textContent = value;
  });

  document.querySelectorAll('[data-i18n-attr]').forEach((element) => {
    element.dataset.i18nAttr.split(',').forEach((entry) => {
      const [attribute, key] = entry.split(':');
      const value = getValue(strings, key);
      if (value !== undefined) element.setAttribute(attribute, value);
    });
  });

  document.querySelectorAll('[data-lang]').forEach((button) => {
    const active = button.dataset.lang === selected;
    button.setAttribute('aria-pressed', String(active));
  });

  try { localStorage.setItem('infinityscale-intro-language', selected); } catch {}
}

const requestedLanguage = new URLSearchParams(window.location.search).get('lang');
let savedLanguage = '';
try { savedLanguage = localStorage.getItem('infinityscale-intro-language') || ''; } catch {}
const initialLanguage = requestedLanguage === 'en' || requestedLanguage === 'zh'
  ? requestedLanguage
  : savedLanguage === 'en' || savedLanguage === 'zh'
    ? savedLanguage
    : navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en';

document.querySelectorAll('[data-lang]').forEach((button) => {
  button.addEventListener('click', () => renderLanguage(button.dataset.lang));
});

renderLanguage(initialLanguage);

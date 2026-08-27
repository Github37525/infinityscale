import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const [html, app, manifest, serviceWorker, intro, introScript] = await Promise.all([
  readFile(new URL('../index.html', import.meta.url), 'utf8'),
  readFile(new URL('../app.js', import.meta.url), 'utf8'),
  readFile(new URL('../manifest.json', import.meta.url), 'utf8'),
  readFile(new URL('../sw.js', import.meta.url), 'utf8'),
  readFile(new URL('../intro/index.html', import.meta.url), 'utf8'),
  readFile(new URL('../intro/script.js', import.meta.url), 'utf8')
]);

const ids = [...app.matchAll(/getElementById\(['"]([^'"]+)['"]\)/g)].map((match) => match[1]);
const missingIds = [...new Set(ids)].filter((id) => !new RegExp(`id=["']${id}["']`).test(html));

assert.deepEqual(missingIds, [], `Missing HTML ids referenced by app.js: ${missingIds.join(', ')}`);
assert.match(html, /内容分析推荐/);
assert.match(html, /局部预览/);
assert.match(html, /输出与导出/);
assert.doesNotMatch(`${html}\n${manifest}`, /极速无损|无损恢复|完全离线/);
assert.match(serviceWorker, /infinityscale-cache-v\d+/);
assert.equal(JSON.parse(manifest).short_name, 'InfinityScale');
assert.match(intro, /data-lang="zh"/);
assert.match(intro, /data-lang="en"/);
assert.match(intro, /\.\/styles\.css/);
assert.match(intro, /\.\/script\.js/);
assert.match(introScript, /infinityscale-intro-language/);
assert.match(introScript, /ESRGAN Thick 2× \/ 4×/);
assert.doesNotMatch(`${intro}\n${introScript}`, /极速无损|AI 去模糊/);
assert.match(`${intro}\n${introScript}`, /不是完全离线|not fully offline/);

console.log(`Smoke checks passed: ${new Set(ids).size} static UI ids verified.`);

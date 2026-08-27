import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const [html, app, manifest, serviceWorker] = await Promise.all([
  readFile(new URL('../index.html', import.meta.url), 'utf8'),
  readFile(new URL('../app.js', import.meta.url), 'utf8'),
  readFile(new URL('../manifest.json', import.meta.url), 'utf8'),
  readFile(new URL('../sw.js', import.meta.url), 'utf8')
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

console.log(`Smoke checks passed: ${new Set(ids).size} static UI ids verified.`);


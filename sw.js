const CACHE_NAME = 'infinityscale-cache-v3';
const ASSETS_TO_CACHE = [
  './index.html',
  './styles.css',
  './app.js',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png'
];

// 缓存外部核心 UMD CDN 库
const CDN_URLS = [
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/upscaler@1.0.0-beta.19/dist/browser/umd/upscaler.min.js',
  'https://cdn.jsdelivr.net/npm/@upscalerjs/esrgan-thick@1.0.0/dist/umd/models/esrgan-thick/src/x2/index.min.js',
  'https://cdn.jsdelivr.net/npm/@upscalerjs/esrgan-thick@1.0.0/dist/umd/models/esrgan-thick/src/x4/index.min.js',
  'https://cdn.jsdelivr.net/npm/pica@10.0.3/dist/pica.min.js',
  'https://cdn.jsdelivr.net/npm/@visioncortex/vtracer@1.0.0-alpha.2/pkg/vtracer_wasm.js',
  'https://cdn.jsdelivr.net/npm/@visioncortex/vtracer@1.0.0-alpha.2/pkg/vtracer_wasm_bg.wasm',
  'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Caching initial shell assets...');
      // 使用 Promise.allSettled 确保某些网络抖动不会导致整个 install 失败
      const cachePromises = [...ASSETS_TO_CACHE, ...CDN_URLS].map(url => {
        return cache.add(url).catch(err => {
          console.warn(`Failed to cache asset: ${url}`, err);
        });
      });
      return Promise.all(cachePromises);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cache) => {
          if (cache !== CACHE_NAME) {
            console.log('Clearing old cache:', cache);
            return caches.delete(cache);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  // 仅拦截 http 或 https 请求，过滤本地 file:// 协议，防止本地运行时报错
  if (!event.request.url.startsWith('http')) return;

  const requestUrl = new URL(event.request.url);
  const isAppShell = requestUrl.origin === self.location.origin;

  if (isAppShell) {
    event.respondWith(
      fetch(event.request).then((response) => {
        if (response && response.status === 200) {
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, response.clone()));
        }
        return response;
      }).catch(() => caches.match(event.request).then((cached) => cached || caches.match('./index.html')))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => cachedResponse || fetch(event.request).then((response) => {
        // 对于第三方 CDN 库或模型，如果是 200 OK，就动态存入缓存
        if (response && (response.status === 200 || response.type === 'opaque')) {
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return response;
      }).catch((err) => {
        console.warn('Fetch failed, resource offline: ', event.request.url, err);
        return Response.error();
      }))
  );
});

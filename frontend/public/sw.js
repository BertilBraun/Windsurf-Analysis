const CACHE_NAME = 'windsurf-video-cache-v2'
const VIDEO_URLS = ['/Surfer1.av1.mp4', '/Surfer1.mp4', '/Surfer2.av1.mp4', '/Surfer2.mp4']

self.addEventListener('install', (event) => {
    self.skipWaiting()
})

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys.map((key) => {
                    if (key !== CACHE_NAME) {
                        return caches.delete(key)
                    }
                    return null
                })
            )
        }).then(() => self.clients.claim())
    )
})

self.addEventListener('fetch', (event) => {
    if (event.request.method !== 'GET') {
        return
    }

    const url = new URL(event.request.url)
    if (!VIDEO_URLS.includes(url.pathname)) {
        return
    }

    event.respondWith(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.match(event.request).then((cached) => {
                if (cached) {
                    return cached
                }
                if (event.request.headers.has('range')) {
                    return fetch(event.request)
                }
                return fetch(event.request).then((response) => {
                    if (response && response.ok) {
                        cache.put(event.request, response.clone())
                    }
                    return response
                })
            })
        })
    )
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'

export default defineConfig({
    resolve: {
        alias: {
            buffer: 'rollup-plugin-node-polyfills/polyfills/buffer-es6',
            process: 'rollup-plugin-node-polyfills/polyfills/process-es6',
        },
    },
    define: {
        global: 'globalThis',
    },
    plugins: [
        react(),
        nodePolyfills({
            globals: {
                Buffer: true,
                global: true,
                process: true,
            },
        }),
    ],
    server: {
        port: 5173,
        host: true,
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
    },
})

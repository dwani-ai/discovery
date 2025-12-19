import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

export default defineConfig({
  plugins: [react()],
    build: {
    outDir: 'build',
  },

  server: {
    host: true, // This allows binding to 0.0.0.0 (needed for Docker/exposed access)
    port: 5173,
    allowedHosts: [
      'localhost',
      '127.0.0.1',
      'app.dwani.ai',        // ← Add your custom domain here
    ],
  },
});
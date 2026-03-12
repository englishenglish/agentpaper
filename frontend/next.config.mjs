/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
      {
        source: '/knowledge/:path*',
        destination: 'http://127.0.0.1:8000/knowledge/:path*',
      },
    ];
  },
};

export default nextConfig;

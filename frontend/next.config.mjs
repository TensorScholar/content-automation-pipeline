const apiProxyTarget = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";
const staticExport = process.env.TAURI_STATIC_EXPORT === "1";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  devIndicators: false,
  ...(staticExport
    ? {
        output: "export"
      }
    : {
        async rewrites() {
          return [
            {
              source: "/api/:path*",
              destination: `${apiProxyTarget}/:path*`
            }
          ];
        }
      })
};

export default nextConfig;

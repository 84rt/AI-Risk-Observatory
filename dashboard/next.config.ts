import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'internationalaisafetyreport.org',
        pathname: '/sites/default/files/**',
      },
    ],
  },
};

export default nextConfig;

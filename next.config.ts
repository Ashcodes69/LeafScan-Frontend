import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  serverExternalPackages: ["onnxruntime-node", "sharp"],
  experimental: {
    serverComponentsExternalPackages: ["onnxruntime-node", "sharp"],
  },
};

export default nextConfig;

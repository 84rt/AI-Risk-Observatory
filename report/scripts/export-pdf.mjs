#!/usr/bin/env node

import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
].filter(Boolean);

function usage() {
  console.error(
    "Usage: node scripts/export-pdf.mjs <input.html> [output.pdf]\n" +
      "Example: node scripts/export-pdf.mjs report-final-2.html report-final-2.clean.pdf",
  );
}

function resolveChromeBinary() {
  for (const candidate of CHROME_CANDIDATES) {
    if (existsSync(candidate)) return candidate;
  }

  console.error("No Chromium-based browser binary found.");
  console.error("Set CHROME_PATH to a Chrome/Chromium/Edge/Brave executable if needed.");
  process.exit(1);
}

const [, , inputArg, outputArg] = process.argv;

if (!inputArg) {
  usage();
  process.exit(1);
}

const inputPath = path.resolve(inputArg);
const outputPath = path.resolve(
  outputArg || inputArg.replace(/\.html?$/i, "") + ".pdf",
);
const outputDir = path.dirname(outputPath);

if (!existsSync(inputPath)) {
  console.error(`Input file not found: ${inputPath}`);
  process.exit(1);
}

mkdirSync(outputDir, { recursive: true });

const browserPath = resolveChromeBinary();
const inputUrl = new URL(`file://${inputPath}`).toString();
const result = spawnSync(
  browserPath,
  [
    "--headless=new",
    "--disable-gpu",
    "--allow-file-access-from-files",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=8000",
    "--no-pdf-header-footer",
    `--print-to-pdf=${outputPath}`,
    inputUrl,
  ],
  { stdio: "inherit" },
);

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}

process.exit(result.status ?? 0);

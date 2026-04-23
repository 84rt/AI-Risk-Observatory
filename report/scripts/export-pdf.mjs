#!/usr/bin/env node

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { PDFDocument, rgb } from "pdf-lib";
import puppeteer from "puppeteer-core";

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

const footerTemplate = `
  <div style="
    width: 100%;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
    color: #b91c1c;
    padding: 0 18mm 6mm;
    text-align: center;
    letter-spacing: 0.08em;
  ">
    <span style="color: #6f777b;">&#8212;</span>
    <span class="pageNumber" style="display: inline-block; min-width: 12mm; font-weight: 700;"></span>
    <span style="color: #6f777b;">&#8212;</span>
  </div>
`;

async function hideFooterOnLeadingPages(pdfPath, numPagesToHide = 2) {
  const pdfBytes = readFileSync(pdfPath);
  const pdfDoc = await PDFDocument.load(pdfBytes);
  const pages = pdfDoc.getPages().slice(0, numPagesToHide);

  for (const page of pages) {
    const { width } = page.getSize();

    // Blank the footer strip on the first pages while leaving body content intact.
    page.drawRectangle({
      x: 0,
      y: 0,
      width,
      height: 42,
      color: rgb(1, 1, 1),
      borderWidth: 0,
    });
  }

  writeFileSync(pdfPath, await pdfDoc.save());
}

const browser = await puppeteer.launch({
  executablePath: browserPath,
  headless: true,
  args: [
    "--allow-file-access-from-files",
    "--disable-gpu",
  ],
});

try {
  const page = await browser.newPage();
  await page.goto(inputUrl, { waitUntil: "networkidle0" });
  await page.pdf({
    path: outputPath,
    format: "A4",
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: "<div></div>",
    footerTemplate,
    margin: {
      top: "18mm",
      right: "18mm",
      bottom: "20mm",
      left: "18mm",
    },
    preferCSSPageSize: true,
  });
} finally {
  await browser.close();
}

await hideFooterOnLeadingPages(outputPath, 2);

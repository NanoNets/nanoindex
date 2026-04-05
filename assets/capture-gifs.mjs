/**
 * Captures animated canvases from generate-diagrams.html as GIFs.
 * Also captures final-frame PNGs for static use.
 *
 * Usage: node assets/capture-gifs.mjs
 */
import puppeteer from "puppeteer";
import GIFEncoder from "gif-encoder-2";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = `file://${path.join(__dirname, "generate-diagrams.html")}`;

const CANVASES = [
  { id: "ingestion",    name: "ingestion-pipeline",       frames: 60, delay: 50, totalMs: 8000 },
  { id: "query_fast",   name: "query-fast",               frames: 60, delay: 50, totalMs: 8000 },
  { id: "query_agentic",name: "query-agentic",            frames: 70, delay: 55, totalMs: 9000 },
  { id: "financebench", name: "financebench-architecture", frames: 1,  delay: 0,  totalMs: 500 },
];

async function captureCanvas(page, canvasId, frameCount, delayMs, totalMs) {
  const frames = [];
  const interval = totalMs / frameCount;

  for (let i = 0; i < frameCount; i++) {
    const dataUrl = await page.evaluate((id) => {
      const c = document.getElementById(id);
      return c ? c.toDataURL("image/png") : null;
    }, canvasId);

    if (!dataUrl) throw new Error(`Canvas #${canvasId} not found`);
    frames.push(Buffer.from(dataUrl.split(",")[1], "base64"));

    if (i < frameCount - 1) {
      await new Promise((r) => setTimeout(r, interval));
    }
  }
  return frames;
}

async function framesToGif(frames, width, height, delay, outPath) {
  const encoder = new GIFEncoder(width, height, "neuquant", true);
  encoder.setDelay(delay);
  encoder.setRepeat(0); // loop forever
  encoder.setQuality(10);
  encoder.start();

  const { createCanvas, loadImage } = await import("canvas").catch(() => null) || {};

  // Use puppeteer to decode PNGs since we may not have node-canvas
  for (const frame of frames) {
    // Write temp file, load as raw RGBA
    const tmpPath = path.join(__dirname, "_tmp_frame.png");
    fs.writeFileSync(tmpPath, frame);
    encoder.addFrame(frame); // gif-encoder-2 accepts Buffer
  }

  encoder.finish();
  const buf = encoder.out.getData();
  fs.writeFileSync(outPath, buf);
  console.log(`  GIF: ${outPath} (${(buf.length / 1024).toFixed(0)} KB)`);

  // Cleanup
  try { fs.unlinkSync(path.join(__dirname, "_tmp_frame.png")); } catch {}
}

async function main() {
  console.log("Launching browser...");
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.setViewport({ width: 920, height: 800 });
  await page.goto(HTML_PATH, { waitUntil: "networkidle0" });

  // Wait for animations to start
  await new Promise((r) => setTimeout(r, 500));

  for (const { id, name, frames: frameCount, delay, totalMs } of CANVASES) {
    console.log(`\nCapturing ${name}...`);

    // Get canvas dimensions
    const dims = await page.evaluate((cid) => {
      const c = document.getElementById(cid);
      return c ? { w: c.width, h: c.height } : null;
    }, id);

    if (!dims) { console.log(`  SKIP: canvas #${id} not found`); continue; }

    // Capture PNG (final frame after full animation)
    await new Promise((r) => setTimeout(r, totalMs + 500)); // wait for animation to complete + hold

    const pngData = await page.evaluate((cid) => {
      return document.getElementById(cid)?.toDataURL("image/png")?.split(",")[1];
    }, id);

    if (pngData) {
      const pngPath = path.join(__dirname, `${name}.png`);
      fs.writeFileSync(pngPath, Buffer.from(pngData, "base64"));
      console.log(`  PNG: ${pngPath} (${(Buffer.from(pngData, "base64").length / 1024).toFixed(0)} KB)`);
    }

    // For animated canvases, capture GIF
    if (frameCount > 1) {
      // Reload page to restart animations
      await page.goto(HTML_PATH, { waitUntil: "networkidle0" });
      await new Promise((r) => setTimeout(r, 200));

      console.log(`  Capturing ${frameCount} frames over ${totalMs}ms...`);
      const rawFrames = await captureCanvas(page, id, frameCount, delay, totalMs);

      // Save GIF using a simpler approach - just save individual frames
      // and use the PNG as the primary asset
      console.log(`  ${rawFrames.length} frames captured (GIF encoding requires canvas module)`);

      // Save first and last frame as well
      const lastPath = path.join(__dirname, `${name}-final.png`);
      fs.writeFileSync(lastPath, rawFrames[rawFrames.length - 1]);
    }
  }

  await browser.close();
  console.log("\nDone! PNGs saved to assets/");
}

main().catch((e) => { console.error(e); process.exit(1); });

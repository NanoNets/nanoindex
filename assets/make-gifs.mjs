/**
 * Captures animated canvases as GIFs using Puppeteer + canvas + gif-encoder-2.
 * Usage: node assets/make-gifs.mjs
 */
import puppeteer from "puppeteer";
import GIFEncoder from "gif-encoder-2";
import { createCanvas, loadImage } from "canvas";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = `file://${path.join(__dirname, "generate-diagrams.html")}`;

const TARGETS = [
  { id: "ingestion",     name: "ingestion-pipeline",  w: 880, h: 520, totalMs: 8000, fps: 12 },
  { id: "query_fast",    name: "query-fast",           w: 880, h: 480, totalMs: 8000, fps: 12 },
  { id: "query_agentic", name: "query-agentic",        w: 880, h: 530, totalMs: 9000, fps: 12 },
];

async function main() {
  console.log("Launching browser...");
  const browser = await puppeteer.launch({ headless: true });

  for (const t of TARGETS) {
    console.log(`\n${t.name}: capturing ${t.fps} fps over ${t.totalMs / 1000}s...`);

    const page = await browser.newPage();
    await page.setViewport({ width: 920, height: 800 });
    await page.goto(HTML_PATH, { waitUntil: "networkidle0" });
    await new Promise((r) => setTimeout(r, 300));

    const frameCount = Math.ceil((t.totalMs / 1000) * t.fps);
    const interval = t.totalMs / frameCount;
    const encoder = new GIFEncoder(t.w, t.h, "neuquant", true);
    encoder.setDelay(Math.round(1000 / t.fps));
    encoder.setRepeat(0);
    encoder.setQuality(10);
    encoder.start();

    const tempCanvas = createCanvas(t.w, t.h);
    const tempCtx = tempCanvas.getContext("2d");

    for (let i = 0; i < frameCount; i++) {
      const dataUrl = await page.evaluate((id) => {
        return document.getElementById(id)?.toDataURL("image/png");
      }, t.id);

      if (!dataUrl) { console.log("  canvas not found, skipping"); break; }

      const img = await loadImage(Buffer.from(dataUrl.split(",")[1], "base64"));
      tempCtx.clearRect(0, 0, t.w, t.h);
      tempCtx.drawImage(img, 0, 0);
      encoder.addFrame(tempCtx);

      if (i < frameCount - 1) await new Promise((r) => setTimeout(r, interval));
      if (i % 10 === 0) process.stdout.write(`  frame ${i + 1}/${frameCount}\r`);
    }

    encoder.finish();
    const buf = encoder.out.getData();
    const outPath = path.join(__dirname, `${t.name}.gif`);
    fs.writeFileSync(outPath, buf);
    console.log(`  saved: ${outPath} (${(buf.length / 1024).toFixed(0)} KB, ${frameCount} frames)`);

    await page.close();
  }

  await browser.close();
  console.log("\nDone!");
}

main().catch((e) => { console.error(e); process.exit(1); });

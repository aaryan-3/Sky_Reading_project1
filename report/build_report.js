// Builds the model evaluation DOCX from results/*.json, report/narrative.json and report/figures/*.png
const fs = require("fs"), path = require("path");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, ImageRun, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber, TableOfContents } = D;

const ROOT = process.env.PROJ || path.join(__dirname, "..");
const rj = f => JSON.parse(fs.readFileSync(path.join(ROOT, f), "utf8").replace(/\bNaN\b/g, "null"));
const R = rj("results/results.json");
const CV = rj("results/cv.json");
const N = JSON.parse(fs.readFileSync(path.join(ROOT, "report/narrative.json")));
const FIG = path.join(ROOT, "report/figures");
const FS = { image_only: "Image only", image_hour: "Image + hour", image_pm_hour: "Image + PM2.5 + PM10 + hour",
  none: "Baseline", hour_only: "Baseline (no image)", pm_hour_only: "Baseline (no image)", location: "Reference" };
const PAGE_W = 9026; // A4 text width with 1" margins (DXA)
const FONT = "Calibri", INK = "1F2328", HEAD = "1D3B5A", RULE = "C9CCD1", ZEBRA = "F3F5F7";

const f2 = v => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(2);
const f1 = v => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(1);
const P = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 288 }, ...opts,
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === "string" ? new TextRun(t) : t) });
const B = t => new TextRun({ text: t, bold: true });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = t => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 },
  children: (Array.isArray(t) ? t : [t]).map(x => typeof x === "string" ? new TextRun(x) : x) });
const caption = t => new Paragraph({ spacing: { before: 60, after: 220 }, children: [new TextRun({ text: t, italics: true, size: 18, color: "5F6670" })] });

function img(file, wIn) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG IHDR
  const W = wIn * 96, Hh = W * h / w;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: W, height: Hh },
      altText: { title: file, description: file, name: file } })] });
}

function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, isHead, r) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: isHead ? { fill: "E4EAF1", type: ShadingType.CLEAR, color: "auto" } : (r % 2 ? { fill: ZEBRA, type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 || (opts.leftCols || []).includes(i) ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text: String(t), bold: isHead || (opts.boldRows || []).includes(r), size: opts.size || 17 })] })] });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, -1)) }),
      ...rows.map((row, r) => new TableRow({ children: row.map((t, i) => cell(t, i, false, r)) }))] });
}

const cands = R.candidates;
const sel = R.selected;
function metricRows(target, level, split = "test") {
  return cands.filter(c => c.target === target).map(c => {
    const m = c[split][level];
    return [FS[c.fset], c.model, m.n, f1(m.MAE), f1(m.RMSE), f2(m.R2), f2(m.Pearson)];
  });
}
const MW = [2050, 2900, 560, 840, 840, 880, 956];
const MH = ["Input set", "Model", "n", "MAE", "RMSE", "R²", "Pearson r"];
const boldSel = (target) => cands.filter(c => c.target === target).map((c, i) => (sel[target][c.fset] && sel[target][c.fset].model === c.model) ? i : -1).filter(i => i >= 0);

const children = [];
// ---- title
children.push(new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: N.title, bold: true, size: 44, color: HEAD })] }));
children.push(new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: N.subtitle, size: 24, color: "5F6670" })] }));
children.push(H1("Summary"));
N.summary.forEach(t => children.push(P(t)));
children.push(table(["Target", "Input set", "Selected model", "Test MAE (group)", "Test R² (group)", "Test R² (image)", "CV R² (group, mean ± sd)"],
  N.final_rows, [800, 1900, 1900, 1050, 1050, 1050, 1276], { leftCols: [1, 2] }));
children.push(caption("Held-out test results for the selected model of each input set. Group level = mean prediction over all images of one pollution observation. CV = 3 × 5-fold grouped cross-validation over all observations."));

// ---- sections from narrative (each: heading, paragraphs, bullets, figures, tables)
function render(sec) {
  children.push(sec.level === 2 ? H2(sec.heading) : H1(sec.heading));
  (sec.blocks || []).forEach(b => {
    if (b.p) children.push(P(b.p));
    if (b.bullets) b.bullets.forEach(t => children.push(bullet(t)));
    if (b.fig) { children.push(img(b.fig, b.w || 6.2)); children.push(caption(b.caption || "")); }
    if (b.table) { children.push(table(b.table.header, b.table.rows, b.table.widths, b.table.opts || {})); if (b.caption) children.push(caption(b.caption)); }
    if (b.metrics) {
      const [target, level, split] = b.metrics;
      children.push(table(MH, metricRows(target, level, split || "test"), MW, { leftCols: [1], boldRows: boldSel(target) }));
      children.push(caption(b.caption || ""));
    }
    if (b.pagebreak) children.push(new Paragraph({ children: [new PageBreak()] }));
  });
}
N.sections.forEach(render);

const doc = new Document({
  creator: "Air pollution CV project", title: N.title,
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: HEAD, font: FONT }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: HEAD, font: FONT }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 1 } }]
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: "8A9099" })] })] }) },
    children }]
});
Packer.toBuffer(doc).then(b => { const out = path.join(ROOT, "report", "Model_Evaluation_Report.docx"); fs.writeFileSync(out, b); console.log("wrote", out); });

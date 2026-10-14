// engine.js — browser/Node mirror of bfcore.py. Keep the two in lock-step.
// Works as a Web Worker body (self.onmessage) and as a CommonJS module for testing.
"use strict";

const SIGMAS = [1.0, 2.0, 4.0, 8.0];
const NORM_SIGMA = 16.0;
const NF = 1 + 8 * SIGMAS.length;

function reflectIndex(i, n) {
  const p = 2 * n;
  i %= p; if (i < 0) i += p;
  return i >= n ? p - 1 - i : i;
}

const _kcache = new Map();
function kernel(s, order) {
  const key = s + ":" + order;
  if (_kcache.has(key)) return _kcache.get(key);
  const r = Math.floor(4.0 * s + 0.5);
  const n = 2 * r + 1;
  const g = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) { const x = i - r; g[i] = Math.exp(-0.5 * x * x / (s * s)); }
  for (let i = 0; i < n; i++) sum += g[i];
  for (let i = 0; i < n; i++) g[i] = g[i] / sum;
  if (order === 1) for (let i = 0; i < n; i++) { const x = i - r; g[i] = x / (s * s) * g[i]; }
  if (order === 2) for (let i = 0; i < n; i++) { const x = i - r; g[i] = (x * x / s ** 4 - 1.0 / (s * s)) * g[i]; }
  _kcache.set(key, g);
  return g;
}

// out[i] = sum_j k[j] * a[reflect(i + j - r)], accumulated in order j = 0..2r
function corr1d(a, H, W, k, axis) {
  const r = (k.length - 1) >> 1;
  const out = new Float64Array(H * W);
  if (axis === 0) {
    for (let j = 0; j < k.length; j++) {
      const kj = k[j];
      for (let y = 0; y < H; y++) {
        const sy = reflectIndex(y + j - r, H) * W, oy = y * W;
        for (let x = 0; x < W; x++) out[oy + x] = out[oy + x] + kj * a[sy + x];
      }
    }
  } else {
    const xi = new Int32Array(W + 2 * r);
    for (let i = 0; i < W + 2 * r; i++) xi[i] = reflectIndex(i - r, W);
    for (let j = 0; j < k.length; j++) {
      const kj = k[j];
      for (let y = 0; y < H; y++) {
        const oy = y * W;
        for (let x = 0; x < W; x++) out[oy + x] = out[oy + x] + kj * a[oy + xi[x + j]];
      }
    }
  }
  return out;
}

function filt(a, H, W, s, oy = 0, ox = 0) {
  return corr1d(corr1d(a, H, W, kernel(s, oy), 0), H, W, kernel(s, ox), 1);
}

function stainOD(rgba, H, W) {
  const N = H * W, hist = new Float64Array(256), g = new Float64Array(N);
  for (let i = 0; i < N; i++) { const v = rgba[4 * i + 1]; g[i] = v; hist[v]++; }
  // numpy linear percentile on sorted values, via the histogram
  const pos = 99.5 / 100.0 * (N - 1), lo = Math.floor(pos), hi = Math.min(lo + 1, N - 1);
  const at = (rank) => { let c = 0; for (let v = 0; v < 256; v++) { c += hist[v]; if (c > rank) return v; } return 255; };
  const vlo = at(lo), vhi = at(hi);
  const bg = Math.max(vlo + (pos - lo) * (vhi - vlo), 1.0);
  const od = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let q = g[i] / bg; if (q < 1e-3) q = 1e-3; if (q > 1.0) q = 1.0;
    od[i] = -Math.log10(q);
  }
  return od;
}

function otsu(a) {
  const nb = 256; let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < a.length; i++) { if (a[i] < lo) lo = a[i]; if (a[i] > hi) hi = a[i]; }
  if (!(hi > lo)) return lo;
  const w = (hi - lo) / nb, hist = new Float64Array(nb), c = new Float64Array(nb);
  for (let i = 0; i < a.length; i++) { let b = Math.trunc((a[i] - lo) / (hi - lo) * nb); if (b > nb - 1) b = nb - 1; hist[b]++; }
  for (let i = 0; i < nb; i++) c[i] = lo + (i + 0.5) * w;
  const w1 = new Float64Array(nb), w2 = new Float64Array(nb), m1 = new Float64Array(nb), m2 = new Float64Array(nb);
  let s = 0, sc = 0;
  for (let i = 0; i < nb; i++) { s += hist[i]; sc += hist[i] * c[i]; w1[i] = s; m1[i] = sc / Math.max(s, 1e-300); }
  s = 0; sc = 0;
  for (let i = nb - 1; i >= 0; i--) { s += hist[i]; sc += hist[i] * c[i]; w2[i] = s; m2[i] = sc / Math.max(s, 1e-300); }
  let best = -Infinity, bi = 0;
  for (let i = 0; i < nb - 1; i++) {
    const d = m1[i] - m2[i + 1], v = w1[i] * w2[i + 1] * (d * d);
    if (v > best) { best = v; bi = i; }
  }
  return c[bi];
}

// sizes of 4-connected components of value `val` in m; returns per-pixel size
function componentSizes(m, H, W, val) {
  const N = H * W, lab = new Int32Array(N), size = [0], stack = new Int32Array(N);
  let cur = 0;
  for (let st = 0; st < N; st++) {
    if (m[st] !== val || lab[st]) continue;
    cur++; lab[st] = cur; let sp = 0, cnt = 0; stack[sp++] = st;
    while (sp) {
      const p = stack[--sp]; cnt++;
      const y = (p / W) | 0, x = p - y * W;
      if (y > 0 && m[p - W] === val && !lab[p - W]) { lab[p - W] = cur; stack[sp++] = p - W; }
      if (y < H - 1 && m[p + W] === val && !lab[p + W]) { lab[p + W] = cur; stack[sp++] = p + W; }
      if (x > 0 && m[p - 1] === val && !lab[p - 1]) { lab[p - 1] = cur; stack[sp++] = p - 1; }
      if (x < W - 1 && m[p + 1] === val && !lab[p + 1]) { lab[p + 1] = cur; stack[sp++] = p + 1; }
    }
    size.push(cnt);
  }
  const out = new Int32Array(N);
  for (let i = 0; i < N; i++) out[i] = lab[i] ? size[lab[i]] : 0;
  return out;
}

function biomassMask(od, H, W, minSize = 30) {
  const sm = filt(od, H, W, 1.0);
  const thr = Math.min(Math.max(otsu(sm), 0.08), 0.25);
  const N = H * W, m = new Uint8Array(N);
  for (let i = 0; i < N; i++) m[i] = sm[i] > thr ? 1 : 0;
  let sz = componentSizes(m, H, W, 1);
  for (let i = 0; i < N; i++) if (m[i] && sz[i] < minSize) m[i] = 0;
  sz = componentSizes(m, H, W, 0);
  for (let i = 0; i < N; i++) if (!m[i] && sz[i] < minSize) m[i] = 1;
  return { mask: m, thr };
}

// pixel-major Float32Array (N * NF)
function features(od, H, W, progress) {
  const N = H * W, F = new Float32Array(N * NF);
  const put = (k, a) => { for (let i = 0; i < N; i++) F[i * NF + k] = a[i]; };
  const mu = filt(od, H, W, NORM_SIGMA);
  const d2 = new Float64Array(N);
  for (let i = 0; i < N; i++) { const d = od[i] - mu[i]; d2[i] = d * d; }
  const v = filt(d2, H, W, NORM_SIGMA);
  const x = new Float64Array(N), xx = new Float64Array(N);
  for (let i = 0; i < N; i++) { x[i] = (od[i] - mu[i]) / Math.sqrt(v[i] + 1e-3); xx[i] = x[i] * x[i]; }
  put(0, x);
  let ix = null, iy = null, k = 1;
  for (let si = 0; si < SIGMAS.length; si++) {
    const s = SIGMAS[si];
    if (progress) progress(si / SIGMAS.length);
    // share the row (axis 0) pass between filters with the same row kernel; results are identical
    const r0 = corr1d(x, H, W, kernel(s, 0), 0), r1 = corr1d(x, H, W, kernel(s, 1), 0), r2 = corr1d(x, H, W, kernel(s, 2), 0);
    const gs = corr1d(r0, H, W, kernel(s, 0), 1), gx = corr1d(r0, H, W, kernel(s, 1), 1), hxx = corr1d(r0, H, W, kernel(s, 2), 1);
    const gy = corr1d(r1, H, W, kernel(s, 0), 1), hxy = corr1d(r1, H, W, kernel(s, 1), 1);
    const hyy = corr1d(r2, H, W, kernel(s, 0), 1);
    if (ix === null) { ix = gx; iy = gy; }
    const pxx = new Float64Array(N), pyy = new Float64Array(N), pxy = new Float64Array(N);
    for (let i = 0; i < N; i++) { pxx[i] = ix[i] * ix[i]; pyy[i] = iy[i] * iy[i]; pxy[i] = ix[i] * iy[i]; }
    const axx = filt(pxx, H, W, 2 * s), ayy = filt(pyy, H, W, 2 * s), axy = filt(pxy, H, W, 2 * s);
    const gxx = filt(xx, H, W, s);
    const s2 = s * s;
    for (let i = 0; i < N; i++) {
      const tr = hxx[i] + hyy[i];
      const h = (hxx[i] - hyy[i]) * 0.5;
      const disc = Math.sqrt(h * h + hxy[i] * hxy[i]);
      const l1 = tr * 0.5 + disc, l2 = tr * 0.5 - disc;
      const da = axx[i] - ayy[i];
      const coh = Math.sqrt(da * da + 4 * axy[i] * axy[i]) / (axx[i] + ayy[i] + 1e-6);
      const t = gxx[i] - gs[i] * gs[i];
      const o = i * NF + k;
      F[o] = gs[i];
      F[o + 1] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
      F[o + 2] = tr * s2;
      F[o + 3] = l1 * s2;
      F[o + 4] = l2 * s2;
      F[o + 5] = (Math.abs(l1) - Math.abs(l2)) / (Math.abs(l1) + Math.abs(l2) + 1e-6);
      F[o + 6] = coh;
      F[o + 7] = Math.sqrt(Math.max(t, 0.0));
    }
    k += 8;
  }
  return F;
}

function predictForest(model, F, mask, N, progress) {
  // tree-major traversal keeps each tree in cache; per-pixel sums still run in tree order (= Python)
  const { feat, thr, left, right, roots } = model;
  const prob = new Float64Array(N), T = roots.length;
  const idx = []; for (let i = 0; i < N; i++) if (mask[i]) idx.push(i);
  const M = idx.length, I = Int32Array.from(idx), acc = new Float64Array(M);
  for (let t = 0; t < T; t++) {
    const root = roots[t];
    for (let j = 0; j < M; j++) {
      const base = I[j] * NF;
      let n = root, f = feat[n];
      while (f !== 255) { n = F[base + f] <= thr[n] ? left[n] : right[n]; f = feat[n]; }
      acc[j] += thr[n];
    }
    if (progress) progress((t + 1) / T);
  }
  for (let j = 0; j < M; j++) prob[I[j]] = acc[j] / T;
  return prob;
}

function diskSum(a, H, W, r) {
  const Wp = W + 2 * r, Hp = H + 2 * r;
  const cs = new Float64Array(Hp * (Wp + 1));
  for (let py = 0; py < Hp; py++) {
    const sy = reflectIndex(py - r, H) * W, o = py * (Wp + 1);
    let c = 0; cs[o] = 0;
    for (let px = 0; px < Wp; px++) { c += a[sy + reflectIndex(px - r, W)]; cs[o + px + 1] = c; }
  }
  const out = new Float64Array(H * W);
  for (let dy = -r; dy <= r; dy++) {
    const hw = Math.floor(Math.sqrt(r * r - dy * dy));
    for (let y = 0; y < H; y++) {
      const o = (y + r + dy) * (Wp + 1), oy = y * W;
      for (let x = 0; x < W; x++) out[oy + x] += cs[o + x + r + hw + 1] - cs[o + x + r - hw];
    }
  }
  return out;
}

function classify(model, meta, rgba, H, W, opts, progress) {
  const window = opts.window ?? meta.window, othr = opts.overlap_thr ?? meta.overlap_thr;
  const N = H * W;
  const P = (stage, f) => progress && progress(stage, f);
  P("Stain map", 0);
  const od = stainOD(rgba, H, W);
  P("Biofilm mask", 0);
  const { mask, thr } = biomassMask(od, H, W);
  P("Morphology features", 0);
  const F = features(od, H, W, (f) => P("Morphology features", f));
  P("Classifying pixels", 0);
  const prob = predictForest(model, F, mask, N, (f) => P("Classifying pixels", f));
  P("Local mix & overlap", 0);
  const hard = new Float64Array(N), mk = new Float64Array(N);
  for (let i = 0; i < N; i++) { mk[i] = mask[i]; hard[i] = (mask[i] && prob[i] >= 0.5) ? 1 : 0; }
  const nb = diskSum(hard, H, W, window), nm = diskSum(mk, H, W, window);
  const fb = new Float64Array(N);
  const { tpr, fpr } = meta;
  for (let i = 0; i < N; i++) {
    let f = nm[i] > 0 ? nb[i] / Math.max(nm[i], 1e-9) : 0.0;
    if (tpr - fpr > 0.2) { f = (f - fpr) / (tpr - fpr); if (f < 0) f = 0; if (f > 1) f = 1; }
    fb[i] = f;
  }
  const labels = new Uint8Array(N);
  let n = 0, c1 = 0, c2 = 0, c3 = 0, sf = 0, wsum = 0, wf = 0, conf = 0;
  for (let i = 0; i < N; i++) {
    if (!mask[i]) continue;
    n++;
    const f = fb[i];
    if (f >= 1 - othr) { labels[i] = 1; c1++; }
    else if (f <= othr) { labels[i] = 2; c2++; }
    else { labels[i] = 3; c3++; }
    sf += f; wsum += od[i]; wf += f * od[i];
    conf += Math.abs(prob[i] - 0.5) * 2;
  }
  const pct = (c) => (n ? 100 * c / n : NaN);
  const stats = {
    biomass_area_pct_of_image: 100 * n / N,
    bsub_only_pct: pct(c1), ecoli_only_pct: pct(c2), overlap_pct: pct(c3),
    bsub_total_pct: pct(sf), ecoli_total_pct: pct(n - sf),
    bsub_total_pct_stain_weighted: wsum ? 100 * wf / wsum : NaN,
    ecoli_total_pct_stain_weighted: wsum ? 100 * (wsum - wf) / wsum : NaN,
    mean_confidence: n ? conf / n : NaN,
    mask_threshold_od: thr,
  };
  return { stats, labels, fb, prob, mask };
}

// ---------------------------------------------------------------- training

function rng(seed) { // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Features at up to n randomly chosen biofilm pixels of one monoculture image.
function extract(rgba, H, W, n, seed, progress) {
  const od = stainOD(rgba, H, W);
  const { mask, thr } = biomassMask(od, H, W);
  const F = features(od, H, W, progress);
  const idx = [];
  for (let i = 0; i < H * W; i++) if (mask[i]) idx.push(i);
  const r = rng(seed);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
  const m = Math.min(n, idx.length), X = new Float32Array(m * NF);
  for (let k = 0; k < m; k++) X.set(F.subarray(idx[k] * NF, idx[k] * NF + NF), k * NF);
  return { X, n: m, maskFrac: idx.length / (H * W), thr };
}

// Quantile bin edges per feature; bin(x) = first b with x <= edge[b] (else last bin)
function makeBins(X, n, nb = 64, seed = 1) {
  const r = rng(seed), take = Math.min(n, 20000), edges = [];
  const tmp = new Float32Array(take);
  for (let f = 0; f < NF; f++) {
    for (let k = 0; k < take; k++) tmp[k] = X[Math.floor(r() * n) * NF + f];
    tmp.sort();
    const e = [];
    for (let q = 1; q < nb; q++) { const v = tmp[Math.min(take - 1, Math.floor(q * take / nb))]; if (!e.length || v > e[e.length - 1]) e.push(v); }
    edges.push(Float32Array.from(e));
  }
  const codes = new Uint8Array(NF * n);
  for (let f = 0; f < NF; f++) {
    const e = edges[f], L = e.length;
    for (let i = 0; i < n; i++) {
      const x = X[i * NF + f]; let lo = 0, hi = L;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (x <= e[mid]) hi = mid; else lo = mid + 1; }
      codes[f * n + i] = lo;
    }
  }
  return { edges, codes };
}

// Random forest (Gini, bootstrap, sqrt(features) per split, balanced class weights) on rows `use`.
function trainForest(X, y, n, use, P, progress) {
  const nTrees = P.nTrees ?? 60, maxDepth = P.maxDepth ?? 14, minLeaf = P.minLeaf ?? 10, mtry = P.mtry ?? Math.round(Math.sqrt(NF));
  const r = rng(P.seed ?? 7);
  const bins = P.bins || makeBins(X, n, 64, (P.seed ?? 7) + 1);
  const { edges, codes } = bins;
  const rows = []; let c1 = 0;
  for (let i = 0; i < n; i++) if (!use || use[i]) { rows.push(i); c1 += y[i]; }
  const R = rows.length, c0 = R - c1;
  if (!c0 || !c1) throw new Error("Training needs pixels of both species.");
  const cw = [R / (2 * c0), R / (2 * c1)];
  const feat = [], thr = [], left = [], right = [], roots = [];
  const cnt = new Uint32Array(n), idx = new Int32Array(R), wv = new Float64Array(n);
  const h0 = new Float64Array(64), h1 = new Float64Array(64), hc = new Uint32Array(64);
  for (let t = 0; t < nTrees; t++) {
    cnt.fill(0);
    for (let k = 0; k < R; k++) cnt[rows[Math.floor(r() * R)]]++;
    let m = 0;
    for (let k = 0; k < R; k++) { const i = rows[k]; if (cnt[i]) { idx[m++] = i; wv[i] = cnt[i] * cw[y[i]]; } }
    roots.push(feat.length);
    const newNode = () => { feat.push(255); thr.push(0); left.push(0); right.push(0); return feat.length - 1; };
    const stack = [[newNode(), 0, m, 0]];
    while (stack.length) {
      const [node, s, e, depth] = stack.pop();
      let w0 = 0, w1 = 0;
      for (let k = s; k < e; k++) { const i = idx[k]; if (y[i]) w1 += wv[i]; else w0 += wv[i]; }
      const W = w0 + w1;
      let best = null;
      if (depth < maxDepth && e - s >= 2 * minLeaf && w0 > 0 && w1 > 0) {
        const parentImp = W - (w0 * w0 + w1 * w1) / W;
        // choose mtry distinct features
        const fs = []; while (fs.length < mtry) { const f = Math.floor(r() * NF); if (!fs.includes(f)) fs.push(f); }
        for (const f of fs) {
          const L = edges[f].length + 1; h0.fill(0, 0, L); h1.fill(0, 0, L); hc.fill(0, 0, L);
          const base = f * n;
          for (let k = s; k < e; k++) { const i = idx[k], b = codes[base + i]; if (y[i]) h1[b] += wv[i]; else h0[b] += wv[i]; hc[b]++; }
          let l0 = 0, l1 = 0, lc = 0;
          for (let b = 0; b < L - 1; b++) {
            l0 += h0[b]; l1 += h1[b]; lc += hc[b];
            const rc = (e - s) - lc;
            if (lc < minLeaf) continue; if (rc < minLeaf) break;
            const wl = l0 + l1, r0 = w0 - l0, r1 = w1 - l1, wr = r0 + r1;
            if (wl <= 0 || wr <= 0) continue;
            const imp = (wl - (l0 * l0 + l1 * l1) / wl) + (wr - (r0 * r0 + r1 * r1) / wr);
            if (imp < parentImp - 1e-12 && (!best || imp < best.imp)) best = { imp, f, b };
          }
        }
      }
      if (!best) { thr[node] = W > 0 ? w1 / W : 0.5; continue; }
      // partition idx[s:e] by bin <= b
      const base = best.f * n; let a = s, z = e - 1;
      while (a <= z) { if (codes[base + idx[a]] <= best.b) a++; else { const t2 = idx[a]; idx[a] = idx[z]; idx[z] = t2; z--; } }
      feat[node] = best.f; thr[node] = edges[best.f][best.b];
      const L = newNode(), Rn = newNode(); left[node] = L; right[node] = Rn;
      stack.push([Rn, a, e, depth + 1], [L, s, a, depth + 1]);
    }
    if (progress) progress((t + 1) / nTrees);
  }
  return { feat: Uint8Array.from(feat), thr: Float64Array.from(thr), left: Uint32Array.from(left), right: Uint32Array.from(right), roots: Uint32Array.from(roots) };
}

function predictRows(model, X, rowsIdx) {
  const out = new Float64Array(rowsIdx.length), T = model.roots.length;
  for (let j = 0; j < rowsIdx.length; j++) {
    const base = rowsIdx[j] * NF; let acc = 0;
    for (let t = 0; t < T; t++) {
      let nd = model.roots[t], f = model.feat[nd];
      while (f !== 255) { nd = X[base + f] <= model.thr[nd] ? model.left[nd] : model.right[nd]; f = model.feat[nd]; }
      acc += model.thr[nd];
    }
    out[j] = acc / T;
  }
  return out;
}

// Structural check for a model read from shared storage: children always after parent (no cycles).
function validModel(M) {
  const N = M.feat.length;
  if (!N || M.thr.length !== N || M.left.length !== N || M.right.length !== N || !M.roots.length) return false;
  for (const r of M.roots) if (r >= N) return false;
  for (let i = 0; i < N; i++) {
    const f = M.feat[i];
    if (f === 255) { if (!(M.thr[i] >= 0 && M.thr[i] <= 1)) return false; continue; }
    if (f >= NF || !(M.left[i] > i) || !(M.right[i] > i) || M.left[i] >= N || M.right[i] >= N || !Number.isFinite(M.thr[i])) return false;
  }
  return true;
}

const api = { SIGMAS, NF, kernel, corr1d, filt, stainOD, otsu, biomassMask, features, predictForest, diskSum, classify,
  rng, extract, makeBins, trainForest, predictRows, validModel };
if (typeof module !== "undefined" && module.exports) module.exports = api;

// ---- Web Worker protocol
if (typeof self !== "undefined" && typeof module === "undefined") {
  const MODELS = new Map();
  const unpack = (buf, h) => ({
    feat: new Uint8Array(buf, h.feat[0], h.feat[1]), thr: new Float64Array(buf, h.thr[0], h.thr[1]),
    left: new Uint32Array(buf, h.left[0], h.left[1]), right: new Uint32Array(buf, h.right[0], h.right[1]),
    roots: new Uint32Array(buf, h.roots[0], h.roots[1]),
  });
  self.onmessage = (e) => {
    const m = e.data;
    const prog = (stage, f) => self.postMessage({ type: "progress", id: m.id, stage, f });
    try {
      if (m.type === "setModel") {
        const M = unpack(m.buffer, m.header);
        if (!validModel(M)) throw new Error("This model file is damaged or not a biofilm model.");
        MODELS.set(m.key, { M, meta: m.meta });
        self.postMessage({ type: "done", id: m.id });
      } else if (m.type === "run") {
        const t0 = Date.now(), mm = MODELS.get(m.key);
        if (!mm) throw new Error("Model not loaded.");
        const r = classify(mm.M, mm.meta, m.rgba, m.H, m.W, m.opts, prog);
        self.postMessage({ type: "done", id: m.id, stats: r.stats, labels: r.labels, fb: new Float32Array(r.fb), ms: Date.now() - t0 }, [r.labels.buffer]);
      } else if (m.type === "predictRows") {
        const mm = MODELS.get(m.key);
        if (!mm) throw new Error("Model not loaded.");
        const p = predictRows(mm.M, m.X, m.idx);
        self.postMessage({ type: "done", id: m.id, p }, [p.buffer]);
      } else if (m.type === "extract") {
        const r = extract(m.rgba, m.H, m.W, m.n, m.seed, (f) => prog("features", f));
        self.postMessage({ type: "done", id: m.id, X: r.X, n: r.n, maskFrac: r.maskFrac }, [r.X.buffer]);
      } else if (m.type === "train") {
        const bins = makeBins(m.X, m.n, 64, 3);
        const F = trainForest(m.X, m.y, m.n, m.use || null, { ...m.params, bins }, (f) => prog("train", f));
        let test = null;
        if (m.testIdx) test = predictRows(F, m.X, m.testIdx);
        self.postMessage({ type: "done", id: m.id, forest: F, test }, [F.feat.buffer, F.thr.buffer, F.left.buffer, F.right.buffer, F.roots.buffer]);
      }
    } catch (err) {
      self.postMessage({ type: "error", id: m.id, message: String(err && err.message || err) });
    }
  };
}

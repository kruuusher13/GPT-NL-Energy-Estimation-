// Renders one full-frame PNG per shot: paper layout with a transparent window (the figure panel).
const { chromium } = require('playwright'); const katex = require('katex'); const fs = require('fs'); const path = require('path');
const EXE = process.env.CHROME || undefined;
const kcss = path.dirname(require.resolve('katex/dist/katex.min.css'));
const fsrc = p => 'file://' + path.dirname(require.resolve('@fontsource/' + p + '/package.json')) + '/files/';
const K = t => katex.renderToString(t, { displayMode: true });
const PX = 96, PY = 118, PW = 1152, PH = 720;           // figure window
const HEAD = 'R. Malik &nbsp;·&nbsp; Estimating the cost of computation, from silicon to HPC';
const sec = {
  s1: { h: '§1 &nbsp;On-board inference &nbsp;·&nbsp; ISRO Space Applications Centre, 2023',
        eq: String.raw`S=\frac{1}{(1-p)+\frac{p}{s}}\;\le\;\frac{1}{1-p}`,
        set: 'Object detection on the FPGA fabric of a Xilinx Zynq MPSoC, instead of downlinking every frame.',
        res: ['&lt; 15 ms end-to-end at a sustained 60+ FPS', '4× over CPU at ≈3× performance per watt',
              'Binding constraint: the data path outside the offloaded region, not the kernel', 'Feasibility study adopted by SAC for future edge-computing architectures'] },
  s2: { h: '§2 &nbsp;Perception without GPS &nbsp;·&nbsp; CSIR National Institute of Oceanography, 2023–24',
        eq: String.raw`\mathbf{F}(q)=-\nabla U_{\mathrm{att}}(q)-\nabla U_{\mathrm{rep}}(q)`,
        set: 'Perception and avoidance for an underwater vehicle, so dam and harbour inspection can run without divers.',
        res: ['&gt; 90% detection in turbid water', '≈ 40% lower planning latency over 50 scenarios',
              'Monocular depth over a stereo rig: a second pressure housing adds a failure mode', 'IEEE APSCON 2024 [2]'] },
  s3: { h: '§3 &nbsp;Move selection as classification &nbsp;·&nbsp; Utrecht University, 2026',
        eq: String.raw`\hat m=\arg\max_i\,(z_i+\mu_i),\quad \mu_i=\begin{cases}0 & i\in\mathcal{L}\\ -\infty & \text{otherwise}\end{cases}`,
        set: 'A closed vocabulary of 1,968 UCI moves, scored in one forward pass; legality applied as a mask before the argmax.',
        res: ['38.9M parameters, trained on 20M positions', 'Illegal moves arithmetically unreachable',
              'First place, INFOMTALC 2026 tournament, against 7B-parameter baselines'] },
  s4: { h: '§4 &nbsp;Systems in production &nbsp;·&nbsp; Torxflow and ConcertLab, 2026',
        eq: String.raw`T_{\mathrm{call}}=\max_i T_i\;\ll\;\textstyle\sum_i T_i`,
        set: 'Founding engineer, Torxflow (UtrechtInc validation programme); data scientist, ConcertLab.',
        res: ['Telephone intake runs plate lookup, parts search and pricing concurrently: 390 ms median response',
              'Intent confidence below 0.94 is routed to a human', 'Four-year P&amp;L and scenario model with the CEO, behind a live activation round'] },
  s5: { h: '§5 &nbsp;Energy of LLM data curation &nbsp;·&nbsp; TNO, GPT-NL, 2026',
        eq: String.raw`E_s(n)=c_{0,s}+c_{1,s}\,n,\qquad n^{*}_s=\frac{c_{0,s}}{c_{1,s}}`,
        eq2: String.raw`\nu^{\top}S^{-1}\nu<\gamma\;\Rightarrow\;\hat x\leftarrow\hat x+K\nu`,
        set: 'Budget a curation run’s energy, cost and CO₂ on Snellius before committing compute (MSc thesis) [1].',
        res: ['Held-out 400k-document run predicted to 2.8% (dominant stage) from calibrations ≤ 100k',
              'Gated EKF on live EAR telemetry: +93% → +8% under co-tenant contamination', 'Physics-based law beat five learned estimators'] },
};
const shots = [
  ['h1', 's1', 'Fig. 1<sup>†</sup>', 'Earth-observation satellite in low orbit (illustrative).'],
  ['f2', 's1', 'Fig. 2', 'Alternating vs overlapped DMA/convolution schedule on the Zynq fabric (interactive figure, [3]). Overlap shortens the frame without changing one arithmetic operation.'],
  ['h3', 's2', 'Fig. 3<sup>†</sup>', 'ROV inspecting a stilling basin in turbid water (illustrative).'],
  ['f4', 's2', 'Fig. 4', 'Simulated environment built for the vehicle: tethered ROV in ROS/Gazebo (original footage).'],
  ['f5', 's2', 'Fig. 5', 'Artificial potential field: raising the repulsive gain k<sub>rep</sub> bends the path around obstacles (interactive figure, [3]).'],
  ['f6', 's3', 'Fig. 6', 'Chess-Bot-20M stepped from position to legality mask; the selected move is the highest-scoring legal one (interactive figure, [3]).'],
  ['f7', 's4', 'Fig. 7', 'The record drawn as a forward pass; highlighted are the routes into Torxflow, then ConcertLab (from [3]).'],
  ['h8', 's5', 'Fig. 8<sup>†</sup>', 'Supercomputer hall (illustrative). All measurements are from Snellius, via EAR energy telemetry.'],
  ['f9', 's5', 'Fig. 9', 'Calibrating below the overhead floor n* leaves the slope unidentifiable, and the extrapolation fan opens (interactive figure, [3]).'],
  ['f10', 's5', 'Fig. 10', 'The released command-line tool: a forecast before the job is submitted [1].'],
  ['f11', 's5', 'Fig. 11', 'Live monitor on a recorded 400k run: the 95% interval collapses onto the measured energy [1].'],
];
const css = `
@font-face{font-family:Spectral;font-weight:400;src:url(${fsrc('spectral')}spectral-latin-400-normal.woff2)}
@font-face{font-family:Spectral;font-weight:400;font-style:italic;src:url(${fsrc('spectral')}spectral-latin-400-italic.woff2)}
@font-face{font-family:Spectral;font-weight:600;src:url(${fsrc('spectral')}spectral-latin-600-normal.woff2)}
@font-face{font-family:Archivo;font-weight:400;src:url(${fsrc('archivo')}archivo-latin-400-normal.woff2)}
@font-face{font-family:Archivo;font-weight:500;src:url(${fsrc('archivo')}archivo-latin-500-normal.woff2)}
@font-face{font-family:Archivo;font-weight:600;src:url(${fsrc('archivo')}archivo-latin-600-normal.woff2)}
html,body{margin:0;width:1920px;height:1080px;background:transparent;overflow:hidden}
body{position:relative;font-family:Spectral,Georgia,serif;color:#000}
.p{position:absolute;background:#f6f7f8}
.head{position:absolute;left:96px;right:96px;top:40px;display:flex;justify-content:space-between;font-family:Archivo;font-size:19px;color:#68727a;border-bottom:1px solid #dcdfe2;padding-bottom:14px}
.head b{font-weight:500;color:#3e464d}
.frame{position:absolute;left:${PX - 1}px;top:${PY - 1}px;width:${PW}px;height:${PH}px;border:1px solid #c2c8cc}
.cap{position:absolute;left:${PX}px;top:${PY + PH + 22}px;width:${PW}px;font-size:25px;line-height:1.38;color:#3e464d}
.cap b{color:#000;font-weight:600}
.col{position:absolute;left:1310px;top:${PY}px;width:520px}
.lab{font-family:Archivo;font-size:15px;letter-spacing:1.6px;text-transform:uppercase;color:#68727a;margin:0 0 10px}
.set{font-size:24px;line-height:1.4;margin:0 0 26px;color:#000}
.eq{background:#fff;border:1px solid #dcdfe2;padding:14px 16px;margin:0 0 26px;font-size:27px;overflow:hidden}
.eq .katex-display{margin:0;text-align:left;width:max-content}.eq .katex-display>.katex{text-align:left}
ul{margin:0;padding:0 0 0 0;list-style:none}
li{font-size:22.5px;line-height:1.38;margin:0 0 12px;padding-left:22px;position:relative;color:#000}
li:before{content:'';position:absolute;left:2px;top:12px;width:8px;height:8px;background:#2a78d6}
`;
function page(sh) {
  const [id, s, fig, cap] = sh, S = sec[s];
  const eqs = `<div class="eq">${K(S.eq)}</div>` + (S.eq2 && (id === 'f10' || id === 'f11') ? `<div class="eq">${K(S.eq2)}</div>` : '');
  return `<html><head><meta charset="utf-8"><link rel="stylesheet" href="file://${kcss}/katex.min.css"><style>${css}</style></head><body>
  <div class="p" style="left:0;top:0;width:1920px;height:${PY}px"></div>
  <div class="p" style="left:0;top:${PY + PH}px;width:1920px;height:${1080 - PY - PH}px"></div>
  <div class="p" style="left:0;top:${PY}px;width:${PX}px;height:${PH}px"></div>
  <div class="p" style="left:${PX + PW}px;top:${PY}px;width:${1920 - PX - PW}px;height:${PH}px"></div>
  <div class="head"><span>${HEAD}</span><b>${S.h}</b></div>
  <div class="frame"></div>
  <div class="cap"><b>${fig}.</b> ${cap}</div>
  <div class="col"><div class="lab">Setting</div><div class="set">${S.set}</div>${eqs}<div class="lab">Results</div><ul>${S.res.map(r => `<li>${r}</li>`).join('')}</ul></div>
  <script>document.fonts.ready.then(()=>document.querySelectorAll('.eq').forEach(e=>{const k=e.querySelector('.katex-display');const s=Math.min(1,(e.clientWidth-52)/e.querySelector('.katex').getBoundingClientRect().width);k.style.transformOrigin='left top';if(s<1){const h=k.offsetHeight;k.style.transform='scale('+s+')';k.style.marginBottom=(-(1-s)*h)+'px'}}));</script>
  </body></html>`;
}
const title = `<html><head><meta charset="utf-8"><link rel="stylesheet" href="file://${kcss}/katex.min.css"><style>${css} body{background:#f6f7f8}
.t{position:absolute;left:180px;top:300px;width:1560px}.t h1{font-family:Spectral;font-weight:400;font-size:66px;line-height:1.12;margin:0 0 34px}
.t .a{font-family:Archivo;font-size:27px;color:#000;margin:0 0 10px}.t .f{font-family:Archivo;font-size:22px;color:#68727a;margin:0 0 46px}
.t .ab{font-size:26px;line-height:1.45;color:#3e464d;max-width:1380px;border-top:1px solid #dcdfe2;padding-top:26px}</style></head><body>
<div class="t"><h1>Estimating the cost of computation,<br>from silicon to HPC</h1><div class="a">Romir Malik</div>
<div class="f">MSc Applied Data Science, Utrecht University &nbsp;·&nbsp; MSc thesis at TNO (GPT-NL) &nbsp;·&nbsp; Selected work, 2023–2026</div>
<div class="ab">Five systems, five scales: 10<sup>−3</sup> s on one FPGA die, 10<sup>0</sup> m of underwater visibility, 10<sup>7</sup> model parameters, 390 ms per phone call, 10<sup>6</sup> J per curation run on a national supercomputer. The same question each time: what will it cost, and can we know before we spend it?</div></div></body></html>`;
const refs = `<html><head><meta charset="utf-8"><style>${css} body{background:#f6f7f8}
.r{position:absolute;left:180px;top:200px;width:1560px}.r h2{font-family:Archivo;font-weight:500;font-size:20px;letter-spacing:1.6px;text-transform:uppercase;color:#68727a;margin:0 0 26px}
.r p{font-size:27px;line-height:1.45;margin:0 0 16px;padding-left:52px;text-indent:-52px}.r .n{font-size:22px;color:#68727a;margin-top:44px;border-top:1px solid #dcdfe2;padding-top:22px;padding-left:0;text-indent:0}
.r .me{font-family:Archivo;font-size:24px;color:#000;margin-top:40px;padding-left:0;text-indent:0}</style></head><body><div class="r"><h2>References</h2>
<p>[1] R. Malik. <i>Complexity-aware energy estimation of the data curation pipeline for GPT-NL on HPC clusters.</i> MSc thesis, TNO and Utrecht University, 2026. github.com/kruuusher13/gptnl-energy-estimation-ekf</p>
<p>[2] R. Malik et al. <i>Rope-augmented path following and control of a remotely operated underwater vehicle using vision for stilling basin surveillance.</i> IEEE APSCON, 2024.</p>
<p>[3] R. Malik. Portfolio with interactive method figures. kruuusher13.github.io/Portfolio</p>
<p class="n">† AI-generated illustrations (Figs. 1, 3, 8). All other footage, figures and numbers come from the cited work.</p>
<p class="me">Romir Malik &nbsp;·&nbsp; romir.ds10@gmail.com &nbsp;·&nbsp; TNO Traineeship, February 2027</p></div></body></html>`;
(async () => {
  const b = await chromium.launch(EXE ? { executablePath: EXE } : {});
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  fs.mkdirSync('lay', { recursive: true });
  const shoot = async (html, f, transparent) => { fs.writeFileSync('lay_tmp.html', html); await p.goto('file://' + path.resolve('lay_tmp.html')); await p.evaluate(() => document.fonts.ready); await p.waitForTimeout(600); await p.screenshot({ path: f, omitBackground: transparent }); };
  for (const sh of shots) await shoot(page(sh), `lay/${sh[0]}.png`, true);
  await shoot(title, 'lay/title.png', false); await shoot(refs, 'lay/refs.png', false);
  await b.close(); console.log('layers done');
})();

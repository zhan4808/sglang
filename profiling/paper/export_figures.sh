#!/bin/bash
# Export all pgfplots figures from main.tex as standalone PNGs
set -e
cd "$(dirname "$0")"
mkdir -p figures

# Shared preamble for all standalone figures
PREAMBLE='
\documentclass[border=2pt]{standalone}
\usepackage{times}
\usepackage{amsmath,amssymb}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{patterns}
\definecolor{fiblue}{HTML}{2171B5}
\definecolor{triorange}{HTML}{E6550D}
\definecolor{reconred}{HTML}{CB181D}
\definecolor{attnblue}{HTML}{6BAED6}
\definecolor{predgreen}{HTML}{31A354}
\definecolor{measred}{HTML}{DE2D26}
\pgfplotsset{
  every axis/.append style={
    label style={font=\small},
    tick label style={font=\footnotesize},
    legend style={font=\footnotesize, legend cell align=left},
  },
}
\newlength{\columnwidth}
\setlength{\columnwidth}{3.4in}
\begin{document}
'

POSTAMBLE='
\end{document}
'

compile_fig() {
    local name="$1"
    local body="$2"
    echo "$PREAMBLE" > "figures/${name}.tex"
    echo "$body" >> "figures/${name}.tex"
    echo "$POSTAMBLE" >> "figures/${name}.tex"
    echo "Compiling ${name}..."
    (cd figures && pdflatex -interaction=nonstopmode "${name}.tex" > /dev/null 2>&1) || true
    pdftoppm -png -r 300 -singlefile "figures/${name}.pdf" "figures/${name}"
    echo "  -> figures/${name}.png"
}

# ── Figure 1: Decode Bandwidth ──
compile_fig "decode_bw" '
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth,
  height=0.58\columnwidth,
  xlabel={Batch Size},
  ylabel={HBM Bandwidth (GB/s)},
  xmode=log, log basis x=2,
  xtick={1,4,16,64,128,256},
  xticklabels={1,4,16,64,128,256},
  ymin=0, ymax=3700,
  ytick={0,500,1000,1500,2000,2500,3000,3500},
  grid=major, grid style={gray!30},
  every axis plot/.append style={thick},
  ylabel near ticks,
  clip=false,
  legend style={at={(0.97,0.03)}, anchor=south east},
]
\addplot[dashed, black!40, thin, forget plot]
  coordinates {(0.7,3350) (350,3350)};
\node[font=\scriptsize, black!50, anchor=south west]
  at (axis cs:1,3370) {H100 peak (3{,}350 GB/s)};
\addplot[dotted, fiblue!40, thin, forget plot]
  coordinates {(0.7,2987) (350,2987)};
\node[font=\scriptsize, fiblue!50, anchor=north east]
  at (axis cs:256,2970) {89\%};
\addplot[dotted, triorange!40, thin, forget plot]
  coordinates {(0.7,2669) (350,2669)};
\node[font=\scriptsize, triorange!50, anchor=north east]
  at (axis cs:256,2650) {80\%};
\addplot[fiblue, mark=*, mark size=2] coordinates {
  (1,298) (4,1056) (16,2317) (64,2870) (128,2957) (256,2987)
};
\addplot[triorange, mark=square*, mark size=2] coordinates {
  (1,147) (4,582) (16,1910) (64,2470) (128,2585) (256,2669)
};
\node[font=\tiny, black!70, anchor=north east] at (axis cs:1.6,270) {2.0$\times$};
\node[font=\tiny, black!70, anchor=north west] at (axis cs:260,2987) {1.12$\times$};
\legend{FlashInfer, Triton}
\end{axis}
\end{tikzpicture}
'

# ── Figure 2: Prefill TFLOPS ──
compile_fig "prefill_tflops" '
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth,
  height=0.62\columnwidth,
  ybar,
  bar width=5pt,
  xlabel={Configuration (batch size $\times$ sequence length)},
  ylabel={TFLOPS},
  symbolic x coords={1x512,1x1K,1x2K,1x4K,4x2K,4x4K,16x2K,16x4K},
  xtick=data,
  x tick label style={rotate=45, anchor=east, font=\scriptsize},
  ymin=0, ymax=1100,
  ytick={0,200,400,600,800,1000},
  grid=major, grid style={gray!30},
  enlarge x limits=0.08,
  clip=false,
]
\addplot[fill=fiblue, draw=fiblue!80] coordinates {
  (1x512,88) (1x1K,264) (1x2K,387) (1x4K,485)
  (4x2K,448) (4x4K,552) (16x2K,444) (16x4K,512)
};
\addplot[fill=triorange, draw=triorange!80] coordinates {
  (1x512,54) (1x1K,126) (1x2K,161) (1x4K,188)
  (4x2K,183) (4x4K,204) (16x2K,195) (16x4K,209)
};
\addplot[dashed, black!40, thin, forget plot, sharp plot]
  coordinates {(1x512,990) (16x4K,990)};
\node[font=\scriptsize, black!50, anchor=north west]
  at (axis cs:1x512,980) {H100 peak (990 TF)};
\node[font=\tiny, fiblue!80, anchor=south]
  at (axis cs:4x4K,558) {\textbf{552}};
\legend{FlashInfer, Triton}
\end{axis}
\end{tikzpicture}
'

# ── Figure 3: MLA Reconstruction Overhead ──
compile_fig "recon_overhead" '
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth,
  height=0.65\columnwidth,
  ybar stacked,
  bar width=11pt,
  xlabel={Batch Size},
  ylabel={Per-Layer Time ($\mu$s)},
  symbolic x coords={1,4,16,64,128,256,512},
  xtick=data,
  ymin=0, ymax=780,
  grid=major, grid style={gray!30},
  enlarge x limits=0.1,
  clip=false,
  legend style={at={(0.03,0.97)}, anchor=north west},
]
\addplot[fill=reconred!75, draw=reconred!90] coordinates {
  (1,35.6) (4,35.4) (16,34.2) (64,34.1) (128,38.4) (256,66.9) (512,109.4)
};
\addplot[fill=attnblue!75, draw=attnblue!90] coordinates {
  (1,23.0) (4,27.2) (16,40.4) (64,83.6) (128,154.4) (256,293.6) (512,584.5)
};
\node[font=\scriptsize\bfseries, reconred!80, anchor=south]
  at (axis cs:1,62) {61\%};
\node[font=\scriptsize\bfseries, reconred!80, anchor=south]
  at (axis cs:4,66) {57\%};
\node[font=\scriptsize\bfseries, reconred!80, anchor=south]
  at (axis cs:16,78) {46\%};
\node[font=\scriptsize, reconred!80, anchor=south]
  at (axis cs:64,121) {29\%};
\node[font=\scriptsize, reconred!80, anchor=south]
  at (axis cs:128,196) {20\%};
\node[font=\scriptsize, reconred!80, anchor=south]
  at (axis cs:256,364) {19\%};
\node[font=\scriptsize, reconred!80, anchor=south]
  at (axis cs:512,698) {16\%};
\legend{Reconstruction BMMs, Attention Kernel}
\end{axis}
\end{tikzpicture}
'

# ── Figure 4: Roofline ──
compile_fig "roofline" '
\begin{tikzpicture}
\begin{loglogaxis}[
  width=0.95\columnwidth,
  height=0.6\columnwidth,
  xlabel={Arithmetic Intensity (FLOP/byte)},
  ylabel={Performance (TFLOPS)},
  xmin=0.4, xmax=600,
  ymin=0.4, ymax=1500,
  grid=major, grid style={gray!20},
  clip=false,
  legend style={at={(0.97,0.03)}, anchor=south east},
]
\addplot[thick, black!60, forget plot, domain=0.4:295, samples=2] {3.35*x};
\addplot[thick, black!60, forget plot, domain=295:600, samples=2] {989.4};
\addplot[only marks, mark=|, mark size=4, black!60, forget plot]
  coordinates {(295, 989.4)};
\node[font=\scriptsize, black!60, rotate=33, anchor=south]
  at (axis cs:8,35) {HBM (3.35 TB/s)};
\node[font=\scriptsize, black!60, anchor=south east]
  at (axis cs:580,989.4) {990 TF};
\node[font=\tiny, black!50, anchor=north]
  at (axis cs:295,850) {AI=295};
\addplot[only marks, mark=*, mark size=3.5, reconred, thick]
  coordinates {
    (0.97, 0.92) (15.5, 17.6) (62.1, 99.4) (93.0, 170.8)
  };
\addlegendentry{Recon.\ BMMs}
\node[font=\scriptsize, reconred, anchor=south west] at (axis cs:1.2,1.1) {bs=1};
\node[font=\scriptsize, reconred, anchor=south west] at (axis cs:17.5,19.5) {bs=16};
\node[font=\scriptsize, reconred, anchor=north east] at (axis cs:56,88) {bs=64};
\node[font=\scriptsize, reconred, anchor=south west] at (axis cs:100,182) {bs=128};
\end{loglogaxis}
\end{tikzpicture}
'

# ── Figure 5: Perplexity ──
compile_fig "ppl" '
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth,
  height=0.58\columnwidth,
  ybar,
  bar width=28pt,
  ylabel={Perplexity (PPL $\downarrow$)},
  symbolic x coords={{FP16 baseline},{INT4 selective},{INT4 all weights}},
  xtick=data,
  x tick label style={font=\small, align=center, text width=2.4cm},
  ymin=0, ymax=15,
  ytick={0,2,4,6,8,10,12,14},
  grid=major, grid style={gray!30},
  enlarge x limits=0.25,
  clip=false,
]
\addplot[fill=fiblue!50, draw=fiblue!70] coordinates {
  ({FP16 baseline}, 5.727)
  ({INT4 selective}, 5.777)
  ({INT4 all weights}, 11.784)
};
\addplot[dashed, reconred!50, thick, forget plot, sharp plot]
  coordinates {({FP16 baseline},6.227) ({INT4 all weights},6.227)};
\node[font=\small, reconred!60, anchor=south west]
  at (axis cs:{FP16 baseline},6.4) {+0.5 PPL threshold};
\node[font=\small\bfseries, white, anchor=north]
  at (axis cs:{FP16 baseline},5.5) {5.73};
\node[font=\small\bfseries, white, anchor=north]
  at (axis cs:{INT4 selective},5.55) {5.78};
\node[font=\small\bfseries, white, anchor=north]
  at (axis cs:{INT4 all weights},11.55) {11.78};
\node[font=\small, predgreen!80!black, anchor=north]
  at (axis cs:{INT4 selective},4.6) {$\Delta$\,=\,+0.051};
\node[font=\small, reconred!80, anchor=south]
  at (axis cs:{INT4 all weights},12.1) {$\Delta$\,=\,+6.057};
\end{axis}
\end{tikzpicture}
'

# ── Figure 6: L2 Cache Barrier ──
compile_fig "l2barrier" '
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth,
  height=0.62\columnwidth,
  ybar,
  bar width=8pt,
  xlabel={Batch Size},
  ylabel={INT4 / FP16 Speedup},
  symbolic x coords={1,4,16,64,128,256},
  xtick=data,
  ymin=0, ymax=4.8,
  ytick={0,0.5,1,1.5,2,2.5,3,3.5,4,4.5},
  grid=major, grid style={gray!30},
  enlarge x limits=0.12,
  clip=false,
  extra y ticks={1.0},
  extra y tick labels={},
  extra y tick style={grid style={black!50, thick, dashed}},
]
\addplot[fill=predgreen!60, draw=predgreen!80] coordinates {
  (1,3.94) (4,3.89) (16,3.67) (64,3.00) (128,2.50) (256,2.00)
};
\addplot[fill=measred!60, draw=measred!80] coordinates {
  (1,0.49) (4,0.50) (16,0.44) (64,0.28) (128,0.21) (256,0.23)
};
\node[font=\scriptsize, black!60, anchor=south west]
  at (axis cs:1,1.04) {1$\times$ parity};
\draw[<->, black!70, thick]
  (axis cs:1,0.58) -- (axis cs:1,3.85)
  node[midway, right, font=\scriptsize, black!70] {8$\times$};
\legend{Roofline predicted, Measured INT4/FP16}
\end{axis}
\end{tikzpicture}
'

# ── Figure 7: E2E Decomposition ──
compile_fig "e2e" '
\begin{tikzpicture}
\begin{axis}[
  width=0.92\columnwidth,
  height=0.52\columnwidth,
  xbar,
  bar width=10pt,
  xlabel={Share of Decode Step (\%)},
  xmin=0, xmax=82,
  symbolic y coords={Other,RoPE,Activation,Norms,{Attention},{Linear}},
  ytick=data,
  y tick label style={font=\small},
  nodes near coords={\pgfmathprintnumber\pgfplotspointmeta\%},
  nodes near coords style={font=\scriptsize, anchor=west},
  grid=major, grid style={gray!20},
  enlarge y limits=0.22,
]
\addplot[fill=fiblue!50, draw=fiblue!70] coordinates {
  (70.6,{Linear})
  (19.6,{Attention})
  (2.0,Norms)
  (1.4,Activation)
  (0.8,RoPE)
  (5.5,Other)
};
\end{axis}
\end{tikzpicture}
'

# ── Figure 8: L2 Sweep ──
compile_fig "l2sweep" '
\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth,
  height=0.62\columnwidth,
  xlabel={FP16 Weight Size (MB)},
  ylabel={INT4 / FP16 Time Ratio},
  xmin=0, xmax=135,
  ymin=0.9, ymax=2.1,
  ytick={1.0,1.2,1.4,1.6,1.8,2.0},
  grid=major, grid style={gray!30},
  clip=false,
  legend style={at={(0.97,0.97)}, anchor=north east},
  extra y ticks={1.0},
  extra y tick labels={},
  extra y tick style={grid style={black!50, thick, dashed}},
]
% L2 capacity region
\fill[predgreen!8] (axis cs:0,0.9) rectangle (axis cs:50,2.1);
\node[font=\scriptsize, predgreen!60, anchor=north west] at (axis cs:1,2.05) {L2-resident};
\draw[predgreen!40, thick, dashed] (axis cs:50,0.9) -- (axis cs:50,2.1);
\node[font=\scriptsize, black!50, anchor=south, rotate=90] at (axis cs:51,1.65) {50\,MB L2};
% bs=1
\addplot[reconred, mark=*, mark size=2.5, thick] coordinates {
  (8,1.912) (12,1.887) (16,1.864) (24,1.908) (32,1.956)
  (40,1.460) (48,1.322) (56,1.282) (64,1.210) (80,1.152) (96,1.124) (128,1.078)
};
% bs=4
\addplot[fiblue, mark=square*, mark size=2, thick] coordinates {
  (8,1.915) (12,1.847) (16,1.858) (24,1.847) (32,1.943)
  (40,1.433) (48,1.285) (56,1.251) (64,1.200) (80,1.159) (96,1.137) (128,1.082)
};
% 1x parity label
\node[font=\scriptsize, black!50, anchor=south west] at (axis cs:100,1.02) {1$\times$ parity};
% MLA operating point
\draw[->, black!60, thick] (axis cs:16,1.55) -- (axis cs:16,1.88);
\node[font=\scriptsize, black!60, anchor=south] at (axis cs:16,1.50) {MLA};
\legend{bs=1, bs=4}
\end{axis}
\end{tikzpicture}
'

echo ""
echo "=== All figures exported ==="
ls -la figures/*.png

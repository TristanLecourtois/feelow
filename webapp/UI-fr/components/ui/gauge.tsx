"use client"

import * as React from "react"

/**
 * SVG semicircular gauge from -1 (Strong Sell / red) to +1 (Strong Buy / green).
 *
 * Props:
 *  - value: number in [-1, 1]
 *  - size: pixel width/height (default 160)
 */
interface GaugeProps {
  value: number
  size?: number
  className?: string
}

function valueToColor(v: number): string {
  // v in [-1, 1] → red (#ef4444) → orange (#f59e0b) → green (#22c55e)
  const t = (v + 1) / 2 // normalize to [0, 1]
  if (t <= 0.5) {
    // red → orange
    const r = Math.round(239 + (245 - 239) * (t / 0.5))
    const g = Math.round(68 + (158 - 68) * (t / 0.5))
    const b = Math.round(68 + (11 - 68) * (t / 0.5))
    return `rgb(${r},${g},${b})`
  }
  // orange → green
  const r = Math.round(245 + (34 - 245) * ((t - 0.5) / 0.5))
  const g = Math.round(158 + (197 - 158) * ((t - 0.5) / 0.5))
  const b = Math.round(11 + (94 - 11) * ((t - 0.5) / 0.5))
  return `rgb(${r},${g},${b})`
}

function valueToLabel(v: number): string {
  if (v >= 0.5) return "Strong Buy"
  if (v >= 0.15) return "Buy"
  if (v > -0.15) return "Hold"
  if (v > -0.5) return "Sell"
  return "Strong Sell"
}

export function Gauge({ value, size = 160, className }: GaugeProps) {
  const clamped = Math.max(-1, Math.min(1, value))
  const color = valueToColor(clamped)
  const label = valueToLabel(clamped)

  // Arc geometry
  const cx = size / 2
  const cy = size * 0.6
  const r = size * 0.4
  const strokeWidth = size * 0.08

  // Needle angle (used only to compute filled arc extent)
  const needleAngle = Math.PI * (1 - (clamped + 1) / 2)

  // Arc path (background)
  const arcStartX = cx - r
  const arcStartY = cy
  const arcEndX = cx + r
  const arcEndY = cy

  // Colored arc from left up to needle position
  const filledAngle = Math.PI - needleAngle // how far from left
  const filledEndX = cx + r * Math.cos(Math.PI - filledAngle)
  const filledEndY = cy - r * Math.sin(Math.PI - filledAngle)
  const largeArc = filledAngle > Math.PI / 2 ? 1 : 0

  // Gradient stops for the background arc
  const score = ((clamped + 1) / 2) * 100

  return (
    <div className={className} style={{ width: size, height: size * 0.72, position: "relative" }}>
      <svg width={size} height={size * 0.72} viewBox={`0 0 ${size} ${size * 0.72}`}>
        {/* Gradient for bg arc */}
        <defs>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ef4444" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#22c55e" />
          </linearGradient>
        </defs>

        {/* Background arc (grey) */}
        <path
          d={`M ${arcStartX} ${arcStartY} A ${r} ${r} 0 0 1 ${arcEndX} ${arcEndY}`}
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Colored arc (gradient) */}
        <path
          d={`M ${arcStartX} ${arcStartY} A ${r} ${r} 0 0 1 ${arcEndX} ${arcEndY}`}
          fill="none"
          stroke="url(#gaugeGrad)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          opacity={0.25}
        />

        {/* Filled arc up to needle (solid color) */}
        {filledAngle > 0.01 && (
          <path
            d={`M ${arcStartX} ${arcStartY} A ${r} ${r} 0 ${largeArc} 1 ${filledEndX} ${filledEndY}`}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        )}

        {/* Labels: Sell / Hold / Buy */}
        <text x={cx - r - 2} y={cy + size * 0.1} textAnchor="start" fontSize={size * 0.065} fill="hsl(var(--muted-foreground))">
          Sell
        </text>
        <text x={cx} y={cy - r - size * 0.04} textAnchor="middle" fontSize={size * 0.06} fill="hsl(var(--muted-foreground))">
          Hold
        </text>
        <text x={cx + r + 2} y={cy + size * 0.1} textAnchor="end" fontSize={size * 0.065} fill="hsl(var(--muted-foreground))">
          Buy
        </text>
      </svg>

      {/* Score + label overlay */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          textAlign: "center",
          lineHeight: 1.1,
        }}
      >
        <div style={{ fontSize: size * 0.17, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>
          {clamped >= 0 ? "+" : ""}{clamped.toFixed(2)}
        </div>
        <div style={{ fontSize: size * 0.09, fontWeight: 600, color, marginTop: 1 }}>
          {label}
        </div>
      </div>
    </div>
  )
}

export { valueToColor, valueToLabel }

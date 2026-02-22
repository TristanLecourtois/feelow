"use client"

import * as React from "react"
import { IconTrendingDown, IconTrendingUp } from "@tabler/icons-react"

import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Spinner } from '@/components/ui/spinner'
import { useTicker } from '@/lib/ticker-context'

interface KpiData {
  ticker: string
  price: number
  pct_change: number
  news_volume: number
  avg_sentiment: number
  signal: string
  rsi: number
}

export function SectionCards() {
  const { ticker } = useTicker()
  const [data, setData] = React.useState<KpiData | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    setLoading(true)
    fetch(`http://localhost:8000/api/kpis?ticker=${ticker}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d: KpiData) => {
        setData(d)
        setError(null)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [ticker])

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 px-4">
        <Spinner className="size-5" />
        <span className="text-sm text-muted-foreground">Loading KPIs…</span>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="px-4 lg:px-6 text-sm text-destructive">
        API error: {error ?? "No data"} — is the backend running on :8000?
      </div>
    )
  }

  const priceUp = data.pct_change >= 0
  const sentimentUp = data.avg_sentiment >= 0
  const rsiStatus =
    data.rsi > 70 ? "Overbought" : data.rsi < 30 ? "Oversold" : "Neutral zone"

  const signalColor =
    data.signal.includes("BUY")
      ? "text-green-500"
      : data.signal.includes("SELL")
        ? "text-red-500"
        : "text-yellow-500"

  return (
    <div className="*:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card dark:*:data-[slot=card]:bg-card grid grid-cols-1 gap-4 px-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:shadow-xs lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
      {/* Price */}
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Price ({data.ticker})</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            ${data.price.toLocaleString("en-US", { minimumFractionDigits: 2 })}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {priceUp ? <IconTrendingUp /> : <IconTrendingDown />}
              {data.pct_change >= 0 ? "+" : ""}{data.pct_change}%
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {priceUp ? "Trending up" : "Trending down"} this week
            {priceUp ? (
              <IconTrendingUp className="size-4" />
            ) : (
              <IconTrendingDown className="size-4" />
            )}
          </div>
          <div className="text-muted-foreground">7-day price change</div>
        </CardFooter>
      </Card>

      {/* News Volume */}
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>News Volume (24h)</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {data.news_volume}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {data.news_volume > 10 ? <IconTrendingUp /> : <IconTrendingDown />}
              {data.news_volume > 10 ? "Active" : "Low"}
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {data.news_volume} articles collected
          </div>
          <div className="text-muted-foreground">
            Yahoo Finance + Finviz
          </div>
        </CardFooter>
      </Card>

      {/* Avg Sentiment */}
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Avg Sentiment</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {data.avg_sentiment >= 0 ? "+" : ""}{data.avg_sentiment.toFixed(3)}
          </CardTitle>
          <CardAction>
            <Badge variant="outline" className={signalColor}>
              {sentimentUp ? <IconTrendingUp /> : <IconTrendingDown />}
              {data.signal}
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            AI Signal: <span className={signalColor}>{data.signal}</span>
          </div>
          <div className="text-muted-foreground">
            FinBERT sentiment analysis
          </div>
        </CardFooter>
      </Card>

      {/* RSI */}
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>RSI (14)</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {data.rsi.toFixed(1)}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {data.rsi > 50 ? <IconTrendingUp /> : <IconTrendingDown />}
              {rsiStatus}
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {rsiStatus}
            {data.rsi > 50 ? (
              <IconTrendingUp className="size-4" />
            ) : (
              <IconTrendingDown className="size-4" />
            )}
          </div>
          <div className="text-muted-foreground">
            14-period Relative Strength Index
          </div>
        </CardFooter>
      </Card>
    </div>
  )
}

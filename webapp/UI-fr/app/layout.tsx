import type { Metadata } from "next"
import { Toaster } from "@/components/ui/sonner"
import "./globals.css"

export const metadata: Metadata = {
  title: "Feelow — Dashboard",
  description: "AI Agentic Sentiment × Price Intelligence",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-svh antialiased">
        {children}
        <Toaster />
      </body>
    </html>
  )
}

import type { Metadata } from "next";
import "./globals.css";
import { AppProviders } from "@/providers/app-providers";

export const metadata: Metadata = {
  title: "Smarlux Content OS",
  description: "AI-powered multilingual content generation and publishing",
  icons: {
    icon: "/logo.png",
    apple: "/logo.png",
  },
  robots: {
    index: false,
    follow: false,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fa" dir="rtl" suppressHydrationWarning>
      <body>
        <div id="toast-root" />
        <div id="app-root" className="min-h-screen bg-[rgb(var(--bg-primary))]">
          <AppProviders>{children}</AppProviders>
        </div>
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";
import { AppProviders } from "@/providers/app-providers";

export const metadata: Metadata = {
  title: "Smarlux Content OS",
  description: "AI-powered multilingual content generation and publishing",
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
    <html lang="fa" dir="rtl" className="macos-app-bg" suppressHydrationWarning>
      <body className="macos-app-bg">
        <div id="toast-root" />
        <div id="app-root" className="macos-app-bg min-h-screen">
          <AppProviders>{children}</AppProviders>
        </div>
      </body>
    </html>
  );
}

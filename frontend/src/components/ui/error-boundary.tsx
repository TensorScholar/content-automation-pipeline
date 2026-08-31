"use client";

import { Component, ErrorInfo, ReactNode } from "react";
import { Button } from "./button";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackMessage: string;
  retryLabel: string;
  resetKey?: string | number;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary] Caught rendering error:", error, info.componentStack);
  }

  componentDidUpdate(previousProps: ErrorBoundaryProps) {
    if (this.state.hasError && previousProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, error: null });
    }
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <section role="alert" className="mx-auto max-w-lg rounded-md border border-line border-s-2 border-s-danger bg-surface p-5 text-start animate-fade-in">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 shrink-0 rounded-full bg-danger" aria-hidden />
            <h2 className="text-base font-semibold text-ink">
              {this.props.fallbackTitle ?? "An error occurred"}
            </h2>
          </div>
          <p className="mt-1.5 text-sm text-ink-secondary">
            {this.props.fallbackMessage}
          </p>
          {this.state.error ? (
            <pre className="mt-3 max-h-32 overflow-auto rounded-sm bg-ink/[0.04] p-2.5 font-mono text-xs text-ink-secondary">
              {this.state.error.message}
            </pre>
          ) : null}
          <div className="mt-4 flex justify-end">
            <Button type="button" variant="primary" size="sm" onClick={this.handleRetry}>
              {this.props.retryLabel}
            </Button>
          </div>
        </section>
      );
    }
    return this.props.children;
  }
}

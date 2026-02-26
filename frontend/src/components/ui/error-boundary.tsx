"use client";

import { Component, ErrorInfo, ReactNode } from "react";

interface ErrorBoundaryProps {
    children: ReactNode;
    fallbackTitle?: string;
    fallbackMessage?: string;
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

    handleRetry = () => {
        this.setState({ hasError: false, error: null });
    };

    render() {
        if (this.state.hasError) {
            return (
                <section className="glass-card mx-auto max-w-lg p-8 text-center animate-fade-in">
                    <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-danger-subtle">
                        <span className="text-heading-lg text-danger" aria-hidden="true">!</span>
                    </div>
                    <h2 className="text-heading-md text-ink">
                        {this.props.fallbackTitle ?? "Something went wrong"}
                    </h2>
                    <p className="mt-2 text-body-md text-ink-tertiary">
                        {this.props.fallbackMessage ?? "An unexpected error occurred in this section."}
                    </p>
                    {this.state.error && (
                        <pre className="mt-4 max-h-32 overflow-auto rounded-xl bg-surface-tertiary px-4 py-3 text-start font-mono text-body-sm text-ink-secondary">
                            {this.state.error.message}
                        </pre>
                    )}
                    <button
                        type="button"
                        onClick={this.handleRetry}
                        className="mt-6 rounded-xl bg-accent px-6 py-2.5 text-body-md font-semibold text-ink-inverse transition-all duration-fast hover:bg-accent-hover"
                    >
                        Try Again
                    </button>
                </section>
            );
        }
        return this.props.children;
    }
}
